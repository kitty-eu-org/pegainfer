#include <metal_stdlib>
using namespace metal;

kernel void copy_kernel(
    const device half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    dst[gid] = src[gid];
}

// out = embed[token_id, :]
kernel void embedding_kernel(
    const device half* embed [[buffer(0)]], // [rows, cols]
    device half* out [[buffer(1)]],         // [cols]
    constant uint& token_id [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) {
        return;
    }
    out[gid] = embed[token_id * cols + gid];
}

kernel void add_kernel(
    const device half* a [[buffer(0)]],
    const device half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    float v = float(a[gid]) + float(b[gid]);
    out[gid] = half(v);
}

// Baseline scalar RMSNorm kernel for correctness-first bring-up.
kernel void rms_norm_kernel(
    const device half* x [[buffer(0)]],
    const device half* w [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    float local_sum_sq = 0.0f;
    if ((n & 3u) == 0u) {
        uint vec_n = n >> 2;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        for (uint i4 = tid; i4 < vec_n; i4 += tg_size) {
            float4 xi = float4(x4[i4]);
            local_sum_sq += dot(xi, xi);
        }
    } else {
        for (uint i = tid; i < n; i += tg_size) {
            float xi = float(x[i]);
            local_sum_sq += xi * xi;
        }
    }
    threadgroup float simd_partials[16];
    float simd_sum_sq = simd_sum(local_sum_sq);
    if (simd_lane == 0) {
        simd_partials[simd_group] = simd_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = 0.0f;
    if (simd_group == 0) {
        float v = (simd_lane < simd_groups) ? simd_partials[simd_lane] : 0.0f;
        total_sum_sq = simd_sum(v);
        if (simd_lane == 0) {
            simd_partials[0] = total_sum_sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(simd_partials[0] / ((float)n) + eps);

    if ((n & 3u) == 0u) {
        uint vec_n = n >> 2;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* w4 = reinterpret_cast<const device half4*>(w);
        device half4* out4 = reinterpret_cast<device half4*>(out);
        for (uint i4 = tid; i4 < vec_n; i4 += tg_size) {
            float4 yi = float4(x4[i4]) * inv_rms * float4(w4[i4]);
            out4[i4] = half4(yi);
        }
    } else {
        for (uint i = tid; i < n; i += tg_size) {
            float yi = float(x[i]) * inv_rms * float(w[i]);
            out[i] = half(yi);
        }
    }
}

kernel void gemv_kernel(
    const device half* a [[buffer(0)]], // [rows, cols]
    const device half* x [[buffer(1)]], // [cols]
    device half* y [[buffer(2)]],       // [rows]
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    constexpr uint ROWS_PER_GROUP = 8;
    uint row0 = tg_pos.x * ROWS_PER_GROUP + 0;
    uint row1 = tg_pos.x * ROWS_PER_GROUP + 1;
    uint row2 = tg_pos.x * ROWS_PER_GROUP + 2;
    uint row3 = tg_pos.x * ROWS_PER_GROUP + 3;
    uint row4 = tg_pos.x * ROWS_PER_GROUP + 4;
    uint row5 = tg_pos.x * ROWS_PER_GROUP + 5;
    uint row6 = tg_pos.x * ROWS_PER_GROUP + 6;
    uint row7 = tg_pos.x * ROWS_PER_GROUP + 7;
    bool active0 = row0 < rows;
    bool active1 = row1 < rows;
    bool active2 = row2 < rows;
    bool active3 = row3 < rows;
    bool active4 = row4 < rows;
    bool active5 = row5 < rows;
    bool active6 = row6 < rows;
    bool active7 = row7 < rows;
    if (!active0 && !active1 && !active2 && !active3 && !active4 && !active5 && !active6 && !active7) {
        return;
    }

    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    uint base0 = row0 * cols;
    uint base1 = row1 * cols;
    uint base2 = row2 * cols;
    uint base3 = row3 * cols;
    uint base4 = row4 * cols;
    uint base5 = row5 * cols;
    uint base6 = row6 * cols;
    uint base7 = row7 * cols;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    if ((cols & 3u) == 0u) {
        uint vec_cols = cols >> 2;
        const device half4* a4_row0 = reinterpret_cast<const device half4*>(a + base0);
        const device half4* a4_row1 = reinterpret_cast<const device half4*>(a + base1);
        const device half4* a4_row2 = reinterpret_cast<const device half4*>(a + base2);
        const device half4* a4_row3 = reinterpret_cast<const device half4*>(a + base3);
        const device half4* a4_row4 = reinterpret_cast<const device half4*>(a + base4);
        const device half4* a4_row5 = reinterpret_cast<const device half4*>(a + base5);
        const device half4* a4_row6 = reinterpret_cast<const device half4*>(a + base6);
        const device half4* a4_row7 = reinterpret_cast<const device half4*>(a + base7);
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        for (uint k4 = tid; k4 < vec_cols; k4 += tg_size) {
            float4 xv = float4(x4[k4]);
            if (active0) {
                acc0 += dot(float4(a4_row0[k4]), xv);
            }
            if (active1) {
                acc1 += dot(float4(a4_row1[k4]), xv);
            }
            if (active2) {
                acc2 += dot(float4(a4_row2[k4]), xv);
            }
            if (active3) {
                acc3 += dot(float4(a4_row3[k4]), xv);
            }
            if (active4) {
                acc4 += dot(float4(a4_row4[k4]), xv);
            }
            if (active5) {
                acc5 += dot(float4(a4_row5[k4]), xv);
            }
            if (active6) {
                acc6 += dot(float4(a4_row6[k4]), xv);
            }
            if (active7) {
                acc7 += dot(float4(a4_row7[k4]), xv);
            }
        }
    } else {
        for (uint k = tid; k < cols; k += tg_size) {
            float xv = float(x[k]);
            if (active0) {
                acc0 += float(a[base0 + k]) * xv;
            }
            if (active1) {
                acc1 += float(a[base1 + k]) * xv;
            }
            if (active2) {
                acc2 += float(a[base2 + k]) * xv;
            }
            if (active3) {
                acc3 += float(a[base3 + k]) * xv;
            }
            if (active4) {
                acc4 += float(a[base4 + k]) * xv;
            }
            if (active5) {
                acc5 += float(a[base5 + k]) * xv;
            }
            if (active6) {
                acc6 += float(a[base6 + k]) * xv;
            }
            if (active7) {
                acc7 += float(a[base7 + k]) * xv;
            }
        }
    }

    threadgroup float partial0[16];
    threadgroup float partial1[16];
    threadgroup float partial2[16];
    threadgroup float partial3[16];
    threadgroup float partial4[16];
    threadgroup float partial5[16];
    threadgroup float partial6[16];
    threadgroup float partial7[16];
    float simd_acc0 = simd_sum(acc0);
    float simd_acc1 = simd_sum(acc1);
    float simd_acc2 = simd_sum(acc2);
    float simd_acc3 = simd_sum(acc3);
    float simd_acc4 = simd_sum(acc4);
    float simd_acc5 = simd_sum(acc5);
    float simd_acc6 = simd_sum(acc6);
    float simd_acc7 = simd_sum(acc7);
    if (simd_lane == 0) {
        partial0[simd_group] = simd_acc0;
        partial1[simd_group] = simd_acc1;
        partial2[simd_group] = simd_acc2;
        partial3[simd_group] = simd_acc3;
        partial4[simd_group] = simd_acc4;
        partial5[simd_group] = simd_acc5;
        partial6[simd_group] = simd_acc6;
        partial7[simd_group] = simd_acc7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v0 = (simd_lane < simd_groups) ? partial0[simd_lane] : 0.0f;
        float v1 = (simd_lane < simd_groups) ? partial1[simd_lane] : 0.0f;
        float v2 = (simd_lane < simd_groups) ? partial2[simd_lane] : 0.0f;
        float v3 = (simd_lane < simd_groups) ? partial3[simd_lane] : 0.0f;
        float v4 = (simd_lane < simd_groups) ? partial4[simd_lane] : 0.0f;
        float v5 = (simd_lane < simd_groups) ? partial5[simd_lane] : 0.0f;
        float v6 = (simd_lane < simd_groups) ? partial6[simd_lane] : 0.0f;
        float v7 = (simd_lane < simd_groups) ? partial7[simd_lane] : 0.0f;
        float total0 = simd_sum(v0);
        float total1 = simd_sum(v1);
        float total2 = simd_sum(v2);
        float total3 = simd_sum(v3);
        float total4 = simd_sum(v4);
        float total5 = simd_sum(v5);
        float total6 = simd_sum(v6);
        float total7 = simd_sum(v7);
        if (simd_lane == 0) {
            if (active0) {
                y[row0] = half(total0);
            }
            if (active1) {
                y[row1] = half(total1);
            }
            if (active2) {
                y[row2] = half(total2);
            }
            if (active3) {
                y[row3] = half(total3);
            }
            if (active4) {
                y[row4] = half(total4);
            }
            if (active5) {
                y[row5] = half(total5);
            }
            if (active6) {
                y[row6] = half(total6);
            }
            if (active7) {
                y[row7] = half(total7);
            }
        }
    }
}

// In-place accumulate GEMV:
// y += a @ x
kernel void gemv_accum_kernel(
    const device half* a [[buffer(0)]], // [rows, cols]
    const device half* x [[buffer(1)]], // [cols]
    device half* y [[buffer(2)]],       // [rows]
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    constexpr uint ROWS_PER_GROUP = 8;
    uint row0 = tg_pos.x * ROWS_PER_GROUP + 0;
    uint row1 = tg_pos.x * ROWS_PER_GROUP + 1;
    uint row2 = tg_pos.x * ROWS_PER_GROUP + 2;
    uint row3 = tg_pos.x * ROWS_PER_GROUP + 3;
    uint row4 = tg_pos.x * ROWS_PER_GROUP + 4;
    uint row5 = tg_pos.x * ROWS_PER_GROUP + 5;
    uint row6 = tg_pos.x * ROWS_PER_GROUP + 6;
    uint row7 = tg_pos.x * ROWS_PER_GROUP + 7;
    bool active0 = row0 < rows;
    bool active1 = row1 < rows;
    bool active2 = row2 < rows;
    bool active3 = row3 < rows;
    bool active4 = row4 < rows;
    bool active5 = row5 < rows;
    bool active6 = row6 < rows;
    bool active7 = row7 < rows;
    if (!active0 && !active1 && !active2 && !active3 && !active4 && !active5 && !active6 && !active7) {
        return;
    }
    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    uint base0 = row0 * cols;
    uint base1 = row1 * cols;
    uint base2 = row2 * cols;
    uint base3 = row3 * cols;
    uint base4 = row4 * cols;
    uint base5 = row5 * cols;
    uint base6 = row6 * cols;
    uint base7 = row7 * cols;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    if ((cols & 3u) == 0u) {
        uint vec_cols = cols >> 2;
        const device half4* a4_row0 = reinterpret_cast<const device half4*>(a + base0);
        const device half4* a4_row1 = reinterpret_cast<const device half4*>(a + base1);
        const device half4* a4_row2 = reinterpret_cast<const device half4*>(a + base2);
        const device half4* a4_row3 = reinterpret_cast<const device half4*>(a + base3);
        const device half4* a4_row4 = reinterpret_cast<const device half4*>(a + base4);
        const device half4* a4_row5 = reinterpret_cast<const device half4*>(a + base5);
        const device half4* a4_row6 = reinterpret_cast<const device half4*>(a + base6);
        const device half4* a4_row7 = reinterpret_cast<const device half4*>(a + base7);
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        for (uint k4 = tid; k4 < vec_cols; k4 += tg_size) {
            float4 xv = float4(x4[k4]);
            if (active0) {
                acc0 += dot(float4(a4_row0[k4]), xv);
            }
            if (active1) {
                acc1 += dot(float4(a4_row1[k4]), xv);
            }
            if (active2) {
                acc2 += dot(float4(a4_row2[k4]), xv);
            }
            if (active3) {
                acc3 += dot(float4(a4_row3[k4]), xv);
            }
            if (active4) {
                acc4 += dot(float4(a4_row4[k4]), xv);
            }
            if (active5) {
                acc5 += dot(float4(a4_row5[k4]), xv);
            }
            if (active6) {
                acc6 += dot(float4(a4_row6[k4]), xv);
            }
            if (active7) {
                acc7 += dot(float4(a4_row7[k4]), xv);
            }
        }
    } else {
        for (uint k = tid; k < cols; k += tg_size) {
            float xv = float(x[k]);
            if (active0) {
                acc0 += float(a[base0 + k]) * xv;
            }
            if (active1) {
                acc1 += float(a[base1 + k]) * xv;
            }
            if (active2) {
                acc2 += float(a[base2 + k]) * xv;
            }
            if (active3) {
                acc3 += float(a[base3 + k]) * xv;
            }
            if (active4) {
                acc4 += float(a[base4 + k]) * xv;
            }
            if (active5) {
                acc5 += float(a[base5 + k]) * xv;
            }
            if (active6) {
                acc6 += float(a[base6 + k]) * xv;
            }
            if (active7) {
                acc7 += float(a[base7 + k]) * xv;
            }
        }
    }

    threadgroup float partial0[16];
    threadgroup float partial1[16];
    threadgroup float partial2[16];
    threadgroup float partial3[16];
    threadgroup float partial4[16];
    threadgroup float partial5[16];
    threadgroup float partial6[16];
    threadgroup float partial7[16];
    float simd_acc0 = simd_sum(acc0);
    float simd_acc1 = simd_sum(acc1);
    float simd_acc2 = simd_sum(acc2);
    float simd_acc3 = simd_sum(acc3);
    float simd_acc4 = simd_sum(acc4);
    float simd_acc5 = simd_sum(acc5);
    float simd_acc6 = simd_sum(acc6);
    float simd_acc7 = simd_sum(acc7);
    if (simd_lane == 0) {
        partial0[simd_group] = simd_acc0;
        partial1[simd_group] = simd_acc1;
        partial2[simd_group] = simd_acc2;
        partial3[simd_group] = simd_acc3;
        partial4[simd_group] = simd_acc4;
        partial5[simd_group] = simd_acc5;
        partial6[simd_group] = simd_acc6;
        partial7[simd_group] = simd_acc7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v0 = (simd_lane < simd_groups) ? partial0[simd_lane] : 0.0f;
        float v1 = (simd_lane < simd_groups) ? partial1[simd_lane] : 0.0f;
        float v2 = (simd_lane < simd_groups) ? partial2[simd_lane] : 0.0f;
        float v3 = (simd_lane < simd_groups) ? partial3[simd_lane] : 0.0f;
        float v4 = (simd_lane < simd_groups) ? partial4[simd_lane] : 0.0f;
        float v5 = (simd_lane < simd_groups) ? partial5[simd_lane] : 0.0f;
        float v6 = (simd_lane < simd_groups) ? partial6[simd_lane] : 0.0f;
        float v7 = (simd_lane < simd_groups) ? partial7[simd_lane] : 0.0f;
        float total0 = simd_sum(v0);
        float total1 = simd_sum(v1);
        float total2 = simd_sum(v2);
        float total3 = simd_sum(v3);
        float total4 = simd_sum(v4);
        float total5 = simd_sum(v5);
        float total6 = simd_sum(v6);
        float total7 = simd_sum(v7);
        if (simd_lane == 0) {
            if (active0) {
                y[row0] = half(float(y[row0]) + total0);
            }
            if (active1) {
                y[row1] = half(float(y[row1]) + total1);
            }
            if (active2) {
                y[row2] = half(float(y[row2]) + total2);
            }
            if (active3) {
                y[row3] = half(float(y[row3]) + total3);
            }
            if (active4) {
                y[row4] = half(float(y[row4]) + total4);
            }
            if (active5) {
                y[row5] = half(float(y[row5]) + total5);
            }
            if (active6) {
                y[row6] = half(float(y[row6]) + total6);
            }
            if (active7) {
                y[row7] = half(float(y[row7]) + total7);
            }
        }
    }
}

// Fused two-way GEMV:
// y0 = a0 @ x, y1 = a1 @ x
kernel void gemv2_kernel(
    const device half* a0 [[buffer(0)]], // [rows0, cols]
    const device half* x [[buffer(1)]],  // [cols]
    const device half* a1 [[buffer(2)]], // [rows1, cols]
    device half* y0 [[buffer(3)]],       // [rows0]
    device half* y1 [[buffer(4)]],       // [rows1]
    constant uint& rows0 [[buffer(5)]],
    constant uint& rows1 [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint row = tg_pos.x;
    uint max_rows = max(rows0, rows1);
    if (row >= max_rows) {
        return;
    }

    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    bool active0 = row < rows0;
    bool active1 = row < rows1;
    uint base0 = row * cols;
    uint base1 = row * cols;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if ((cols & 3u) == 0u) {
        uint vec_cols = cols >> 2;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* a0_row = reinterpret_cast<const device half4*>(a0 + base0);
        const device half4* a1_row = reinterpret_cast<const device half4*>(a1 + base1);
        for (uint k4 = tid; k4 < vec_cols; k4 += tg_size) {
            float4 xv = float4(x4[k4]);
            if (active0) {
                acc0 += dot(float4(a0_row[k4]), xv);
            }
            if (active1) {
                acc1 += dot(float4(a1_row[k4]), xv);
            }
        }
    } else {
        for (uint k = tid; k < cols; k += tg_size) {
            float xv = float(x[k]);
            if (active0) {
                acc0 += float(a0[base0 + k]) * xv;
            }
            if (active1) {
                acc1 += float(a1[base1 + k]) * xv;
            }
        }
    }

    threadgroup float partial0[16];
    threadgroup float partial1[16];
    float simd_acc0 = simd_sum(acc0);
    float simd_acc1 = simd_sum(acc1);
    if (simd_lane == 0) {
        partial0[simd_group] = simd_acc0;
        partial1[simd_group] = simd_acc1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v0 = (simd_lane < simd_groups) ? partial0[simd_lane] : 0.0f;
        float v1 = (simd_lane < simd_groups) ? partial1[simd_lane] : 0.0f;
        float total0 = simd_sum(v0);
        float total1 = simd_sum(v1);
        if (simd_lane == 0) {
            if (active0) {
                y0[row] = half(total0);
            }
            if (active1) {
                y1[row] = half(total1);
            }
        }
    }
}

// Fused three-way GEMV:
// y0 = a0 @ x, y1 = a1 @ x, y2 = a2 @ x
kernel void gemv3_kernel(
    const device half* a0 [[buffer(0)]], // [rows0, cols]
    const device half* x [[buffer(1)]],  // [cols]
    const device half* a1 [[buffer(2)]], // [rows1, cols]
    const device half* a2 [[buffer(3)]], // [rows2, cols]
    device half* y0 [[buffer(4)]],       // [rows0]
    device half* y1 [[buffer(5)]],       // [rows1]
    device half* y2 [[buffer(6)]],       // [rows2]
    constant uint& rows0 [[buffer(7)]],
    constant uint& rows1 [[buffer(8)]],
    constant uint& rows2 [[buffer(9)]],
    constant uint& cols [[buffer(10)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint row = tg_pos.x;
    uint max_rows = max(rows0, max(rows1, rows2));
    if (row >= max_rows) {
        return;
    }

    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    bool active0 = row < rows0;
    bool active1 = row < rows1;
    bool active2 = row < rows2;
    uint base0 = row * cols;
    uint base1 = row * cols;
    uint base2 = row * cols;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    if ((cols & 3u) == 0u) {
        uint vec_cols = cols >> 2;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* a0_row = reinterpret_cast<const device half4*>(a0 + base0);
        const device half4* a1_row = reinterpret_cast<const device half4*>(a1 + base1);
        const device half4* a2_row = reinterpret_cast<const device half4*>(a2 + base2);
        for (uint k4 = tid; k4 < vec_cols; k4 += tg_size) {
            float4 xv = float4(x4[k4]);
            if (active0) {
                acc0 += dot(float4(a0_row[k4]), xv);
            }
            if (active1) {
                acc1 += dot(float4(a1_row[k4]), xv);
            }
            if (active2) {
                acc2 += dot(float4(a2_row[k4]), xv);
            }
        }
    } else {
        for (uint k = tid; k < cols; k += tg_size) {
            float xv = float(x[k]);
            if (active0) {
                acc0 += float(a0[base0 + k]) * xv;
            }
            if (active1) {
                acc1 += float(a1[base1 + k]) * xv;
            }
            if (active2) {
                acc2 += float(a2[base2 + k]) * xv;
            }
        }
    }

    threadgroup float partial0[16];
    threadgroup float partial1[16];
    threadgroup float partial2[16];
    float simd_acc0 = simd_sum(acc0);
    float simd_acc1 = simd_sum(acc1);
    float simd_acc2 = simd_sum(acc2);
    if (simd_lane == 0) {
        partial0[simd_group] = simd_acc0;
        partial1[simd_group] = simd_acc1;
        partial2[simd_group] = simd_acc2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v0 = (simd_lane < simd_groups) ? partial0[simd_lane] : 0.0f;
        float v1 = (simd_lane < simd_groups) ? partial1[simd_lane] : 0.0f;
        float v2 = (simd_lane < simd_groups) ? partial2[simd_lane] : 0.0f;
        float total0 = simd_sum(v0);
        float total1 = simd_sum(v1);
        float total2 = simd_sum(v2);
        if (simd_lane == 0) {
            if (active0) {
                y0[row] = half(total0);
            }
            if (active1) {
                y1[row] = half(total1);
            }
            if (active2) {
                y2[row] = half(total2);
            }
        }
    }
}

// Y[token, out] = sum_k X[token, k] * W[out, k]
kernel void gemm_kernel(
    const device half* w [[buffer(0)]], // [out_dim, in_dim]
    const device half* x [[buffer(1)]], // [seq_len, in_dim]
    device half* y [[buffer(2)]],       // [seq_len, out_dim]
    constant uint& out_dim [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& in_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_i = gid.x;
    uint token = gid.y;

    if (out_i >= out_dim || token >= seq_len) {
        return;
    }

    float acc = 0.0f;
    uint w_base = out_i * in_dim;
    uint x_base = token * in_dim;
    for (uint k = 0; k < in_dim; ++k) {
        acc += float(w[w_base + k]) * float(x[x_base + k]);
    }
    y[token * out_dim + out_i] = half(acc);
}

kernel void silu_mul_kernel(
    const device half* gate [[buffer(0)]],
    const device half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }
    float x = float(gate[gid]);
    float silu = x / (1.0f + exp(-x));
    float u = float(up[gid]);
    out[gid] = half(silu * u);
}

// Baseline scalar argmax kernel for greedy decode.
kernel void argmax_kernel(
    const device half* x [[buffer(0)]],
    device int* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }
    float best = -INFINITY;
    uint best_idx = 0;
    for (uint i = 0; i < n; ++i) {
        float v = float(x[i]);
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    out[0] = (int)best_idx;
}

// Stage-1 parallel argmax: one threadgroup reduces one chunk of logits.
kernel void argmax_stage1_kernel(
    const device half* x [[buffer(0)]],
    device float* partial_vals [[buffer(1)]],
    device uint* partial_idxs [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    uint group_id = tg_pos.x;
    uint idx = group_id * tg_size + tid;

    float best = -INFINITY;
    uint best_idx = 0;
    if (idx < n) {
        best = float(x[idx]);
        best_idx = idx;
    }

    threadgroup float vals[256];
    threadgroup uint idxs[256];
    vals[tid] = best;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v2 = vals[tid + stride];
            uint i2 = idxs[tid + stride];
            if (v2 > vals[tid] || (v2 == vals[tid] && i2 < idxs[tid])) {
                vals[tid] = v2;
                idxs[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_vals[group_id] = vals[0];
        partial_idxs[group_id] = idxs[0];
    }
}

// Stage-1 fused lm_head GEMV + argmax:
// computes argmax(weight @ x) without materializing full logits.
kernel void gemv_argmax_stage1_kernel(
    const device half* a [[buffer(0)]], // [rows, cols]
    const device half* x [[buffer(1)]], // [cols]
    device float* partial_vals [[buffer(2)]],
    device uint* partial_idxs [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    constexpr uint ROWS_PER_GROUP = 8;
    uint row0 = tg_pos.x * ROWS_PER_GROUP + 0;
    uint row1 = tg_pos.x * ROWS_PER_GROUP + 1;
    uint row2 = tg_pos.x * ROWS_PER_GROUP + 2;
    uint row3 = tg_pos.x * ROWS_PER_GROUP + 3;
    uint row4 = tg_pos.x * ROWS_PER_GROUP + 4;
    uint row5 = tg_pos.x * ROWS_PER_GROUP + 5;
    uint row6 = tg_pos.x * ROWS_PER_GROUP + 6;
    uint row7 = tg_pos.x * ROWS_PER_GROUP + 7;
    bool active0 = row0 < rows;
    bool active1 = row1 < rows;
    bool active2 = row2 < rows;
    bool active3 = row3 < rows;
    bool active4 = row4 < rows;
    bool active5 = row5 < rows;
    bool active6 = row6 < rows;
    bool active7 = row7 < rows;
    if (!active0 && !active1 && !active2 && !active3 && !active4 && !active5 && !active6 && !active7) {
        return;
    }

    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    uint base0 = row0 * cols;
    uint base1 = row1 * cols;
    uint base2 = row2 * cols;
    uint base3 = row3 * cols;
    uint base4 = row4 * cols;
    uint base5 = row5 * cols;
    uint base6 = row6 * cols;
    uint base7 = row7 * cols;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    if ((cols & 3u) == 0u) {
        uint vec_cols = cols >> 2;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* a4_row0 = reinterpret_cast<const device half4*>(a + base0);
        const device half4* a4_row1 = reinterpret_cast<const device half4*>(a + base1);
        const device half4* a4_row2 = reinterpret_cast<const device half4*>(a + base2);
        const device half4* a4_row3 = reinterpret_cast<const device half4*>(a + base3);
        const device half4* a4_row4 = reinterpret_cast<const device half4*>(a + base4);
        const device half4* a4_row5 = reinterpret_cast<const device half4*>(a + base5);
        const device half4* a4_row6 = reinterpret_cast<const device half4*>(a + base6);
        const device half4* a4_row7 = reinterpret_cast<const device half4*>(a + base7);
        for (uint k4 = tid; k4 < vec_cols; k4 += tg_size) {
            float4 xv = float4(x4[k4]);
            if (active0) {
                acc0 += dot(float4(a4_row0[k4]), xv);
            }
            if (active1) {
                acc1 += dot(float4(a4_row1[k4]), xv);
            }
            if (active2) {
                acc2 += dot(float4(a4_row2[k4]), xv);
            }
            if (active3) {
                acc3 += dot(float4(a4_row3[k4]), xv);
            }
            if (active4) {
                acc4 += dot(float4(a4_row4[k4]), xv);
            }
            if (active5) {
                acc5 += dot(float4(a4_row5[k4]), xv);
            }
            if (active6) {
                acc6 += dot(float4(a4_row6[k4]), xv);
            }
            if (active7) {
                acc7 += dot(float4(a4_row7[k4]), xv);
            }
        }
    } else {
        for (uint k = tid; k < cols; k += tg_size) {
            float xv = float(x[k]);
            if (active0) {
                acc0 += float(a[base0 + k]) * xv;
            }
            if (active1) {
                acc1 += float(a[base1 + k]) * xv;
            }
            if (active2) {
                acc2 += float(a[base2 + k]) * xv;
            }
            if (active3) {
                acc3 += float(a[base3 + k]) * xv;
            }
            if (active4) {
                acc4 += float(a[base4 + k]) * xv;
            }
            if (active5) {
                acc5 += float(a[base5 + k]) * xv;
            }
            if (active6) {
                acc6 += float(a[base6 + k]) * xv;
            }
            if (active7) {
                acc7 += float(a[base7 + k]) * xv;
            }
        }
    }

    threadgroup float partial0[16];
    threadgroup float partial1[16];
    threadgroup float partial2[16];
    threadgroup float partial3[16];
    threadgroup float partial4[16];
    threadgroup float partial5[16];
    threadgroup float partial6[16];
    threadgroup float partial7[16];
    float simd_acc0 = simd_sum(acc0);
    float simd_acc1 = simd_sum(acc1);
    float simd_acc2 = simd_sum(acc2);
    float simd_acc3 = simd_sum(acc3);
    float simd_acc4 = simd_sum(acc4);
    float simd_acc5 = simd_sum(acc5);
    float simd_acc6 = simd_sum(acc6);
    float simd_acc7 = simd_sum(acc7);
    if (simd_lane == 0) {
        partial0[simd_group] = simd_acc0;
        partial1[simd_group] = simd_acc1;
        partial2[simd_group] = simd_acc2;
        partial3[simd_group] = simd_acc3;
        partial4[simd_group] = simd_acc4;
        partial5[simd_group] = simd_acc5;
        partial6[simd_group] = simd_acc6;
        partial7[simd_group] = simd_acc7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v0 = (simd_lane < simd_groups) ? partial0[simd_lane] : 0.0f;
        float v1 = (simd_lane < simd_groups) ? partial1[simd_lane] : 0.0f;
        float v2 = (simd_lane < simd_groups) ? partial2[simd_lane] : 0.0f;
        float v3 = (simd_lane < simd_groups) ? partial3[simd_lane] : 0.0f;
        float v4 = (simd_lane < simd_groups) ? partial4[simd_lane] : 0.0f;
        float v5 = (simd_lane < simd_groups) ? partial5[simd_lane] : 0.0f;
        float v6 = (simd_lane < simd_groups) ? partial6[simd_lane] : 0.0f;
        float v7 = (simd_lane < simd_groups) ? partial7[simd_lane] : 0.0f;
        float total0 = simd_sum(v0);
        float total1 = simd_sum(v1);
        float total2 = simd_sum(v2);
        float total3 = simd_sum(v3);
        float total4 = simd_sum(v4);
        float total5 = simd_sum(v5);
        float total6 = simd_sum(v6);
        float total7 = simd_sum(v7);
        if (simd_lane == 0) {
            float best = -INFINITY;
            uint best_idx = 0;
            if (active0) {
                best = total0;
                best_idx = row0;
            }
            if (active1 && (total1 > best || (total1 == best && row1 < best_idx))) {
                best = total1;
                best_idx = row1;
            }
            if (active2 && (total2 > best || (total2 == best && row2 < best_idx))) {
                best = total2;
                best_idx = row2;
            }
            if (active3 && (total3 > best || (total3 == best && row3 < best_idx))) {
                best = total3;
                best_idx = row3;
            }
            if (active4 && (total4 > best || (total4 == best && row4 < best_idx))) {
                best = total4;
                best_idx = row4;
            }
            if (active5 && (total5 > best || (total5 == best && row5 < best_idx))) {
                best = total5;
                best_idx = row5;
            }
            if (active6 && (total6 > best || (total6 == best && row6 < best_idx))) {
                best = total6;
                best_idx = row6;
            }
            if (active7 && (total7 > best || (total7 == best && row7 < best_idx))) {
                best = total7;
                best_idx = row7;
            }
            partial_vals[tg_pos.x] = best;
            partial_idxs[tg_pos.x] = best_idx;
        }
    }
}

// Stage-2 argmax over partial results.
kernel void argmax_stage2_kernel(
    const device float* partial_vals [[buffer(0)]],
    const device uint* partial_idxs [[buffer(1)]],
    device int* out [[buffer(2)]],
    constant uint& num_partials [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    float best = -INFINITY;
    uint best_idx = 0;
    for (uint i = tid; i < num_partials; i += tg_size) {
        float v = partial_vals[i];
        uint idx = partial_idxs[i];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }

    threadgroup float vals[256];
    threadgroup uint idxs[256];
    vals[tid] = best;
    idxs[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v2 = vals[tid + stride];
            uint i2 = idxs[tid + stride];
            if (v2 > vals[tid] || (v2 == vals[tid] && i2 < idxs[tid])) {
                vals[tid] = v2;
                idxs[tid] = i2;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[0] = (int)idxs[0];
    }
}

// Prepare KV cache for current decode token:
// - K: per-KV-head RMSNorm + RoPE, then write into K cache at [head, pos, :]
// - V: copy raw V projection into V cache at [head, dim, pos]
kernel void prepare_kv_decode_kernel(
    const device half* k_full [[buffer(0)]],
    const device half* v_full [[buffer(1)]],
    const device half* k_norm [[buffer(2)]],
    const device half* cos_cache [[buffer(3)]],
    const device half* sin_cache [[buffer(4)]],
    device half* k_cache [[buffer(5)]],
    device half* v_cache [[buffer(6)]],
    constant uint& num_kv_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& pos [[buffer(9)]],
    constant uint& max_seq_len [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint kv_head [[thread_position_in_grid]]
) {
    if (kv_head >= num_kv_heads || head_dim == 0 || (head_dim & 1u) != 0) {
        return;
    }

    uint in_base = kv_head * head_dim;
    uint k_cache_base = (kv_head * max_seq_len + pos) * head_dim;
    uint v_cache_head_base = kv_head * head_dim * max_seq_len;
    uint rope_base = pos * head_dim;
    uint half_dim = head_dim / 2;

    float sum_sq = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        float kv = float(k_full[in_base + d]);
        sum_sq += kv * kv;
    }
    float inv_rms = rsqrt(sum_sq / ((float)head_dim) + eps);

    for (uint i = 0; i < half_dim; ++i) {
        float x0 = float(k_full[in_base + i]) * inv_rms * float(k_norm[i]);
        float x1 =
            float(k_full[in_base + half_dim + i]) * inv_rms * float(k_norm[half_dim + i]);
        float c = float(cos_cache[rope_base + i]);
        float s = float(sin_cache[rope_base + i]);
        k_cache[k_cache_base + i] = half(x0 * c - x1 * s);
        k_cache[k_cache_base + half_dim + i] = half(x0 * s + x1 * c);
    }

    for (uint d = 0; d < head_dim; ++d) {
        v_cache[v_cache_head_base + d * max_seq_len + pos] = v_full[in_base + d];
    }
}

constant uint MAX_DECODE_HEAD_DIM = 256;
constant uint MAX_DECODE_SEQ = 4096;

// Decode attention for all Q heads.
// One thread handles one Q head and computes:
// Q RMSNorm+RoPE -> scores -> softmax -> weighted sum over V cache.
kernel void attention_decode_heads_kernel(
    const device half* q_full [[buffer(0)]],
    const device half* q_norm [[buffer(1)]],
    const device half* cos_cache [[buffer(2)]],
    const device half* sin_cache [[buffer(3)]],
    const device half* k_cache [[buffer(4)]],
    const device half* v_cache [[buffer(5)]],
    device half* out [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint& gqa_ratio [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& pos [[buffer(11)]],
    constant uint& seq_len [[buffer(12)]],
    constant uint& max_seq_len [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    constant float& eps [[buffer(15)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint q_head = tg_pos.x;
    if (q_head >= num_heads || num_kv_heads == 0 || gqa_ratio == 0 || head_dim == 0) {
        return;
    }
    if (head_dim > MAX_DECODE_HEAD_DIM || (head_dim & 1u) != 0 || seq_len == 0 || seq_len > MAX_DECODE_SEQ) {
        return;
    }
    uint tg_size = threads_per_group.x;
    if (tg_size == 0 || tg_size > 256) {
        return;
    }

    uint q_base = q_head * head_dim;
    uint rope_base = pos * head_dim;
    uint half_dim = head_dim / 2;

    threadgroup float q_tg[MAX_DECODE_HEAD_DIM];
    if (tid == 0) {
        float q_local[MAX_DECODE_HEAD_DIM];
        float sum_sq = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q_full[q_base + d]);
            q_local[d] = qv;
            sum_sq += qv * qv;
        }
        float inv_rms = rsqrt(sum_sq / ((float)head_dim) + eps);
        for (uint d = 0; d < head_dim; ++d) {
            q_local[d] *= inv_rms * float(q_norm[d]);
        }

        for (uint i = 0; i < half_dim; ++i) {
            float x0 = q_local[i];
            float x1 = q_local[half_dim + i];
            float c = float(cos_cache[rope_base + i]);
            float s = float(sin_cache[rope_base + i]);
            q_local[i] = x0 * c - x1 * s;
            q_local[half_dim + i] = x0 * s + x1 * c;
        }

        for (uint d = 0; d < head_dim; ++d) {
            q_tg[d] = q_local[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint kv_head = q_head / gqa_ratio;
    uint kv_base = kv_head * max_seq_len * head_dim;
    uint v_head_base = kv_head * head_dim * max_seq_len;
    uint vec_head_dim = head_dim >> 2;
    uint tail_start = vec_head_dim << 2;
    threadgroup half weights[MAX_DECODE_SEQ];

    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += tg_size) {
        uint k_base = kv_base + t * head_dim;
        float score_dot = 0.0f;
        for (uint d4 = 0; d4 < vec_head_dim; ++d4) {
            uint d = d4 << 2;
            float4 qv = float4(q_tg[d], q_tg[d + 1], q_tg[d + 2], q_tg[d + 3]);
            const device half4* k4 = reinterpret_cast<const device half4*>(k_cache + k_base + d);
            score_dot += dot(qv, float4(k4[0]));
        }
        for (uint d = tail_start; d < head_dim; ++d) {
            score_dot += q_tg[d] * float(k_cache[k_base + d]);
        }
        float score = score_dot * scale;
        weights[t] = half(score);
        local_max = max(local_max, score);
    }

    threadgroup float partial_max[16];
    float simd_max_val = simd_max(local_max);
    if (simd_lane == 0) {
        partial_max[simd_group] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v = (simd_lane < simd_groups) ? partial_max[simd_lane] : -INFINITY;
        float m = simd_max(v);
        if (simd_lane == 0) {
            partial_max[0] = m;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = partial_max[0];

    float local_sum_exp = 0.0f;
    for (uint t = tid; t < seq_len; t += tg_size) {
        float score = float(weights[t]);
        float w = fast::exp(score - max_score);
        weights[t] = half(w);
        local_sum_exp += w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float partial_sum[16];
    float simd_sum_val = simd_sum(local_sum_exp);
    if (simd_lane == 0) {
        partial_sum[simd_group] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float v = (simd_lane < simd_groups) ? partial_sum[simd_lane] : 0.0f;
        float s = simd_sum(v);
        if (simd_lane == 0) {
            partial_sum[0] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum_exp = 1.0f / max(partial_sum[0], 1e-20f);

    if ((head_dim & 3u) == 0u) {
        uint vec_head_dim = head_dim >> 2;
        device half4* out4 = reinterpret_cast<device half4*>(out + q_base);
        for (uint d4 = tid; d4 < vec_head_dim; d4 += tg_size) {
            float4 acc = float4(0.0f);
            uint d = d4 << 2;
            for (uint t = 0; t < seq_len; ++t) {
                float w = float(weights[t]);
                uint d0_base = v_head_base + d * max_seq_len;
                uint d1_base = d0_base + max_seq_len;
                uint d2_base = d1_base + max_seq_len;
                uint d3_base = d2_base + max_seq_len;
                acc.x += w * float(v_cache[d0_base + t]);
                acc.y += w * float(v_cache[d1_base + t]);
                acc.z += w * float(v_cache[d2_base + t]);
                acc.w += w * float(v_cache[d3_base + t]);
            }
            out4[d4] = half4(acc * inv_sum_exp);
        }
    } else {
        for (uint d = tid; d < head_dim; d += tg_size) {
            float acc = 0.0f;
            uint v_base = v_head_base + d * max_seq_len;
            for (uint t = 0; t < seq_len; ++t) {
                acc += float(weights[t]) * float(v_cache[v_base + t]);
            }
            out[q_base + d] = half(acc * inv_sum_exp);
        }
    }
}
