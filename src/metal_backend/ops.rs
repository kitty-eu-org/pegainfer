//! Baseline Metal ops for backend bring-up.

use anyhow::{bail, Result};
use metal::{ComputePipelineStateRef, MTLResourceOptions, MTLSize};

use crate::metal_backend::tensor::{
    MetalDeviceContext, MetalDeviceMatrix, MetalDeviceVec, MetalHiddenStates,
};

fn dispatch_1d_encoded<F>(
    pipeline: &ComputePipelineStateRef,
    total_threads: usize,
    encoder: &metal::ComputeCommandEncoderRef,
    bind: F,
) where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    if total_threads == 0 {
        return;
    }

    encoder.set_compute_pipeline_state(pipeline);
    bind(encoder);

    let thread_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let threads_per_group = std::cmp::min(thread_width, max_threads);
    let threads_per_group = std::cmp::min(threads_per_group, total_threads as u64).max(1);

    encoder.dispatch_threads(
        MTLSize::new(total_threads as u64, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
}

fn dispatch_1d<F>(
    ctx: &MetalDeviceContext,
    pipeline: &ComputePipelineStateRef,
    total_threads: usize,
    bind: F,
) -> Result<()>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    dispatch_1d_encoded(pipeline, total_threads, &encoder, bind);
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

fn dispatch_2d_encoded<F>(
    pipeline: &ComputePipelineStateRef,
    width: usize,
    height: usize,
    encoder: &metal::ComputeCommandEncoderRef,
    bind: F,
) where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    if width == 0 || height == 0 {
        return;
    }

    encoder.set_compute_pipeline_state(pipeline);
    bind(encoder);

    let thread_width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let threads_per_group = std::cmp::min(thread_width, max_threads);
    let threads_per_group = std::cmp::min(threads_per_group, width as u64).max(1);

    encoder.dispatch_threads(
        MTLSize::new(width as u64, height as u64, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
}

fn dispatch_2d<F>(
    ctx: &MetalDeviceContext,
    pipeline: &ComputePipelineStateRef,
    width: usize,
    height: usize,
    bind: F,
) -> Result<()>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef),
{
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    dispatch_2d_encoded(pipeline, width, height, &encoder, bind);
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

fn largest_power_of_two_leq(x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    1usize << (usize::BITS as usize - 1 - x.leading_zeros() as usize)
}

pub fn copy_into(
    ctx: &MetalDeviceContext,
    src: &MetalDeviceVec,
    dst: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(
        src.len, dst.len,
        "src len {} != dst len {}",
        src.len, dst.len
    );
    let n = src.len as u32;
    let pipeline = ctx.pipeline("copy_kernel")?;
    dispatch_1d(ctx, &pipeline, src.len, |encoder| {
        encoder.set_buffer(0, Some(&src.buffer), 0);
        encoder.set_buffer(1, Some(&dst.buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&n as *const u32).cast(),
        );
    })
}

pub fn embedding_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    embed: &MetalDeviceMatrix,
    token_id: u32,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    if out.len != embed.cols {
        bail!("out len {} != embed cols {}", out.len, embed.cols);
    }
    if token_id as usize >= embed.rows {
        bail!("token_id {} out of range [0, {})", token_id, embed.rows);
    }

    let cols_u32 = embed.cols as u32;
    let pipeline = ctx.pipeline("embedding_kernel")?;
    dispatch_1d_encoded(&pipeline, embed.cols, encoder, |enc| {
        enc.set_buffer(0, Some(&embed.buffer), 0);
        enc.set_buffer(1, Some(&out.buffer), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&token_id as *const u32).cast(),
        );
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&cols_u32 as *const u32).cast(),
        );
    });
    Ok(())
}

pub fn embedding_into(
    ctx: &MetalDeviceContext,
    embed: &MetalDeviceMatrix,
    token_id: u32,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    embedding_into_encoded(ctx, &encoder, embed, token_id, out)?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

pub fn add_inplace_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    a: &mut MetalDeviceVec,
    b: &MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(a.len, b.len, "a len {} != b len {}", a.len, b.len);
    let n = a.len as u32;
    let pipeline = ctx.pipeline("add_kernel")?;
    dispatch_1d_encoded(&pipeline, a.len, encoder, |enc| {
        enc.set_buffer(0, Some(&a.buffer), 0);
        enc.set_buffer(1, Some(&b.buffer), 0);
        enc.set_buffer(2, Some(&a.buffer), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&n as *const u32).cast(),
        );
    });
    Ok(())
}

pub fn add_inplace(
    ctx: &MetalDeviceContext,
    a: &mut MetalDeviceVec,
    b: &MetalDeviceVec,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    add_inplace_encoded(ctx, &encoder, a, b)?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

pub fn add(
    ctx: &MetalDeviceContext,
    a: &MetalDeviceVec,
    b: &MetalDeviceVec,
) -> Result<MetalDeviceVec> {
    debug_assert_eq!(a.len, b.len, "a len {} != b len {}", a.len, b.len);
    let out = MetalDeviceVec::zeros(ctx, a.len)?;
    let n = a.len as u32;
    let pipeline = ctx.pipeline("add_kernel")?;
    dispatch_1d(ctx, &pipeline, a.len, |encoder| {
        encoder.set_buffer(0, Some(&a.buffer), 0);
        encoder.set_buffer(1, Some(&b.buffer), 0);
        encoder.set_buffer(2, Some(&out.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&n as *const u32).cast(),
        );
    })?;
    Ok(out)
}

pub fn rms_norm_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    x: &MetalDeviceVec,
    weight: &MetalDeviceVec,
    eps: f32,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(x.len, out.len, "x len {} != out len {}", x.len, out.len);
    debug_assert_eq!(
        x.len, weight.len,
        "x len {} != weight len {}",
        x.len, weight.len
    );

    let n = x.len as u32;
    let pipeline = ctx.pipeline("rms_norm_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 256);
    let tg_candidate = std::cmp::min(tg_cap, x.len.max(1));
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&x.buffer), 0);
    encoder.set_buffer(1, Some(&weight.buffer), 0);
    encoder.set_buffer(2, Some(&out.buffer), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&n as *const u32).cast(),
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<f32>() as u64,
        (&eps as *const f32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

pub fn rms_norm_into(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    weight: &MetalDeviceVec,
    eps: f32,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    rms_norm_into_encoded(ctx, &encoder, x, weight, eps, out)?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

pub fn rms_norm(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    weight: &MetalDeviceVec,
    eps: f32,
) -> Result<MetalDeviceVec> {
    let mut out = MetalDeviceVec::zeros(ctx, x.len)?;
    rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

/// Matrix-vector multiplication: y = A @ x
/// A: (rows, cols) row-major, x: (cols,), y: (rows,)
pub fn gemv_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    a: &MetalDeviceMatrix,
    x: &MetalDeviceVec,
    y: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    debug_assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    let rows = a.rows as u32;
    let cols = a.cols as u32;
    let pipeline = ctx.pipeline("gemv_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 256);
    let tg_candidate = std::cmp::min(tg_cap, a.cols.max(1));
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);
    let row_groups = a.rows.div_ceil(8);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&a.buffer), 0);
    encoder.set_buffer(1, Some(&x.buffer), 0);
    encoder.set_buffer(2, Some(&y.buffer), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&rows as *const u32).cast(),
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        (&cols as *const u32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(row_groups as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

pub fn gemv(
    ctx: &MetalDeviceContext,
    a: &MetalDeviceMatrix,
    x: &MetalDeviceVec,
    y: &mut MetalDeviceVec,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    gemv_encoded(ctx, &encoder, a, x, y)?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

/// Linear layer: y = weight @ x
pub fn linear(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    weight: &MetalDeviceMatrix,
) -> Result<MetalDeviceVec> {
    let mut y = MetalDeviceVec::zeros(ctx, weight.rows)?;
    linear_into(ctx, x, weight, &mut y)?;
    Ok(y)
}

/// Linear layer into pre-allocated output: y = weight @ x
pub fn linear_into(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    weight: &MetalDeviceMatrix,
    y: &mut MetalDeviceVec,
) -> Result<()> {
    gemv(ctx, weight, x, y)
}

pub fn linear_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    x: &MetalDeviceVec,
    weight: &MetalDeviceMatrix,
    y: &mut MetalDeviceVec,
) -> Result<()> {
    gemv_encoded(ctx, encoder, weight, x, y)
}

pub fn linear2_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    x: &MetalDeviceVec,
    w0: &MetalDeviceMatrix,
    w1: &MetalDeviceMatrix,
    y0: &mut MetalDeviceVec,
    y1: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(w0.cols, x.len, "w0 cols {} != x len {}", w0.cols, x.len);
    debug_assert_eq!(w1.cols, x.len, "w1 cols {} != x len {}", w1.cols, x.len);
    debug_assert_eq!(w0.rows, y0.len, "w0 rows {} != y0 len {}", w0.rows, y0.len);
    debug_assert_eq!(w1.rows, y1.len, "w1 rows {} != y1 len {}", w1.rows, y1.len);

    let rows0 = w0.rows as u32;
    let rows1 = w1.rows as u32;
    let cols = x.len as u32;
    let max_rows = w0.rows.max(w1.rows);
    let pipeline = ctx.pipeline("gemv2_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 256);
    let tg_candidate = std::cmp::min(tg_cap, x.len.max(1));
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&w0.buffer), 0);
    encoder.set_buffer(1, Some(&x.buffer), 0);
    encoder.set_buffer(2, Some(&w1.buffer), 0);
    encoder.set_buffer(3, Some(&y0.buffer), 0);
    encoder.set_buffer(4, Some(&y1.buffer), 0);
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        (&rows0 as *const u32).cast(),
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        (&rows1 as *const u32).cast(),
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        (&cols as *const u32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(max_rows as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

pub fn linear3_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    x: &MetalDeviceVec,
    w0: &MetalDeviceMatrix,
    w1: &MetalDeviceMatrix,
    w2: &MetalDeviceMatrix,
    y0: &mut MetalDeviceVec,
    y1: &mut MetalDeviceVec,
    y2: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(w0.cols, x.len, "w0 cols {} != x len {}", w0.cols, x.len);
    debug_assert_eq!(w1.cols, x.len, "w1 cols {} != x len {}", w1.cols, x.len);
    debug_assert_eq!(w2.cols, x.len, "w2 cols {} != x len {}", w2.cols, x.len);
    debug_assert_eq!(w0.rows, y0.len, "w0 rows {} != y0 len {}", w0.rows, y0.len);
    debug_assert_eq!(w1.rows, y1.len, "w1 rows {} != y1 len {}", w1.rows, y1.len);
    debug_assert_eq!(w2.rows, y2.len, "w2 rows {} != y2 len {}", w2.rows, y2.len);

    let rows0 = w0.rows as u32;
    let rows1 = w1.rows as u32;
    let rows2 = w2.rows as u32;
    let cols = x.len as u32;
    let max_rows = w0.rows.max(w1.rows).max(w2.rows);
    let pipeline = ctx.pipeline("gemv3_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 256);
    let tg_candidate = std::cmp::min(tg_cap, x.len.max(1));
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&w0.buffer), 0);
    encoder.set_buffer(1, Some(&x.buffer), 0);
    encoder.set_buffer(2, Some(&w1.buffer), 0);
    encoder.set_buffer(3, Some(&w2.buffer), 0);
    encoder.set_buffer(4, Some(&y0.buffer), 0);
    encoder.set_buffer(5, Some(&y1.buffer), 0);
    encoder.set_buffer(6, Some(&y2.buffer), 0);
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        (&rows0 as *const u32).cast(),
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        (&rows1 as *const u32).cast(),
    );
    encoder.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        (&rows2 as *const u32).cast(),
    );
    encoder.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        (&cols as *const u32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(max_rows as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

pub fn linear_accum_inplace_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    x: &MetalDeviceVec,
    weight: &MetalDeviceMatrix,
    y: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(
        weight.cols, x.len,
        "weight cols {} != x len {}",
        weight.cols, x.len
    );
    debug_assert_eq!(
        weight.rows, y.len,
        "weight rows {} != y len {}",
        weight.rows, y.len
    );

    let rows = weight.rows as u32;
    let cols = weight.cols as u32;
    let pipeline = ctx.pipeline("gemv_accum_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 256);
    let tg_candidate = std::cmp::min(tg_cap, weight.cols.max(1));
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);
    let row_groups = weight.rows.div_ceil(8);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weight.buffer), 0);
    encoder.set_buffer(1, Some(&x.buffer), 0);
    encoder.set_buffer(2, Some(&y.buffer), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&rows as *const u32).cast(),
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        (&cols as *const u32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(row_groups as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

/// GEMM: Y = weight @ X
/// weight: [out_dim, in_dim], x: [seq_len, in_dim], y: [seq_len, out_dim]
pub fn gemm(
    ctx: &MetalDeviceContext,
    weight: &MetalDeviceMatrix,
    x: &MetalHiddenStates,
) -> Result<MetalHiddenStates> {
    debug_assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    let out_dim = weight.rows;
    let seq_len = x.seq_len;
    let out = MetalHiddenStates::zeros(ctx, out_dim, seq_len)?;

    let out_dim_u32 = out_dim as u32;
    let seq_len_u32 = seq_len as u32;
    let in_dim_u32 = weight.cols as u32;

    let pipeline = ctx.pipeline("gemm_kernel")?;
    dispatch_2d(ctx, &pipeline, out_dim, seq_len, |encoder| {
        encoder.set_buffer(0, Some(&weight.buffer), 0);
        encoder.set_buffer(1, Some(&x.buffer), 0);
        encoder.set_buffer(2, Some(&out.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&out_dim_u32 as *const u32).cast(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            (&seq_len_u32 as *const u32).cast(),
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            (&in_dim_u32 as *const u32).cast(),
        );
    })?;

    Ok(out)
}

pub fn silu_mul(
    ctx: &MetalDeviceContext,
    gate: &MetalDeviceVec,
    up: &MetalDeviceVec,
) -> Result<MetalDeviceVec> {
    debug_assert_eq!(
        gate.len, up.len,
        "gate len {} != up len {}",
        gate.len, up.len
    );
    let mut out = MetalDeviceVec::zeros(ctx, gate.len)?;
    silu_mul_into(ctx, gate, up, &mut out)?;
    Ok(out)
}

pub fn silu_mul_into(
    ctx: &MetalDeviceContext,
    gate: &MetalDeviceVec,
    up: &MetalDeviceVec,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    silu_mul_into_encoded(ctx, &encoder, gate, up, out)?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

pub fn silu_mul_into_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    gate: &MetalDeviceVec,
    up: &MetalDeviceVec,
    out: &mut MetalDeviceVec,
) -> Result<()> {
    debug_assert_eq!(
        gate.len, up.len,
        "gate len {} != up len {}",
        gate.len, up.len
    );
    debug_assert_eq!(
        out.len, gate.len,
        "out len {} != gate len {}",
        out.len, gate.len
    );
    let n = gate.len as u32;
    let pipeline = ctx.pipeline("silu_mul_kernel")?;
    dispatch_1d_encoded(&pipeline, gate.len, encoder, |enc| {
        enc.set_buffer(0, Some(&gate.buffer), 0);
        enc.set_buffer(1, Some(&up.buffer), 0);
        enc.set_buffer(2, Some(&out.buffer), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&n as *const u32).cast(),
        );
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_kv_cache_decode_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    k_full: &MetalDeviceVec,
    v_full: &MetalDeviceVec,
    k_norm_weight: &MetalDeviceVec,
    cos_cache: &MetalDeviceVec,
    sin_cache: &MetalDeviceVec,
    k_cache: &mut MetalDeviceVec,
    v_cache: &mut MetalDeviceVec,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq_len: usize,
    eps: f32,
) -> Result<()> {
    let expected_kv_dim = num_kv_heads * head_dim;
    debug_assert_eq!(
        k_full.len, expected_kv_dim,
        "k_full len {} != {}",
        k_full.len, expected_kv_dim
    );
    debug_assert_eq!(
        v_full.len, expected_kv_dim,
        "v_full len {} != {}",
        v_full.len, expected_kv_dim
    );
    debug_assert_eq!(
        k_norm_weight.len, head_dim,
        "k_norm_weight len {} != {}",
        k_norm_weight.len, head_dim
    );
    if pos >= max_seq_len {
        bail!("position {} exceeds max_seq_len {}", pos, max_seq_len);
    }
    let rope_need = (pos + 1) * head_dim;
    debug_assert!(
        cos_cache.len >= rope_need,
        "cos cache len {} < required {}",
        cos_cache.len,
        rope_need
    );
    debug_assert!(
        sin_cache.len >= rope_need,
        "sin cache len {} < required {}",
        sin_cache.len,
        rope_need
    );

    let expected_cache = num_kv_heads * max_seq_len * head_dim;
    debug_assert_eq!(
        k_cache.len, expected_cache,
        "k_cache len {} != {}",
        k_cache.len, expected_cache
    );
    debug_assert_eq!(
        v_cache.len, expected_cache,
        "v_cache len {} != {}",
        v_cache.len, expected_cache
    );

    let num_kv_heads_u32 = num_kv_heads as u32;
    let head_dim_u32 = head_dim as u32;
    let pos_u32 = pos as u32;
    let max_seq_len_u32 = max_seq_len as u32;
    let pipeline = ctx.pipeline("prepare_kv_decode_kernel")?;
    dispatch_1d_encoded(&pipeline, num_kv_heads, encoder, |enc| {
        enc.set_buffer(0, Some(&k_full.buffer), 0);
        enc.set_buffer(1, Some(&v_full.buffer), 0);
        enc.set_buffer(2, Some(&k_norm_weight.buffer), 0);
        enc.set_buffer(3, Some(&cos_cache.buffer), 0);
        enc.set_buffer(4, Some(&sin_cache.buffer), 0);
        enc.set_buffer(5, Some(&k_cache.buffer), 0);
        enc.set_buffer(6, Some(&v_cache.buffer), 0);
        enc.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            (&num_kv_heads_u32 as *const u32).cast(),
        );
        enc.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            (&head_dim_u32 as *const u32).cast(),
        );
        enc.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            (&pos_u32 as *const u32).cast(),
        );
        enc.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            (&max_seq_len_u32 as *const u32).cast(),
        );
        enc.set_bytes(
            11,
            std::mem::size_of::<f32>() as u64,
            (&eps as *const f32).cast(),
        );
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_kv_cache_decode(
    ctx: &MetalDeviceContext,
    k_full: &MetalDeviceVec,
    v_full: &MetalDeviceVec,
    k_norm_weight: &MetalDeviceVec,
    cos_cache: &MetalDeviceVec,
    sin_cache: &MetalDeviceVec,
    k_cache: &mut MetalDeviceVec,
    v_cache: &mut MetalDeviceVec,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    max_seq_len: usize,
    eps: f32,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    prepare_kv_cache_decode_encoded(
        ctx,
        &encoder,
        k_full,
        v_full,
        k_norm_weight,
        cos_cache,
        sin_cache,
        k_cache,
        v_cache,
        num_kv_heads,
        head_dim,
        pos,
        max_seq_len,
        eps,
    )?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_heads_encoded(
    ctx: &MetalDeviceContext,
    encoder: &metal::ComputeCommandEncoderRef,
    q_full: &MetalDeviceVec,
    q_norm_weight: &MetalDeviceVec,
    cos_cache: &MetalDeviceVec,
    sin_cache: &MetalDeviceVec,
    k_cache: &MetalDeviceVec,
    v_cache: &MetalDeviceVec,
    out: &mut MetalDeviceVec,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    max_seq_len: usize,
    scale: f32,
    eps: f32,
) -> Result<()> {
    if num_kv_heads == 0 || num_heads == 0 {
        bail!(
            "invalid head config: num_heads={}, num_kv_heads={}",
            num_heads,
            num_kv_heads
        );
    }
    if num_heads % num_kv_heads != 0 {
        bail!(
            "num_heads {} must be divisible by num_kv_heads {}",
            num_heads,
            num_kv_heads
        );
    }
    if head_dim > 256 {
        bail!("head_dim {} exceeds kernel limit 256", head_dim);
    }
    if seq_len == 0 || seq_len > max_seq_len {
        bail!("invalid seq_len {} (max_seq_len {})", seq_len, max_seq_len);
    }
    if pos >= max_seq_len {
        bail!("position {} exceeds max_seq_len {}", pos, max_seq_len);
    }
    let rope_need = (pos + 1) * head_dim;
    debug_assert!(
        cos_cache.len >= rope_need,
        "cos cache len {} < required {}",
        cos_cache.len,
        rope_need
    );
    debug_assert!(
        sin_cache.len >= rope_need,
        "sin cache len {} < required {}",
        sin_cache.len,
        rope_need
    );

    let expected_q = num_heads * head_dim;
    debug_assert_eq!(
        q_full.len, expected_q,
        "q_full len {} != {}",
        q_full.len, expected_q
    );
    debug_assert_eq!(
        q_norm_weight.len, head_dim,
        "q_norm_weight len {} != {}",
        q_norm_weight.len, head_dim
    );
    debug_assert_eq!(out.len, expected_q, "out len {} != {}", out.len, expected_q);

    let expected_cache = num_kv_heads * max_seq_len * head_dim;
    debug_assert_eq!(
        k_cache.len, expected_cache,
        "k_cache len {} != {}",
        k_cache.len, expected_cache
    );
    debug_assert_eq!(
        v_cache.len, expected_cache,
        "v_cache len {} != {}",
        v_cache.len, expected_cache
    );

    let num_heads_u32 = num_heads as u32;
    let num_kv_heads_u32 = num_kv_heads as u32;
    let gqa_ratio_u32 = (num_heads / num_kv_heads) as u32;
    let head_dim_u32 = head_dim as u32;
    let pos_u32 = pos as u32;
    let seq_len_u32 = seq_len as u32;
    let max_seq_len_u32 = max_seq_len as u32;
    let pipeline = ctx.pipeline("attention_decode_heads_kernel")?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap = std::cmp::min(max_threads, 128);
    let tg_target = head_dim.max(seq_len.min(128)).max(1);
    let tg_candidate = std::cmp::min(tg_cap, tg_target);
    let threads_per_group = largest_power_of_two_leq(tg_candidate).max(1);

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&q_full.buffer), 0);
    encoder.set_buffer(1, Some(&q_norm_weight.buffer), 0);
    encoder.set_buffer(2, Some(&cos_cache.buffer), 0);
    encoder.set_buffer(3, Some(&sin_cache.buffer), 0);
    encoder.set_buffer(4, Some(&k_cache.buffer), 0);
    encoder.set_buffer(5, Some(&v_cache.buffer), 0);
    encoder.set_buffer(6, Some(&out.buffer), 0);
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        (&num_heads_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        (&num_kv_heads_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        (&gqa_ratio_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        (&head_dim_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        (&pos_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        (&seq_len_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        (&max_seq_len_u32 as *const u32).cast(),
    );
    encoder.set_bytes(
        14,
        std::mem::size_of::<f32>() as u64,
        (&scale as *const f32).cast(),
    );
    encoder.set_bytes(
        15,
        std::mem::size_of::<f32>() as u64,
        (&eps as *const f32).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_heads(
    ctx: &MetalDeviceContext,
    q_full: &MetalDeviceVec,
    q_norm_weight: &MetalDeviceVec,
    cos_cache: &MetalDeviceVec,
    sin_cache: &MetalDeviceVec,
    k_cache: &MetalDeviceVec,
    v_cache: &MetalDeviceVec,
    out: &mut MetalDeviceVec,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    max_seq_len: usize,
    scale: f32,
    eps: f32,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    attention_decode_heads_encoded(
        ctx,
        &encoder,
        q_full,
        q_norm_weight,
        cos_cache,
        sin_cache,
        k_cache,
        v_cache,
        out,
        num_heads,
        num_kv_heads,
        head_dim,
        pos,
        seq_len,
        max_seq_len,
        scale,
        eps,
    )?;
    encoder.end_encoding();
    cmd.commit();
    Ok(())
}

pub fn linear_argmax_with_workspace(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    weight: &MetalDeviceMatrix,
    out: &metal::BufferRef,
    partial_vals: &metal::BufferRef,
    partial_idxs: &metal::BufferRef,
) -> Result<u32> {
    if weight.cols != x.len {
        bail!("weight cols {} != x len {}", weight.cols, x.len);
    }
    if weight.rows == 0 {
        bail!("linear_argmax requires non-empty rows");
    }
    if out.length() < std::mem::size_of::<i32>() as u64 {
        bail!("argmax scratch buffer too small: {}", out.length());
    }
    unsafe {
        *(out.contents() as *mut i32) = 0;
    }

    let rows = weight.rows as u32;
    let cols = weight.cols as u32;
    let rows_per_group = 8usize;
    let num_groups = weight.rows.div_ceil(rows_per_group) as u32;
    let need_partial_vals = num_groups as u64 * std::mem::size_of::<f32>() as u64;
    let need_partial_idxs = num_groups as u64 * std::mem::size_of::<u32>() as u64;
    if partial_vals.length() < need_partial_vals {
        bail!(
            "partial_vals scratch too small: have {}, need {}",
            partial_vals.length(),
            need_partial_vals
        );
    }
    if partial_idxs.length() < need_partial_idxs {
        bail!(
            "partial_idxs scratch too small: have {}, need {}",
            partial_idxs.length(),
            need_partial_idxs
        );
    }

    let pipeline1 = ctx.pipeline("gemv_argmax_stage1_kernel")?;
    let max_threads1 = pipeline1.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap1 = std::cmp::min(max_threads1, 256);
    let tg_candidate1 = std::cmp::min(tg_cap1, weight.cols.max(1));
    let threads_per_group1 = largest_power_of_two_leq(tg_candidate1).max(1);

    let pipeline2 = ctx.pipeline("argmax_stage2_kernel")?;
    let max_threads2 = pipeline2.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap2 = std::cmp::min(max_threads2, 256);
    let tg_candidate2 = std::cmp::min(tg_cap2, num_groups.max(1) as usize);
    let threads_per_group2 = largest_power_of_two_leq(tg_candidate2).max(1);

    let cmd = ctx.queue.new_command_buffer();

    let enc1 = cmd.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pipeline1);
    enc1.set_buffer(0, Some(&weight.buffer), 0);
    enc1.set_buffer(1, Some(&x.buffer), 0);
    enc1.set_buffer(2, Some(partial_vals), 0);
    enc1.set_buffer(3, Some(partial_idxs), 0);
    enc1.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        (&rows as *const u32).cast(),
    );
    enc1.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        (&cols as *const u32).cast(),
    );
    enc1.dispatch_thread_groups(
        MTLSize::new(num_groups as u64, 1, 1),
        MTLSize::new(threads_per_group1 as u64, 1, 1),
    );
    enc1.end_encoding();

    let enc2 = cmd.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pipeline2);
    enc2.set_buffer(0, Some(partial_vals), 0);
    enc2.set_buffer(1, Some(partial_idxs), 0);
    enc2.set_buffer(2, Some(out), 0);
    enc2.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&num_groups as *const u32).cast(),
    );
    enc2.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(threads_per_group2 as u64, 1, 1),
    );
    enc2.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();
    let idx = unsafe { *(out.contents() as *const i32) };
    Ok(idx.max(0) as u32)
}

pub fn argmax_with_workspace(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    out: &metal::BufferRef,
    partial_vals: &metal::BufferRef,
    partial_idxs: &metal::BufferRef,
) -> Result<u32> {
    if x.len == 0 {
        bail!("argmax requires non-empty input");
    }
    let n = x.len as u32;
    if out.length() < std::mem::size_of::<i32>() as u64 {
        bail!("argmax scratch buffer too small: {}", out.length());
    }
    unsafe {
        *(out.contents() as *mut i32) = 0;
    }

    let pipeline1 = ctx.pipeline("argmax_stage1_kernel")?;
    let max_threads1 = pipeline1.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap1 = std::cmp::min(max_threads1, 256);
    let tg_candidate1 = std::cmp::min(tg_cap1, x.len.max(1));
    let threads_per_group1 = largest_power_of_two_leq(tg_candidate1).max(1);
    let num_groups = x.len.div_ceil(threads_per_group1) as u32;

    let need_partial_vals = num_groups as u64 * std::mem::size_of::<f32>() as u64;
    let need_partial_idxs = num_groups as u64 * std::mem::size_of::<u32>() as u64;
    if partial_vals.length() < need_partial_vals {
        bail!(
            "partial_vals scratch too small: have {}, need {}",
            partial_vals.length(),
            need_partial_vals
        );
    }
    if partial_idxs.length() < need_partial_idxs {
        bail!(
            "partial_idxs scratch too small: have {}, need {}",
            partial_idxs.length(),
            need_partial_idxs
        );
    }

    let cmd = ctx.queue.new_command_buffer();

    let enc1 = cmd.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pipeline1);
    enc1.set_buffer(0, Some(&x.buffer), 0);
    enc1.set_buffer(1, Some(partial_vals), 0);
    enc1.set_buffer(2, Some(partial_idxs), 0);
    enc1.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&n as *const u32).cast(),
    );
    enc1.dispatch_thread_groups(
        MTLSize::new(num_groups as u64, 1, 1),
        MTLSize::new(threads_per_group1 as u64, 1, 1),
    );
    enc1.end_encoding();

    let pipeline2 = ctx.pipeline("argmax_stage2_kernel")?;
    let max_threads2 = pipeline2.max_total_threads_per_threadgroup().max(1) as usize;
    let tg_cap2 = std::cmp::min(max_threads2, 256);
    let tg_candidate2 = std::cmp::min(tg_cap2, num_groups.max(1) as usize);
    let threads_per_group2 = largest_power_of_two_leq(tg_candidate2).max(1);

    let enc2 = cmd.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pipeline2);
    enc2.set_buffer(0, Some(partial_vals), 0);
    enc2.set_buffer(1, Some(partial_idxs), 0);
    enc2.set_buffer(2, Some(out), 0);
    enc2.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        (&num_groups as *const u32).cast(),
    );
    enc2.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(threads_per_group2 as u64, 1, 1),
    );
    enc2.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let idx = unsafe { *(out.contents() as *const i32) };
    Ok(idx.max(0) as u32)
}

pub fn argmax_with_scratch(
    ctx: &MetalDeviceContext,
    x: &MetalDeviceVec,
    out: &metal::BufferRef,
) -> Result<u32> {
    let partial_vals = ctx.device.new_buffer(
        (x.len.max(1) * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_idxs = ctx.device.new_buffer(
        (x.len.max(1) * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    argmax_with_workspace(ctx, x, out, &partial_vals, &partial_idxs)
}

pub fn argmax(ctx: &MetalDeviceContext, x: &MetalDeviceVec) -> Result<u32> {
    let out = ctx.device.new_buffer(
        std::mem::size_of::<i32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    argmax_with_scratch(ctx, x, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use metal::Device;

    fn f16_vec(data: &[f32]) -> Vec<f16> {
        data.iter().map(|&x| f16::from_f32(x)).collect()
    }

    fn make_ctx() -> Option<MetalDeviceContext> {
        if Device::system_default().is_none() {
            return None;
        }
        MetalDeviceContext::new().ok()
    }

    fn synced_vec(ctx: &MetalDeviceContext, x: &MetalDeviceVec) -> Result<Vec<f32>> {
        ctx.sync()?;
        x.to_host()
    }

    fn synced_hidden(ctx: &MetalDeviceContext, x: &MetalHiddenStates) -> Result<Vec<f32>> {
        ctx.sync()?;
        x.to_host()
    }

    #[test]
    fn test_add() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let a = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 2.0, -1.0]))?;
        let b = MetalDeviceVec::from_host(&ctx, &f16_vec(&[0.5, -1.5, 2.0]))?;
        let out = add(&ctx, &a, &b)?;
        let result = synced_vec(&ctx, &out)?;
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
        assert!((result[2] - 1.0).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let x = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 2.0, 3.0, 4.0]))?;
        let w = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 1.0, 1.0, 1.0]))?;
        let out = rms_norm(&ctx, &x, &w, 1e-6)?;
        let result = synced_vec(&ctx, &out)?;
        let rms = (7.5_f32 + 1e-6).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 0.02);
        assert!((result[1] - 2.0 / rms).abs() < 0.02);
        Ok(())
    }

    #[test]
    fn test_gemv() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let a_data = f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // [2,3]
        let x_data = f16_vec(&[1.0, 2.0, 3.0]); // [3]
        let a = MetalDeviceMatrix::from_host(&ctx, &a_data, 2, 3)?;
        let x = MetalDeviceVec::from_host(&ctx, &x_data)?;
        let y = linear(&ctx, &x, &a)?;
        let result = synced_vec(&ctx, &y)?;
        assert!((result[0] - 14.0).abs() < 0.1);
        assert!((result[1] - 32.0).abs() < 0.1);
        Ok(())
    }

    #[test]
    fn test_gemm() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let w = MetalDeviceMatrix::from_host(
            &ctx,
            &f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // [2,3]
            2,
            3,
        )?;
        let x = MetalHiddenStates::from_host(
            &ctx,
            &f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // [seq=2, in=3]
            3,
            2,
        )?;
        let y = gemm(&ctx, &w, &x)?;
        let result = synced_hidden(&ctx, &y)?;

        // token0 => [14, 32], token1 => [32, 77]
        assert!((result[0] - 14.0).abs() < 0.1);
        assert!((result[1] - 32.0).abs() < 0.1);
        assert!((result[2] - 32.0).abs() < 0.1);
        assert!((result[3] - 77.0).abs() < 0.1);
        Ok(())
    }

    #[test]
    fn test_silu_mul() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let gate = MetalDeviceVec::from_host(&ctx, &f16_vec(&[-1.0, 0.0, 2.0]))?;
        let up = MetalDeviceVec::from_host(&ctx, &f16_vec(&[2.0, 3.0, 4.0]))?;
        let out = silu_mul(&ctx, &gate, &up)?;
        let v = synced_vec(&ctx, &out)?;
        let ref0 = (-1.0f32 / (1.0 + 1.0f32.exp())) * 2.0;
        let ref1 = 0.0;
        let ref2 = (2.0f32 / (1.0 + (-2.0f32).exp())) * 4.0;
        assert!((v[0] - ref0).abs() < 0.02);
        assert!((v[1] - ref1).abs() < 0.02);
        assert!((v[2] - ref2).abs() < 0.02);
        Ok(())
    }

    #[test]
    fn test_argmax() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };
        let x = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, -2.0, 4.5, 3.2]))?;
        let idx = argmax(&ctx, &x)?;
        assert_eq!(idx, 2);
        Ok(())
    }

    #[test]
    fn test_decode_attention_heads_single_step() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };

        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let max_seq_len = 4usize;
        let pos = 0usize;
        let seq_len = 1usize;
        let eps = 1e-6f32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_full = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 0.0]))?;
        let k_full = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 0.0]))?;
        let v_full = MetalDeviceVec::from_host(&ctx, &f16_vec(&[3.0, -2.0]))?;
        let q_norm = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 1.0]))?;
        let k_norm = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 1.0]))?;

        let cos = MetalDeviceVec::from_host(&ctx, &f16_vec(&vec![1.0; max_seq_len * head_dim]))?;
        let sin = MetalDeviceVec::from_host(&ctx, &f16_vec(&vec![0.0; max_seq_len * head_dim]))?;
        let mut k_cache = MetalDeviceVec::zeros(&ctx, num_kv_heads * max_seq_len * head_dim)?;
        let mut v_cache = MetalDeviceVec::zeros(&ctx, num_kv_heads * max_seq_len * head_dim)?;

        prepare_kv_cache_decode(
            &ctx,
            &k_full,
            &v_full,
            &k_norm,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            num_kv_heads,
            head_dim,
            pos,
            max_seq_len,
            eps,
        )?;

        let mut out = MetalDeviceVec::zeros(&ctx, num_heads * head_dim)?;
        attention_decode_heads(
            &ctx,
            &q_full,
            &q_norm,
            &cos,
            &sin,
            &k_cache,
            &v_cache,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            pos,
            seq_len,
            max_seq_len,
            scale,
            eps,
        )?;

        let out_host = synced_vec(&ctx, &out)?;
        assert!((out_host[0] - 3.0).abs() < 0.03, "out[0]={}", out_host[0]);
        assert!((out_host[1] + 2.0).abs() < 0.03, "out[1]={}", out_host[1]);
        Ok(())
    }

    #[test]
    fn test_decode_attention_q_rms_norm_matches_reference() -> Result<()> {
        let Some(ctx) = make_ctx() else {
            return Ok(());
        };

        let num_heads = 1usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let max_seq_len = 4usize;
        let eps = 1e-6f32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_full = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 0.0]))?;
        let q_norm = MetalDeviceVec::from_host(&ctx, &f16_vec(&[2.0, 1.0]))?;
        let k_norm = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 1.0]))?;

        let k0 = MetalDeviceVec::from_host(&ctx, &f16_vec(&[1.0, 0.0]))?;
        let v0 = MetalDeviceVec::from_host(&ctx, &f16_vec(&[10.0, 0.0]))?;
        let k1 = MetalDeviceVec::from_host(&ctx, &f16_vec(&[0.0, 1.0]))?;
        let v1 = MetalDeviceVec::from_host(&ctx, &f16_vec(&[0.0, 20.0]))?;

        let cos = MetalDeviceVec::from_host(&ctx, &f16_vec(&vec![1.0; max_seq_len * head_dim]))?;
        let sin = MetalDeviceVec::from_host(&ctx, &f16_vec(&vec![0.0; max_seq_len * head_dim]))?;
        let mut k_cache = MetalDeviceVec::zeros(&ctx, num_kv_heads * max_seq_len * head_dim)?;
        let mut v_cache = MetalDeviceVec::zeros(&ctx, num_kv_heads * max_seq_len * head_dim)?;

        prepare_kv_cache_decode(
            &ctx,
            &k0,
            &v0,
            &k_norm,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            num_kv_heads,
            head_dim,
            0,
            max_seq_len,
            eps,
        )?;
        prepare_kv_cache_decode(
            &ctx,
            &k1,
            &v1,
            &k_norm,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            num_kv_heads,
            head_dim,
            1,
            max_seq_len,
            eps,
        )?;

        let mut out = MetalDeviceVec::zeros(&ctx, num_heads * head_dim)?;
        attention_decode_heads(
            &ctx,
            &q_full,
            &q_norm,
            &cos,
            &sin,
            &k_cache,
            &v_cache,
            &mut out,
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            2,
            max_seq_len,
            scale,
            eps,
        )?;

        let out_host = synced_vec(&ctx, &out)?;
        assert!((out_host[0] - 9.44).abs() < 0.2, "out[0]={}", out_host[0]);
        assert!((out_host[1] - 1.12).abs() < 0.2, "out[1]={}", out_host[1]);
        Ok(())
    }
}
