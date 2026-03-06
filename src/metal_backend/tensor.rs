//! Metal device tensor types and context.

use anyhow::{anyhow, Result};
use half::{bf16, f16};
use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLStorageMode,
};
use std::collections::HashMap;

/// Metal device context holding selected device and command queue.
pub struct MetalDeviceContext {
    pub device: Device,
    pub queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalDeviceContext {
    pub fn new() -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow!("No Metal device available"))?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        let library = device
            .new_library_with_source(include_str!("shaders/basic.metal"), &options)
            .map_err(|e| anyhow!("Failed to compile Metal shaders: {}", e))?;
        const KERNEL_FUNCTIONS: [&str; 15] = [
            "copy_kernel",
            "embedding_kernel",
            "add_kernel",
            "rms_norm_kernel",
            "gemv_kernel",
            "gemv_accum_kernel",
            "gemv2_kernel",
            "gemv3_kernel",
            "gemm_kernel",
            "silu_mul_kernel",
            "argmax_stage1_kernel",
            "gemv_argmax_stage1_kernel",
            "argmax_stage2_kernel",
            "prepare_kv_decode_kernel",
            "attention_decode_heads_kernel",
        ];
        let mut pipelines = HashMap::with_capacity(KERNEL_FUNCTIONS.len());
        for &name in &KERNEL_FUNCTIONS {
            let function = library
                .get_function(name, None)
                .map_err(|e| anyhow!("Failed to load Metal function '{}': {}", name, e))?;
            let state = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow!("Failed to build Metal pipeline '{}': {}", name, e))?;
            pipelines.insert(name, state);
        }
        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    /// Synchronize previously committed command buffers.
    pub fn sync(&self) -> Result<()> {
        // This empty command buffer acts as a fence for all prior work in this queue.
        let cmd = self.queue.new_command_buffer();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    pub fn pipeline(&self, function_name: &'static str) -> Result<ComputePipelineState> {
        self.pipelines
            .get(function_name)
            .cloned()
            .ok_or_else(|| anyhow!("Missing Metal pipeline '{}'", function_name))
    }
}

/// 1D device tensor (vector) stored as fp16 in a shared MTLBuffer.
pub struct MetalDeviceVec {
    pub buffer: metal::Buffer,
    pub len: usize,
}

impl MetalDeviceVec {
    /// Create from host data (fp16)
    pub fn from_host(ctx: &MetalDeviceContext, data: &[f16]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data) as u64;
        let buffer = ctx.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            buffer,
            len: data.len(),
        })
    }

    /// Create read-only tensor in private storage (GPU-only), uploaded via blit once.
    pub fn from_host_private(ctx: &MetalDeviceContext, data: &[f16]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data) as u64;
        let staging = ctx.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer = ctx
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModePrivate);
        let cmd = ctx.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&staging, 0, &buffer, 0, bytes);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(Self {
            buffer,
            len: data.len(),
        })
    }

    pub fn from_safetensors(ctx: &MetalDeviceContext, data: &[u8]) -> Result<Self> {
        if data.len() % 2 != 0 {
            return Err(anyhow!(
                "Data length must be even for bf16: got {} bytes",
                data.len()
            ));
        }
        let len = data.len() / 2;
        // Qwen safetensors are bf16; convert once to fp16 for faster Metal compute.
        let bf = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
        let mut fp = Vec::with_capacity(len);
        for &v in bf {
            fp.push(f16::from_f32(v.to_f32()));
        }
        Self::from_host_private(ctx, &fp)
    }

    /// Create zeroed tensor
    pub fn zeros(ctx: &MetalDeviceContext, len: usize) -> Result<Self> {
        let bytes = (len * std::mem::size_of::<f16>()) as u64;
        let buffer = ctx
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(buffer.contents(), 0, bytes as usize);
        }
        Ok(Self { buffer, len })
    }

    /// Create an uninitialized tensor in private GPU storage.
    pub fn empty_private(ctx: &MetalDeviceContext, len: usize) -> Result<Self> {
        let bytes = (len * std::mem::size_of::<f16>()) as u64;
        let buffer = ctx
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModePrivate);
        Ok(Self { buffer, len })
    }

    /// Copy to host as f32 (for compatibility with existing code paths)
    pub fn to_host(&self) -> Result<Vec<f32>> {
        let ptr = self.buffer.contents().cast::<f16>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        Ok(slice.iter().map(|x| x.to_f32()).collect())
    }

    /// Copy to host as f32, including private GPU buffers via a temporary blit.
    pub fn to_host_with_ctx(&self, ctx: &MetalDeviceContext) -> Result<Vec<f32>> {
        match self.buffer.storage_mode() {
            MTLStorageMode::Private => {
                let bytes = (self.len * std::mem::size_of::<f16>()) as u64;
                let staging = ctx
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                let cmd = ctx.queue.new_command_buffer();
                let blit = cmd.new_blit_command_encoder();
                blit.copy_from_buffer(&self.buffer, 0, &staging, 0, bytes);
                blit.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let ptr = staging.contents().cast::<f16>();
                let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
                Ok(slice.iter().map(|x| x.to_f32()).collect())
            }
            _ => {
                ctx.sync()?;
                self.to_host()
            }
        }
    }
}

/// 2D device tensor (matrix) stored in row-major order as fp16.
pub struct MetalDeviceMatrix {
    pub buffer: metal::Buffer,
    pub rows: usize,
    pub cols: usize,
}

impl MetalDeviceMatrix {
    /// Create from host data (row-major, fp16)
    pub fn from_host(
        ctx: &MetalDeviceContext,
        data: &[f16],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let bytes = std::mem::size_of_val(data) as u64;
        let buffer = ctx.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self { buffer, rows, cols })
    }

    /// Create read-only matrix in private storage (GPU-only), uploaded via blit once.
    pub fn from_host_private(
        ctx: &MetalDeviceContext,
        data: &[f16],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let bytes = std::mem::size_of_val(data) as u64;
        let staging = ctx.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer = ctx
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModePrivate);
        let cmd = ctx.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&staging, 0, &buffer, 0, bytes);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(Self { buffer, rows, cols })
    }

    pub fn from_safetensors(
        ctx: &MetalDeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        let bf = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        let mut fp = Vec::with_capacity(rows * cols);
        for &v in bf {
            fp.push(f16::from_f32(v.to_f32()));
        }
        Self::from_host_private(ctx, &fp, rows, cols)
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
pub struct MetalHiddenStates {
    pub buffer: metal::Buffer,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl MetalHiddenStates {
    /// Create from host data (token-major, row-major): [seq_len * hidden_dim]
    pub fn from_host(
        ctx: &MetalDeviceContext,
        data: &[f16],
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        assert_eq!(
            data.len(),
            hidden_dim * seq_len,
            "data len {} != {}x{}",
            data.len(),
            hidden_dim,
            seq_len
        );
        let bytes = std::mem::size_of_val(data) as u64;
        let buffer = ctx.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self {
            buffer,
            hidden_dim,
            seq_len,
        })
    }

    /// Create zeroed batch
    pub fn zeros(ctx: &MetalDeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let count = hidden_dim * seq_len;
        let bytes = (count * std::mem::size_of::<f16>()) as u64;
        let buffer = ctx
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(buffer.contents(), 0, bytes as usize);
        }
        Ok(Self {
            buffer,
            hidden_dim,
            seq_len,
        })
    }

    pub fn to_host(&self) -> Result<Vec<f32>> {
        let len = self.hidden_dim * self.seq_len;
        let ptr = self.buffer.contents().cast::<f16>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        Ok(slice.iter().map(|x| x.to_f32()).collect())
    }
}
