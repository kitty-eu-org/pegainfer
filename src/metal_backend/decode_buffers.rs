//! Pre-allocated Metal buffers for single-token decode.

use anyhow::Result;

use crate::metal_backend::tensor::{MetalDeviceContext, MetalDeviceVec};
use crate::qwen3_config::Config;

pub struct MetalDecodeBuffers {
    pub hidden: MetalDeviceVec,
    pub normed: MetalDeviceVec,
    pub q: MetalDeviceVec,
    pub k: MetalDeviceVec,
    pub v: MetalDeviceVec,
    pub attn_out: MetalDeviceVec,
    pub attn_proj: MetalDeviceVec,
    pub gate: MetalDeviceVec,
    pub up: MetalDeviceVec,
    pub mlp_act: MetalDeviceVec,
    pub mlp_out: MetalDeviceVec,
    pub logits: MetalDeviceVec,
}

impl MetalDecodeBuffers {
    pub fn new(ctx: &MetalDeviceContext, config: &Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let vocab = config.vocab_size;

        Ok(Self {
            hidden: MetalDeviceVec::zeros(ctx, hidden_size)?,
            normed: MetalDeviceVec::zeros(ctx, hidden_size)?,
            q: MetalDeviceVec::zeros(ctx, q_dim)?,
            k: MetalDeviceVec::zeros(ctx, kv_dim)?,
            v: MetalDeviceVec::zeros(ctx, kv_dim)?,
            attn_out: MetalDeviceVec::zeros(ctx, q_dim)?,
            attn_proj: MetalDeviceVec::zeros(ctx, hidden_size)?,
            gate: MetalDeviceVec::zeros(ctx, intermediate)?,
            up: MetalDeviceVec::zeros(ctx, intermediate)?,
            mlp_act: MetalDeviceVec::zeros(ctx, intermediate)?,
            mlp_out: MetalDeviceVec::zeros(ctx, hidden_size)?,
            logits: MetalDeviceVec::empty_private(ctx, vocab)?,
        })
    }
}
