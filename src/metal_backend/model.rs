//! Qwen3 model for Metal backend.

use std::fs;

use anyhow::{anyhow, bail, Result};
use log::info;
use metal::MTLResourceOptions;
use rand::rngs::StdRng;
use safetensors::SafeTensors;

use crate::metal_backend::decode_buffers::MetalDecodeBuffers;
use crate::metal_backend::ops;
use crate::metal_backend::tensor::{MetalDeviceContext, MetalDeviceMatrix, MetalDeviceVec};
use crate::metal_backend::weight_loader::{
    load_shard_info, load_tensor_1d, load_tensor_2d, precompute_rope,
};
use crate::qwen3_config::Config;
use crate::sampler::{self, SamplingParams};

const MAX_SEQ_LEN: usize = 4096;

/// Streaming generation summary for transport layers.
pub struct StreamingStats {
    pub emitted_tokens: usize,
    pub hit_eos: bool,
    pub consumer_dropped: bool,
}

pub struct Attention {
    pub q_proj: MetalDeviceMatrix,
    pub k_proj: MetalDeviceMatrix,
    pub v_proj: MetalDeviceMatrix,
    pub o_proj: MetalDeviceMatrix,
    pub q_norm: MetalDeviceVec,
    pub k_norm: MetalDeviceVec,
}

pub struct Mlp {
    pub gate_proj: MetalDeviceMatrix,
    pub up_proj: MetalDeviceMatrix,
    pub down_proj: MetalDeviceMatrix,
}

pub struct TransformerBlock {
    pub input_layernorm: MetalDeviceVec,
    pub attention: Attention,
    pub post_attention_layernorm: MetalDeviceVec,
    pub mlp: Mlp,
}

struct LayerCache {
    // Flattened [num_kv_heads * max_seq_len * head_dim], bf16 storage on Metal shared buffers
    k: MetalDeviceVec,
    v: MetalDeviceVec,
}

struct KvCache {
    layers: Vec<LayerCache>,
    seq_len: usize,
    max_seq_len: usize,
}

impl KvCache {
    fn new(
        ctx: &MetalDeviceContext,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let layer_elems = num_kv_heads * max_seq_len * head_dim;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerCache {
                k: MetalDeviceVec::zeros(ctx, layer_elems)?,
                v: MetalDeviceVec::zeros(ctx, layer_elems)?,
            });
        }
        Ok(Self {
            layers,
            seq_len: 0,
            max_seq_len,
        })
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) {
        self.seq_len = 0;
    }

    fn increment_seq_len(&mut self) {
        self.seq_len += 1;
    }
}

pub struct Qwen3MetalModel {
    pub ctx: MetalDeviceContext,
    pub config: Config,
    pub embed_tokens: MetalDeviceMatrix,
    pub layers: Vec<TransformerBlock>,
    pub norm: MetalDeviceVec,
    cos_cache: MetalDeviceVec,
    sin_cache: MetalDeviceVec,
    kv_cache: KvCache,
    decode_bufs: Option<MetalDecodeBuffers>,
    argmax_scratch_out: metal::Buffer,
    argmax_partial_vals: metal::Buffer,
    argmax_partial_idxs: metal::Buffer,
}

impl Qwen3MetalModel {
    const PREFILL_CHUNK_SIZE: usize = 256;

    pub fn from_safetensors(model_path: &str) -> Result<Self> {
        info!("Loading model for Metal backend from: {}", model_path);
        let ctx = MetalDeviceContext::new()?;
        let config = Config::from_file(model_path)?;

        if config.head_dim % 2 != 0 {
            bail!("head_dim must be even for RoPE, got {}", config.head_dim);
        }

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        let shard_data: Vec<Vec<u8>> = shard_paths
            .iter()
            .map(fs::read)
            .collect::<std::io::Result<_>>()?;
        let shards: Vec<SafeTensors> = shard_data
            .iter()
            .map(|d| SafeTensors::deserialize(d).map_err(|e| anyhow!("Deserialize error: {}", e)))
            .collect::<Result<_>>()?;

        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: Attention {
                    q_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_proj.weight", prefix),
                    )?,
                    k_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_proj.weight", prefix),
                    )?,
                    v_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.v_proj.weight", prefix),
                    )?,
                    o_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.o_proj.weight", prefix),
                    )?,
                    q_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_norm.weight", prefix),
                    )?,
                    k_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_norm.weight", prefix),
                    )?,
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: Mlp {
                    gate_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.gate_proj.weight", prefix),
                    )?,
                    up_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.up_proj.weight", prefix),
                    )?,
                    down_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.down_proj.weight", prefix),
                    )?,
                },
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "model.norm.weight")?;
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, MAX_SEQ_LEN, config.rope_theta)?;
        let kv_cache = KvCache::new(
            &ctx,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            MAX_SEQ_LEN,
        )?;
        let argmax_scratch_out = ctx.device.new_buffer(
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let argmax_partial_vals = ctx.device.new_buffer(
            (config.vocab_size.max(1) * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let argmax_partial_idxs = ctx.device.new_buffer(
            (config.vocab_size.max(1) * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
            kv_cache,
            decode_bufs: None,
            argmax_scratch_out,
            argmax_partial_vals,
            argmax_partial_idxs,
        })
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        if max_new_tokens == 0 {
            return Ok(prompt_tokens.to_vec());
        }

        self.kv_cache.reset();
        let mut bufs = self.take_decode_bufs()?;
        let mut tokens = prompt_tokens.to_vec();

        if params.is_greedy() {
            if prompt_tokens.is_empty() {
                self.forward_one_token_into(self.config.bos_token_id, &mut bufs, false)?;
            } else {
                self.forward_prompt_tokens_into(prompt_tokens, &mut bufs, false)?;
            }

            for _ in 0..max_new_tokens {
                let next_token = self.select_token_greedy_from_normed(&bufs.normed)?;
                if next_token == self.config.eos_token_id {
                    break;
                }
                tokens.push(next_token);
                self.forward_one_token_into(next_token, &mut bufs, false)?;
            }
        } else {
            if prompt_tokens.is_empty() {
                self.forward_one_token_into(self.config.bos_token_id, &mut bufs, true)?;
            } else {
                // Only the final prompt token needs logits for sampling.
                self.forward_prompt_tokens_into(prompt_tokens, &mut bufs, true)?;
            }

            for _ in 0..max_new_tokens {
                let next_token = self.select_token(&bufs.logits, params, rng)?;
                if next_token == self.config.eos_token_id {
                    break;
                }
                tokens.push(next_token);
                self.forward_one_token_into(next_token, &mut bufs, true)?;
            }
        }

        self.decode_bufs = Some(bufs);
        Ok(tokens)
    }

    pub fn generate_streaming_with_callback<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        mut on_token: F,
    ) -> Result<StreamingStats>
    where
        F: FnMut(u32) -> bool,
    {
        if max_new_tokens == 0 {
            return Ok(StreamingStats {
                emitted_tokens: 0,
                hit_eos: false,
                consumer_dropped: false,
            });
        }

        self.kv_cache.reset();
        let mut bufs = self.take_decode_bufs()?;
        let mut emitted_tokens = 0usize;

        if params.is_greedy() {
            if prompt_tokens.is_empty() {
                self.forward_one_token_into(self.config.bos_token_id, &mut bufs, false)?;
            } else {
                self.forward_prompt_tokens_into(prompt_tokens, &mut bufs, false)?;
            }

            for _ in 0..max_new_tokens {
                let next_token = self.select_token_greedy_from_normed(&bufs.normed)?;
                if next_token == self.config.eos_token_id {
                    self.decode_bufs = Some(bufs);
                    return Ok(StreamingStats {
                        emitted_tokens,
                        hit_eos: true,
                        consumer_dropped: false,
                    });
                }

                if !on_token(next_token) {
                    self.decode_bufs = Some(bufs);
                    return Ok(StreamingStats {
                        emitted_tokens,
                        hit_eos: false,
                        consumer_dropped: true,
                    });
                }
                emitted_tokens += 1;
                self.forward_one_token_into(next_token, &mut bufs, false)?;
            }
        } else {
            if prompt_tokens.is_empty() {
                self.forward_one_token_into(self.config.bos_token_id, &mut bufs, true)?;
            } else {
                self.forward_prompt_tokens_into(prompt_tokens, &mut bufs, true)?;
            }

            for _ in 0..max_new_tokens {
                let next_token = self.select_token(&bufs.logits, params, rng)?;
                if next_token == self.config.eos_token_id {
                    self.decode_bufs = Some(bufs);
                    return Ok(StreamingStats {
                        emitted_tokens,
                        hit_eos: true,
                        consumer_dropped: false,
                    });
                }

                if !on_token(next_token) {
                    self.decode_bufs = Some(bufs);
                    return Ok(StreamingStats {
                        emitted_tokens,
                        hit_eos: false,
                        consumer_dropped: true,
                    });
                }
                emitted_tokens += 1;
                self.forward_one_token_into(next_token, &mut bufs, true)?;
            }
        }

        self.decode_bufs = Some(bufs);
        Ok(StreamingStats {
            emitted_tokens,
            hit_eos: false,
            consumer_dropped: false,
        })
    }

    /// Debug helper: return logits for the next token after consuming prompt tokens.
    /// Uses the same decode path as generation and synchronizes before host readback.
    pub fn debug_last_logits_for_prompt(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
        self.kv_cache.reset();
        let mut bufs = self.take_decode_bufs()?;

        if prompt_tokens.is_empty() {
            self.forward_one_token_into(self.config.bos_token_id, &mut bufs, true)?;
        } else {
            self.forward_prompt_tokens_into(prompt_tokens, &mut bufs, true)?;
        }

        let logits = bufs.logits.to_host_with_ctx(&self.ctx)?;
        self.decode_bufs = Some(bufs);
        Ok(logits)
    }

    fn select_token(
        &self,
        logits: &MetalDeviceVec,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        if params.is_greedy() {
            ops::argmax_with_workspace(
                &self.ctx,
                logits,
                &self.argmax_scratch_out,
                &self.argmax_partial_vals,
                &self.argmax_partial_idxs,
            )
        } else {
            let host = logits.to_host_with_ctx(&self.ctx)?;
            Ok(sampler::sample(&host, params, rng))
        }
    }

    fn select_token_greedy_from_normed(&self, normed: &MetalDeviceVec) -> Result<u32> {
        ops::linear_argmax_with_workspace(
            &self.ctx,
            normed,
            &self.embed_tokens,
            &self.argmax_scratch_out,
            &self.argmax_partial_vals,
            &self.argmax_partial_idxs,
        )
    }

    fn forward_prompt_tokens_into(
        &mut self,
        prompt_tokens: &[u32],
        bufs: &mut MetalDecodeBuffers,
        produce_logits_on_last: bool,
    ) -> Result<()> {
        if prompt_tokens.is_empty() {
            return Ok(());
        }
        let total_chunks = prompt_tokens.len().div_ceil(Self::PREFILL_CHUNK_SIZE);
        for (chunk_idx, chunk) in prompt_tokens.chunks(Self::PREFILL_CHUNK_SIZE).enumerate() {
            let is_last_chunk = chunk_idx + 1 == total_chunks;
            self.forward_token_chunk_into(chunk, bufs, produce_logits_on_last && is_last_chunk)?;
        }
        Ok(())
    }

    fn forward_token_chunk_into(
        &mut self,
        tokens: &[u32],
        bufs: &mut MetalDecodeBuffers,
        produce_logits_on_last: bool,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        if self.kv_cache.len() + tokens.len() > self.kv_cache.max_seq_len {
            bail!(
                "Sequence length exceeded max_seq_len={}",
                self.kv_cache.max_seq_len
            );
        }

        let queue = self.ctx.queue.clone();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        let last = tokens.len() - 1;

        for (i, &token_id) in tokens.iter().enumerate() {
            let pos = self.kv_cache.len();
            ops::embedding_into_encoded(
                &self.ctx,
                &encoder,
                &self.embed_tokens,
                token_id,
                &mut bufs.hidden,
            )?;

            for layer_idx in 0..self.layers.len() {
                self.forward_layer_token_into(layer_idx, pos, bufs, &encoder)?;
            }

            self.kv_cache.increment_seq_len();
            ops::rms_norm_into_encoded(
                &self.ctx,
                &encoder,
                &bufs.hidden,
                &self.norm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
            if produce_logits_on_last && i == last {
                ops::linear_into_encoded(
                    &self.ctx,
                    &encoder,
                    &bufs.normed,
                    &self.embed_tokens,
                    &mut bufs.logits,
                )?;
            }
        }

        encoder.end_encoding();
        cmd.commit();
        Ok(())
    }

    fn forward_one_token_into(
        &mut self,
        token_id: u32,
        bufs: &mut MetalDecodeBuffers,
        produce_logits: bool,
    ) -> Result<()> {
        if self.kv_cache.len() >= self.kv_cache.max_seq_len {
            bail!(
                "Sequence length exceeded max_seq_len={}",
                self.kv_cache.max_seq_len
            );
        }

        let pos = self.kv_cache.len();
        let queue = self.ctx.queue.clone();
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        ops::embedding_into_encoded(&self.ctx, &encoder, &self.embed_tokens, token_id, &mut bufs.hidden)?;

        for layer_idx in 0..self.layers.len() {
            self.forward_layer_token_into(layer_idx, pos, bufs, &encoder)?;
        }

        self.kv_cache.increment_seq_len();

        ops::rms_norm_into_encoded(
            &self.ctx,
            &encoder,
            &bufs.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;
        if produce_logits {
            ops::linear_into_encoded(
                &self.ctx,
                &encoder,
                &bufs.normed,
                &self.embed_tokens,
                &mut bufs.logits,
            )?;
        }
        encoder.end_encoding();
        cmd.commit();
        Ok(())
    }

    fn forward_layer_token_into(
        &mut self,
        layer_idx: usize,
        pos: usize,
        bufs: &mut MetalDecodeBuffers,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<()> {
        {
            let layer = &self.layers[layer_idx];
            ops::rms_norm_into_encoded(
                &self.ctx,
                encoder,
                &bufs.hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
            ops::linear3_into_encoded(
                &self.ctx,
                encoder,
                &bufs.normed,
                &layer.attention.q_proj,
                &layer.attention.k_proj,
                &layer.attention.v_proj,
                &mut bufs.q,
                &mut bufs.k,
                &mut bufs.v,
            )?;
        }

        self.attention_decode_into(
            layer_idx,
            pos,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            &mut bufs.attn_out,
            encoder,
        )?;
        {
            let layer = &self.layers[layer_idx];
            ops::linear_accum_inplace_encoded(
                &self.ctx,
                encoder,
                &bufs.attn_out,
                &layer.attention.o_proj,
                &mut bufs.hidden,
            )?;
        }

        {
            let layer = &self.layers[layer_idx];
            ops::rms_norm_into_encoded(
                &self.ctx,
                encoder,
                &bufs.hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
            ops::linear2_into_encoded(
                &self.ctx,
                encoder,
                &bufs.normed,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &mut bufs.gate,
                &mut bufs.up,
            )?;
            ops::silu_mul_into_encoded(&self.ctx, encoder, &bufs.gate, &bufs.up, &mut bufs.mlp_act)?;
            ops::linear_accum_inplace_encoded(
                &self.ctx,
                encoder,
                &bufs.mlp_act,
                &layer.mlp.down_proj,
                &mut bufs.hidden,
            )?;
        }
        Ok(())
    }

    fn attention_decode_into(
        &mut self,
        layer_idx: usize,
        pos: usize,
        q_raw: &MetalDeviceVec,
        k_raw: &MetalDeviceVec,
        v_raw: &MetalDeviceVec,
        out: &mut MetalDeviceVec,
        encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        if num_heads % num_kv_heads != 0 {
            bail!(
                "num_attention_heads {} not divisible by num_key_value_heads {}",
                num_heads,
                num_kv_heads
            );
        }
        let gqa_ratio = num_heads / num_kv_heads;
        let head_dim = self.config.head_dim;
        if head_dim > 256 {
            bail!(
                "head_dim {} exceeds Metal decode kernel limit 256",
                head_dim
            );
        }
        let seq_len = pos + 1;
        let scale = 1.0 / (head_dim as f32).sqrt();
        debug_assert!(gqa_ratio > 0);
        let (q_norm_w, k_norm_w) = {
            let layer = &self.layers[layer_idx];
            (&layer.attention.q_norm, &layer.attention.k_norm)
        };

        {
            let layer_cache = &mut self.kv_cache.layers[layer_idx];
            ops::prepare_kv_cache_decode_encoded(
                &self.ctx,
                encoder,
                k_raw,
                v_raw,
                k_norm_w,
                &self.cos_cache,
                &self.sin_cache,
                &mut layer_cache.k,
                &mut layer_cache.v,
                num_kv_heads,
                head_dim,
                pos,
                self.kv_cache.max_seq_len,
                self.config.rms_norm_eps,
            )?;
        }

        let layer_cache = &self.kv_cache.layers[layer_idx];
        ops::attention_decode_heads_encoded(
            &self.ctx,
            encoder,
            q_raw,
            q_norm_w,
            &self.cos_cache,
            &self.sin_cache,
            &layer_cache.k,
            &layer_cache.v,
            out,
            num_heads,
            num_kv_heads,
            head_dim,
            pos,
            seq_len,
            self.kv_cache.max_seq_len,
            scale,
            self.config.rms_norm_eps,
        )?;
        Ok(())
    }

    fn take_decode_bufs(&mut self) -> Result<MetalDecodeBuffers> {
        if let Some(bufs) = self.decode_bufs.take() {
            Ok(bufs)
        } else {
            MetalDecodeBuffers::new(&self.ctx, &self.config)
        }
    }
}
