//! Safetensors loading helpers for Metal backend.

use std::collections::HashMap;
use std::fs;

use anyhow::{anyhow, Result};
use half::{bf16, f16};
use safetensors::SafeTensors;

use crate::metal_backend::tensor::{MetalDeviceContext, MetalDeviceMatrix, MetalDeviceVec};

/// Load shard metadata. Returns (shard_file_paths, weight_map: tensor_name -> shard_index)
pub fn load_shard_info(model_path: &str) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let single_path = format!("{}/model.safetensors", model_path);
    if std::path::Path::new(&single_path).exists() {
        return Ok((vec![single_path], HashMap::new()));
    }

    let index_path = format!("{}/model.safetensors.index.json", model_path);
    let index_content = fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map_json = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("Invalid index.json: missing weight_map"))?;

    let mut shard_files: Vec<String> = Vec::new();
    let mut file_to_idx: HashMap<String, usize> = HashMap::new();
    let mut weight_map: HashMap<String, usize> = HashMap::new();

    for (tensor_name, shard_file_val) in weight_map_json {
        let shard_file = shard_file_val.as_str().unwrap().to_string();
        let idx = if let Some(&idx) = file_to_idx.get(&shard_file) {
            idx
        } else {
            let idx = shard_files.len();
            shard_files.push(format!("{}/{}", model_path, &shard_file));
            file_to_idx.insert(shard_file, idx);
            idx
        };
        weight_map.insert(tensor_name.clone(), idx);
    }

    Ok((shard_files, weight_map))
}

pub fn find_tensor<'a>(
    shards: &'a [SafeTensors<'a>],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'a>> {
    if let Some(&idx) = weight_map.get(name) {
        shards[idx]
            .tensor(name)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))
    } else {
        // Fallback: try all shards (single-file case)
        for shard in shards {
            if let Ok(t) = shard.tensor(name) {
                return Ok(t);
            }
        }
        Err(anyhow!("Tensor '{}' not found in any shard", name))
    }
}

pub fn load_tensor_1d(
    ctx: &MetalDeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<MetalDeviceVec> {
    let tensor = find_tensor(shards, weight_map, name)?;
    MetalDeviceVec::from_safetensors(ctx, tensor.data())
}

pub fn load_tensor_1d_f32(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<Vec<f32>> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let data = tensor.data();
    if data.len() % 2 != 0 {
        return Err(anyhow!(
            "Tensor '{}' byte len must be even for bf16: {}",
            name,
            data.len()
        ));
    }
    let len = data.len() / 2;
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
    Ok(slice.iter().map(|v| v.to_f32()).collect())
}

pub fn load_tensor_2d(
    ctx: &MetalDeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
) -> Result<MetalDeviceMatrix> {
    let tensor = find_tensor(shards, weight_map, name)?;
    let shape = tensor.shape();
    MetalDeviceMatrix::from_safetensors(ctx, tensor.data(), shape[0], shape[1])
}

/// Precompute RoPE cos/sin cache as contiguous host buffers.
/// Layout: [max_seq_len * head_dim] — position `pos` at offset `pos * head_dim`.
pub fn precompute_rope_host(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();

    let total = max_seq_len * head_dim;
    let mut cos_host = vec![0.0f32; total];
    let mut sin_host = vec![0.0f32; total];

    for pos in 0..max_seq_len {
        let base = pos * head_dim;
        for (i, inv) in inv_freq.iter().enumerate() {
            let freq = pos as f32 * *inv;
            let cos_val = freq.cos();
            let sin_val = freq.sin();
            // Half-split layout: [cos(0)..cos(63), cos(0)..cos(63)]
            cos_host[base + i] = cos_val;
            cos_host[base + i + half_dim] = cos_val;
            sin_host[base + i] = sin_val;
            sin_host[base + i + half_dim] = sin_val;
        }
    }

    Ok((cos_host, sin_host))
}

/// Precompute RoPE cache and upload to Metal as fp16 buffers.
pub fn precompute_rope(
    ctx: &MetalDeviceContext,
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
) -> Result<(MetalDeviceVec, MetalDeviceVec)> {
    let (cos_host, sin_host) = precompute_rope_host(head_dim, max_seq_len, theta)?;
    let cos_fp16: Vec<f16> = cos_host.iter().map(|&x| f16::from_f32(x)).collect();
    let sin_fp16: Vec<f16> = sin_host.iter().map(|&x| f16::from_f32(x)).collect();
    let cos = MetalDeviceVec::from_host_private(ctx, &cos_fp16)?;
    let sin = MetalDeviceVec::from_host_private(ctx, &sin_fp16)?;
    Ok((cos, sin))
}
