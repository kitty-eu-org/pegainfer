#[cfg(not(feature = "metal"))]
use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
#[cfg(feature = "metal")]
use pegainfer::metal_backend::model::Qwen3MetalModel;
use pegainfer::tokenizer::Tokenizer;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "metal_align_dump")]
struct Args {
    #[arg(long)]
    model_path: String,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
}

#[derive(Serialize)]
struct TopItem {
    id: usize,
    logit: f32,
    text: String,
}

#[derive(Serialize)]
struct Output {
    prompt: String,
    prompt_tokens: Vec<u32>,
    top: Vec<TopItem>,
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k.min(idx.len()));
    idx
}

fn main() -> Result<()> {
    #[cfg(not(feature = "metal"))]
    return Err(anyhow!("metal_align_dump requires --features metal"));

    #[cfg(feature = "metal")]
    {
        let args = Args::parse();
        let tokenizer = Tokenizer::from_file(&args.model_path)?;
        let mut model = Qwen3MetalModel::from_safetensors(&args.model_path)?;
        let prompt_tokens = tokenizer.encode(&args.prompt)?;
        let logits = model.debug_last_logits_for_prompt(&prompt_tokens)?;
        let top_ids = top_k_indices(&logits, args.top_k);

        let mut top = Vec::with_capacity(top_ids.len());
        for &id in &top_ids {
            let text = tokenizer
                .decode(&[id as u32])
                .unwrap_or_else(|_| "<decode_err>".to_string());
            top.push(TopItem {
                id,
                logit: logits[id],
                text,
            });
        }

        let out = Output {
            prompt: args.prompt,
            prompt_tokens,
            top,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        Ok(())
    }
}
