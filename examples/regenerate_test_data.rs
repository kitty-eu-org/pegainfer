//! Regenerate test_data/Qwen3-4B.json using pegainfer's greedy output.
//! Run: cargo run -r --example regenerate_test_data

#[cfg(feature = "cuda")]
use pegainfer::model::Qwen3Model;
#[cfg(feature = "cuda")]
use pegainfer::sampler::SamplingParams;
#[cfg(feature = "cuda")]
use pegainfer::tokenizer::Tokenizer;
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use rand::SeedableRng;
#[cfg(feature = "cuda")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
#[cfg(feature = "cuda")]
const OUTPUT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/Qwen3-4B.json");

#[cfg(feature = "cuda")]
#[derive(Deserialize)]
struct InputData {
    model_name: String,
    cases: Vec<InputCase>,
}

#[cfg(feature = "cuda")]
#[derive(Deserialize)]
struct InputCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    #[allow(dead_code)]
    output: String,
}

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct OutputData {
    model_name: String,
    engine: String,
    cases: Vec<OutputCase>,
}

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct OutputCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    output: String,
}

#[cfg(feature = "cuda")]
fn main() {
    pegainfer::logging::init_stderr("info");

    let content = std::fs::read_to_string(OUTPUT_PATH).expect("Failed to read test data");
    let input: InputData = serde_json::from_str(&content).expect("Failed to parse JSON");

    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    let mut model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load model");
    let greedy = SamplingParams::default();
    let mut rng = StdRng::seed_from_u64(42);

    let mut cases = Vec::new();
    for case in &input.cases {
        let prompt_tokens = tokenizer.encode(&case.prompt).expect("encode failed");
        let output_tokens = model
            .generate(&prompt_tokens, case.max_new_tokens, &greedy, &mut rng)
            .expect("generate failed");

        let new_tokens = &output_tokens[prompt_tokens.len()..];
        let output_text = tokenizer.decode(new_tokens).expect("decode failed");

        eprintln!("[{}] prompt={:?}", case.name, case.prompt);
        eprintln!("  output={:?}", output_text);

        cases.push(OutputCase {
            name: case.name.clone(),
            prompt: case.prompt.clone(),
            max_new_tokens: case.max_new_tokens,
            output: output_text,
        });
    }

    let output = OutputData {
        model_name: input.model_name,
        engine: "pegainfer".to_string(),
        cases,
    };

    let json = serde_json::to_string_pretty(&output).expect("serialize failed");
    std::fs::write(OUTPUT_PATH, json).expect("Failed to write test data");
    eprintln!("\nWrote {OUTPUT_PATH}");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("`regenerate_test_data` requires `--features cuda`.");
}
