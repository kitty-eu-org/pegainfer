use anyhow::Result;
use tokio::sync::mpsc::UnboundedSender;

use crate::sampler::SamplingParams;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
        }
    }
}

impl Default for Backend {
    fn default() -> Self {
        default_backend()
    }
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn default_backend() -> Backend {
    Backend::Cuda
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn default_backend() -> Backend {
    Backend::Metal
}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn default_backend() -> Backend {
    if cfg!(target_os = "macos") {
        Backend::Metal
    } else {
        Backend::Cuda
    }
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub enable_cuda_graph: bool,
    pub backend: Backend,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            backend: Backend::default(),
        }
    }
}

pub struct CompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct CompleteOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

pub struct StreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
}

pub trait ServerEngine: Send {
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput>;

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()>;

    fn vocab_size(&self) -> usize;
}

pub fn load_engine(
    model_path: &str,
    seed: u64,
    options: EngineOptions,
) -> Result<Box<dyn ServerEngine>> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let _ = (model_path, seed);

    match options.backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Box::new(CudaServerEngine::load_with_options(
            model_path, seed, options,
        )?)),
        #[cfg(feature = "metal")]
        Backend::Metal => Ok(Box::new(MetalServerEngine::load_with_options(
            model_path, seed, options,
        )?)),
    }
}

#[cfg(feature = "cuda")]
use rand::SeedableRng;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use rand::SeedableRng;
#[cfg(feature = "cuda")]
use rand::rngs::StdRng;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use rand::rngs::StdRng;

#[cfg(feature = "cuda")]
use crate::model::{ModelRuntimeConfig, Qwen3Model};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
pub struct CudaServerEngine {
    model: Qwen3Model,
    tokenizer: Tokenizer,
    rng: StdRng,
}

#[cfg(feature = "cuda")]
impl CudaServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)?;
        let model = Qwen3Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.enable_cuda_graph,
            },
        )?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            model,
            tokenizer,
            rng,
        })
    }
}

#[cfg(feature = "cuda")]
impl ServerEngine for CudaServerEngine {
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let output_tokens =
            self.model
                .generate(&prompt_tokens, req.max_tokens, &req.sampling, &mut self.rng)?;
        let completion_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
        let text = self
            .tokenizer
            .decode(&output_tokens[prompt_tokens.len()..])?;
        let finish_reason = if completion_tokens >= req.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let usage = Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: output_tokens.len(),
        };
        Ok(CompleteOutput {
            text,
            finish_reason,
            usage,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;

        // TODO: Buffer incomplete subword/UTF-8 sequences before sending deltas.
        let stats = self.model.generate_streaming_with_callback(
            &prompt_tokens,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| {
                let text_delta = self.tokenizer.decode(&[token_id]).unwrap_or_else(|e| {
                    log::warn!("Failed to decode token {}: {}", token_id, e);
                    "\u{FFFD}".to_string()
                });
                tx.send(StreamDelta {
                    text_delta,
                    finish_reason: None,
                })
                .is_ok()
            },
        )?;

        if stats.consumer_dropped {
            return Ok(());
        }

        let finish_reason = if stats.hit_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let _ = tx.send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
        });

        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(feature = "cuda")]
pub type RealServerEngine = CudaServerEngine;

#[cfg(feature = "metal")]
use crate::metal_backend::model::Qwen3MetalModel;

#[cfg(feature = "metal")]
pub struct MetalServerEngine {
    model: Qwen3MetalModel,
    tokenizer: Tokenizer,
    rng: StdRng,
}

#[cfg(feature = "metal")]
impl MetalServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let _ = options;
        let model = Qwen3MetalModel::from_safetensors(model_path)?;
        let tokenizer = Tokenizer::from_file(model_path)?;
        let rng = StdRng::seed_from_u64(seed);

        Ok(Self {
            model,
            tokenizer,
            rng,
        })
    }
}

#[cfg(feature = "metal")]
impl ServerEngine for MetalServerEngine {
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let output_tokens =
            self.model
                .generate(&prompt_tokens, req.max_tokens, &req.sampling, &mut self.rng)?;
        let completion_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
        let text = self
            .tokenizer
            .decode(&output_tokens[prompt_tokens.len()..])?;
        let finish_reason = if completion_tokens >= req.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let usage = Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: output_tokens.len(),
        };
        Ok(CompleteOutput {
            text,
            finish_reason,
            usage,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;

        let stats = self.model.generate_streaming_with_callback(
            &prompt_tokens,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| {
                let text_delta = self.tokenizer.decode(&[token_id]).unwrap_or_else(|e| {
                    log::warn!("Failed to decode token {}: {}", token_id, e);
                    "\u{FFFD}".to_string()
                });
                tx.send(StreamDelta {
                    text_delta,
                    finish_reason: None,
                })
                .is_ok()
            },
        )?;

        if stats.consumer_dropped {
            return Ok(());
        }

        let finish_reason = if stats.hit_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        let _ = tx.send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
        });
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
pub type RealServerEngine = MetalServerEngine;
