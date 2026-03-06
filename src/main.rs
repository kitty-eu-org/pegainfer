use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use log::info;
use pegainfer::http_server::build_app;
use pegainfer::logging;
use pegainfer::server_engine::{Backend, EngineOptions, load_engine};
use pegainfer::trace_reporter::FileReporter;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

fn default_backend_arg() -> BackendArg {
    default_backend_arg_impl()
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn default_backend_arg_impl() -> BackendArg {
    BackendArg::Cuda
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn default_backend_arg_impl() -> BackendArg {
    BackendArg::Metal
}

#[cfg(all(feature = "cuda", feature = "metal"))]
fn default_backend_arg_impl() -> BackendArg {
    if cfg!(target_os = "macos") {
        BackendArg::Metal
    } else {
        BackendArg::Cuda
    }
}

impl From<BackendArg> for Backend {
    fn from(value: BackendArg) -> Self {
        match value {
            #[cfg(feature = "cuda")]
            BackendArg::Cuda => Backend::Cuda,
            #[cfg(feature = "metal")]
            BackendArg::Metal => Backend::Metal,
        }
    }
}

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3 inference server")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Runtime backend
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// Model path (local model directory)
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: String,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();
    let backend = args.backend.unwrap_or_else(default_backend_arg);
    let engine_options = EngineOptions {
        enable_cuda_graph: args.cuda_graph,
        backend: backend.into(),
    };

    if let Some(ref trace_path) = args.trace_output_path {
        std::fs::create_dir_all(trace_path).expect("Failed to create trace output directory");
        fastrace::set_reporter(
            FileReporter::new(trace_path.clone()),
            fastrace::collector::Config::default(),
        );
        info!("Tracing enabled: output_dir={}", trace_path.display());
    }

    info!("=== Rust LLM Server - Qwen3 ===");
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: backend={}, cuda_graph={}, model_path={}",
        engine_options.backend.as_str(),
        engine_options.enable_cuda_graph,
        args.model_path
    );
    let engine = load_engine(&args.model_path, 42, engine_options).expect("Failed to load engine");
    info!(
        "Engine loaded: elapsed_ms={}, vocab_size={}",
        start.elapsed().as_millis(),
        engine.vocab_size()
    );

    let app = build_app(engine);
    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    if args.trace_output_path.is_some() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
}
