# llama.cpp vs pegainfer (Metal) Server Benchmark (2026-03-06)

## Goal

Use the same `server + curl` flow to compare end-to-end throughput (`prefill + decode`) between:

- `llama.cpp` (`llama-server`)
- `pegainfer` (`--backend metal`)

## Environment

- Date: 2026-03-06
- Machine: Apple M4 (Metal)
- Model (llama.cpp): `/Users/hezhaozhao/myself/models/Qwen3-0.6B-F16-GGUF/Qwen3-0.6B-F16.gguf`
- Model (pegainfer): `/Users/hezhaozhao/myself/models/Qwen3-0.6B`

## Server Startup

### llama.cpp

```bash
llama-server \
  -m /Users/hezhaozhao/myself/models/Qwen3-0.6B-F16-GGUF/Qwen3-0.6B-F16.gguf \
  --host 127.0.0.1 --port 8011 -ngl 999 -fa on -t 4 --log-disable
```

### pegainfer

```bash
cargo run --release --no-default-features --features metal --bin pegainfer -- \
  --backend metal --model-path /Users/hezhaozhao/myself/models/Qwen3-0.6B --port 8001
```

## Request Settings

- Endpoint: `POST /v1/completions`
- Params:
  - `max_tokens=128`
  - `temperature=0.0`
  - `top_k=1`
- 5 runs for each case

Two prompts:

1. Short prompt: `"2+2="` (about 4 prompt tokens)
2. Long prompt: 512x `"benchmark "` (about 513 prompt tokens)

## Metrics

- `completion_tokps = completion_tokens / time_total`
- `total_tokps = (prompt_tokens + completion_tokens) / time_total` (overall prefill+decode)

## Results

| System | Prompt Tokens | completion_tokps | total_tokps |
| --- | ---: | ---: | ---: |
| llama.cpp (`llama-server`, FA on) short | 4 | 70.278684 | 72.474892 |
| pegainfer (metal) short | 4 | 66.031798 | 68.095292 |
| llama.cpp (`llama-server`, FA on) long | 513 | 65.035204 | 325.684109 |
| pegainfer (metal) long | 513 | 14.064881 | 70.434286 |

## Key Findings

- Short prompt: pegainfer is close to llama.cpp (~94% of llama.cpp completion throughput).
- Long prompt: pegainfer is much slower in end-to-end throughput (~4.6x gap in `total_tokps`).
- Main bottleneck is prefill path for long prompts.

## Interpretation

Current Metal path in pegainfer is decode-style token-by-token even for prompt ingestion, which causes large overhead on long prefill workloads compared with llama.cpp's optimized prompt processing path.

## Optimization Update (same day)

Implemented prompt prefill submission optimization in Metal backend:

- Prompt ingestion changed from per-token command-buffer submission to chunked submission.
- Current kept setting: `PREFILL_CHUNK_SIZE = 256`.
- For non-greedy prefill, logits are computed only on the final prompt token.

### pegainfer re-test after optimization (best observed)

Same `server + curl` methodology and request parameters:

| System | Prompt Tokens | completion_tokps | total_tokps |
| --- | ---: | ---: | ---: |
| pegainfer (metal, optimized) short | 4 | 67.195883 | 69.295755 |
| pegainfer (metal, optimized) long | 513 | 15.090782 | 75.571804 |

### Additional confirmation under sustained load

Later repeated long-prompt runs (same build, 3 runs) showed:

| System | Prompt Tokens | completion_tokps | total_tokps |
| --- | ---: | ---: | ---: |
| pegainfer (metal, optimized, sustained) long | 513 | 14.129177 | 70.756268 |

This indicates notable run-to-run variability (likely thermal/power/system-load effects), but the optimization direction is still prefill-focused.

### Delta vs previous pegainfer baseline (best observed)

- Short prompt:
  - `completion_tokps`: `66.031798 -> 67.195883` (+1.76%)
  - `total_tokps`: `68.095292 -> 69.295755` (+1.76%)
- Long prompt:
  - `completion_tokps`: `14.064881 -> 15.090782` (+7.29%)
  - `total_tokps`: `70.434286 -> 75.571804` (+7.29%)
