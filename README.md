# QwenASR API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

An [OpenAI Speech to Text API](https://platform.openai.com/docs/api-reference/audio/createTranscription) compatible wrapper for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

Models are dynamically loaded on demand and automatically unloaded after a configurable period of inactivity (default: 300 seconds). Designed for use with Docker.

## Model Loading Behavior

Two independent model slots are managed:

**ASR model slot**
- Only one ASR model is kept in VRAM at a time.
- When a request specifies a model that is not currently loaded, the model is downloaded from HuggingFace (if not cached) and loaded into VRAM.
- All requests are processed sequentially. If a request arrives while another is being processed, it waits in queue.
- The inactivity timer resets after each request completes. When the timer expires, the model is unloaded from VRAM.

**ForcedAligner slot** (only when `ENABLE_TIMESTAMPS=true`)
- Manages a single ForcedAligner model (`ALIGNER_MODEL`) independently from the ASR slot.
- Dynamically loaded on the first request requiring timestamps and unloaded after inactivity.
- When timestamps are requested, both the ASR model and ForcedAligner may be in VRAM simultaneously.

Downloaded models are cached on disk and reused across container restarts via a mounted volume.

## API

### `POST /v1/audio/transcriptions`

| Parameter | Support |
|---|---|
| `file` | Supported |
| `model` | Supported (see note below) |
| `language` | Supported |
| `prompt` | Supported |
| `response_format` | Supported |
| `temperature` | Supported |
| `timestamp_granularities[]` | Supported when `ENABLE_TIMESTAMPS=true` |
| `stream` | Not supported |

**`model` field:** Any Qwen3-ASR model name from HuggingFace can be specified (e.g. `Qwen/Qwen3-ASR-1.7B`). Specifying `whisper-1` uses the model defined by the `DEFAULT_MODEL` environment variable.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Model to use when `whisper-1` is specified |
| `MODEL_UNLOAD_TIMEOUT` | `300` | Seconds of inactivity before unloading the model |
| `DEVICE` | `cuda` | Compute device: `cuda`, `cpu`, or `mps` |
| `ENABLE_TIMESTAMPS` | `false` | Enable `timestamp_granularities[]` support |
| `ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | ForcedAligner model (used when `ENABLE_TIMESTAMPS=true`) |

## Docker

Two Dockerfiles are provided:

| Dockerfile | Base image | Target |
|---|---|---|
| `Dockerfile` | `pytorch/pytorch` (CUDA) | Linux x86_64 with NVIDIA GPU |
| `Dockerfile.cpu` | `python:3.11-slim` (multi-arch) | Linux x86_64/ARM64, macOS (Docker) |

Compose files are located in the `docker-compose/` directory:

| File | Role |
|---|---|
| `docker-compose/compose.yml` | Base (ports, volumes) |
| `docker-compose/compose.cuda.yml` | CUDA — pulls `arianpg/qwen-asr-api:cuda` from Docker Hub |
| `docker-compose/compose.cpu.yml` | CPU — pulls `arianpg/qwen-asr-api:cpu` from Docker Hub |
| `docker-compose/compose.dev.yml` | Build override for CUDA (local development) |
| `docker-compose/compose.dev.cpu.yml` | Build override for CPU (local development) |
| `docker-compose/compose.mps.yml` | MPS — for native macOS execution (non-Docker) |

> **Note:** MPS (Metal) is not accessible inside Docker on macOS as containers run in a Linux VM.
> `docker-compose/compose.mps.yml` and `DEVICE=mps` are intended for running the server directly (outside Docker) on Apple Silicon.

```bash
# Pull from Docker Hub — CUDA
docker compose -f docker-compose/compose.yml -f docker-compose/compose.cuda.yml up

# Pull from Docker Hub — CPU (Linux ARM64 / macOS Docker)
docker compose -f docker-compose/compose.yml -f docker-compose/compose.cpu.yml up

# Build locally — CUDA
docker compose -f docker-compose/compose.yml -f docker-compose/compose.cuda.yml -f docker-compose/compose.dev.yml up --build

# Build locally — CPU
docker compose -f docker-compose/compose.yml -f docker-compose/compose.cpu.yml -f docker-compose/compose.dev.cpu.yml up --build
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Built with the assistance of [Claude Code](https://claude.ai/code).
