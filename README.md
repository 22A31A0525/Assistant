# Personal Assistant

A modular, high-performance AI Assistant architecture powered by remote GPU inference on Kaggle (Dual Tesla T4) and custom local persona workflows.

---

## 📌 Project Architecture

The repository is structured around a root configuration, local testing harnesses, and a modular `kaggle/` backend subsystem:

```text
Assistant/
├── .env                       # Local secrets & API tokens (git-ignored)
├── .gitignore                 # Exclusion rules for secrets, caches, and logs
├── README.md                  # Project overview & architectural guide
├── assistant_persona/         # Persona profiles, system prompts & chat bindings
├── tests/                     # Unit test suites (pytest)
│   ├── test_deploy_notebook.py
│   └── test_model_compatibility.py
└── kaggle/                    # Dedicated Kaggle remote GPU backend package
    ├── __init__.py
    ├── cli.py                 # Unified Kaggle Hub CLI (python -m kaggle.cli)
    ├── chat.py                # Interactive streaming CLI client with remote GPU
    ├── config/
    │   ├── auth_config/       # Multi-account credentials & active switcher
    │   │   ├── manager.py     # Account loader, synchronizer & validator
    │   │   └── models.py      # Account data models & slug generator
    │   ├── model_config/      # Curated 3-model catalog, manifests & profiles
    │   │   ├── manifest.py    # Subtask 2 ModelCompatibilityManifest & CapabilityState
    │   │   ├── catalog.py     # Registry with fixed owner datasets & vision mmproj
    │   │   ├── common.py      # Shared CUDA 12 & runtime wheel datasets
    │   │   └── profile.py     # ModelProfile dataclass (text & multimodal)
    │   └── api_end_point_config/ # Endpoint resolver, ping probe & tunnel sync
    │       ├── llama_config.py
    │       └── tunnel.py
    ├── deploy_notebook/       # Modular Remote Deployment Subsystem
    │   ├── environment.py     # Strict 2x T4 check, CUDA 12 packages, & physical VRAM query
    │   ├── model_resolver.py  # Binary GGUF metadata parser & cautious tensor logging
    │   ├── gpu_allocator.py   # Mathematical KV cache estimation & admission check
    │   ├── server_runner.py   # Single authoritative command builder, readiness probe, & role validation
    │   ├── auth_gateway.py    # Zero-dependency reverse proxy with Bearer auth & chunked SSE streaming
    │   ├── metadata.py        # Generates kernel-metadata.json with shared wheels
    │   ├── deployer.py        # Bundles sub-modules into self-contained worker.py & pushes to Kaggle
    │   └── worker.py          # Auto-compiled standalone runner executed on remote container
    └── tracker/
        └── kernel_stopper.py  # Quota management: remote /v1/shutdown & Kaggle CLI kernel cleanup
```

---

## 🚀 Subtask 1: Dual-T4 Remote Infrastructure

### 1. Unified Kaggle Control Center
Run from the workspace root:
```bash
python -m kaggle.cli
```
Operations available:
- `[1]` **Authentication & Multi-Account Switcher**: Seamlessly switch between configured accounts (`alluriakshay`, `rohitvijayadapa`, `saitejavenkatavinay`) and sync `kaggle.json`.
- `[2]` **Model Catalog**: Inspect architectures, layers, KV heads, and context targets.
- `[3]` **Test Kaggle API Connection**: Validate active account credentials.
- `[4]` **Ping Remote Tunnel Endpoint**: Check health status (`/v1/models`) and sync active Cloudflare URL to `.env`.
- `[5]` **Live Chat with Active Model**: Token-by-token streaming chat session.
- `[6]` **Kernel Deployment**: One-click bundle and push to remote Dual-T4 GPU worker.
- `[7]` **Kernel Execution Logs & Status**: Monitor live run state.
- `[8]` **GPU Stopper & Quota Releaser**: Safely terminate remote worker and release weekly GPU quota.

### 2. Dual Tesla T4 Inference Engine (`llama-server`)
- **Distribution:** `--split-mode layer` evenly splitting weights and attention layers across both T4 GPUs.
- **Context Capacity:** 65,536 tokens admitted with dedicated `q8_0` KV cache quantization.
- **Single Inference Slot:** Configured with `--parallel 1` to dedicate entire KV cache to a single-user Codex assistant without slot memory fragmentation.
- **Context Efficiency:** `--no-reasoning-preserve` strips intermediate reasoning history across conversational turns while preserving full native CoT reasoning during token generation.
- **Flash Attention & Optimizations:** Uses `--flash-attn on`, `--batch-size 2048`, and `--ubatch-size 512`.

### 3. Authenticated Gateway, Chunked SSE Streaming & Health Safeguards (`auth_gateway.py`)
- Reverse proxy running on port 4000 in front of `llama-server:8080`.
- **Security:** Rejects unauthorized requests with `HTTP 401 Unauthorized` using Bearer authentication (`LITELLM_MASTER_KEY`).
- **Real-Time Streaming:** Implements true HTTP/1.1 `Transfer-Encoding: chunked` for Server-Sent Events (`text/event-stream`), delivering real-time token streaming to local clients.
- **Upstream Resilience:** Explicitly returns `503 Service Unavailable` with `Retry-After: 5` header when the underlying inference server is starting up or unreachable.
- **Remote Shutdown Endpoint:** Exposes authenticated `POST /v1/shutdown` to allow remote self-termination (`os._exit(0)`), instantly terminating the remote GPU container.

### 4. Hardware Verification & Resource Monitoring (`environment.py` & `gpu_allocator.py`)
- **Strict Topology Check:** Requires exactly 2x NVIDIA Tesla T4 GPUs before proceeding.
- **Per-GPU Memory Admission:** Mathematical allocation check verifying both GPU 0 and GPU 1 remain under their respective 14.0 GB safe limits (~1.8 GB safety headroom per card).
- **Physical GPU VRAM Monitoring:** Post-startup `nvidia-smi` query reporting exact real-time VRAM allocation (`used_mib` / `total_mib`).
- **Host CPU RAM Visibility:** Inspects `/proc/meminfo` at startup and readiness to track host memory, Linux page cache (`Buffers + Cached`), and swap pressure without external dependencies.

---

## 🧠 Subtask 2: Model Loading & Inference Engine Architecture

### 1. Model Compatibility Manifest (`manifest.py`)
A formal metadata and capability tracker governing how `Qwen3.8-27B` is interpreted and executed:
- **Dynamic Binary GGUF Inspection:** Reads header fields directly from the file (detecting `architecture: qwen3`, layer count, and embedded template presence) without hardcoded assumptions.
- **Template Source Classification:** Tracks whether the chat template is `EMBEDDED_GGUF_TEMPLATE`, `RUNTIME_BUILTIN_FALLBACK`, or `UNVERIFIED_TEMPLATE`.

### 2. Independent Capability Gating
Decouples model features so unverified modes cannot fail silently:
- `text`: `VERIFIED_BASIC`
- `vision_simple`: `VERIFIED_BASIC` (authorized upon passing semantic visual evaluation)
- `vision_complex`: `UNVERIFIED` (remains gated for dedicated multi-image and screenshot verification)
- `tool_calling`: `UNVERIFIED` (remains gated for Codex agent multi-turn validation)

### 3. Message Role & Tool Schema Formatting Verification (`server_runner.py`)
- Validates that `system`, `user`, and `assistant` role boundaries transition cleanly in the model's native Jinja template.
- Submits OpenAI-standard function tool definitions (`tools: [...]`) to confirm that Qwen's template engine parses and accepts tool calling schemas without crashing.

### 4. Handling Architectural Warnings
- **Auxiliary Tensors (`blk.64`):** Categorized conservatively as `DETECTED_UNVERIFIED_AUXILIARY_LAYER`. In Qwen 3/3.8 architectures, this is the trailing MTP (Multi-Token Prediction) speculative drafting block. Kept safely disabled (`--spec-type none` / no draft flags) to prevent memory index assertion crashes.
- **Multimodal Token Positions (`non-consecutive token position`):** Documented as an expected artifact of Qwen-VL's M-RoPE 2D/3D grid image patch coordinates during sub-batching. Basic semantic output (*"Solid red square image"*) verified.

### 5. Curated Model Catalog (`kaggle/config/model_config/`)

| Model | Capability | Context | Primary Kaggle Source | Vision Projector (`mmproj`) |
|---|---|---|---|---|
| **Ornith-1.0-35B** | Pure Text / Reasoning | 64K | `rohitvijayadapa/ornith-1-0-35b` | None |
| **Qwen-3.6-35B** | Multimodal (Vision) | 64K | `saitejavenkatavinay/qwen36-35b-a3b-q4km` | `saitejavenkatavinay/qwen36-35b-mmproj` |
| **Qwen-3.8-27B** | Multimodal (Vision) | 64K | `saitejavenkatavinay/ornith-qwen38-27b` | `mmproj-Qwen3.8-27B-f16.gguf` |

---

## 🧪 Testing & Verification

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```
**Test Coverage (10/10 Passing):**
- Native `llama-server` command flag generation (`--parallel 1`, `--no-reasoning-preserve`, `--flash-attn on`, `--ctx-size 65536`).
- Hardware VRAM and system host RAM inspection fallbacks.
- Bearer token security boundary & 401 gatekeeper.
- Per-GPU memory allocation & mathematical layer distribution check.
- Manifest dynamic metadata recording & independent capability transitions.
- Chat template source classification (`EMBEDDED_GGUF_TEMPLATE`).
