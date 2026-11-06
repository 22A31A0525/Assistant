# Personal Assistant

A modular, high-performance AI Assistant architecture powered by remote GPU inference on Kaggle (Dual Tesla T4) and custom local persona workflows.

---

## 📌 Project Architecture

The repository is structured around a root configuration, local testing harnesses, and a modular `kaggle/` backend subsystem:

```text
Assistant/
├── .env                       # Local secrets, active tunnel URL & API tokens (git-ignored)
├── .gitignore                 # Exclusion rules for secrets, caches, and logs
├── README.md                  # Project overview & architectural guide
├── assistant_persona/         # Persona profiles, system prompts & chat bindings
├── tests/                     # Unit test suites (pytest)
│   ├── test_deploy_notebook.py
│   ├── test_model_compatibility.py
│   ├── test_subtask3_memory_optimization.py
│   ├── test_subtask4_runtime.py
│   └── test_subtask5_gateway.py
└── kaggle/                    # Dedicated Kaggle remote GPU backend package
    ├── __init__.py
    ├── cli.py                 # Unified Kaggle Hub CLI (python -m kaggle.cli)
    ├── chat.py                # Interactive streaming CLI client with /think toggle & 85% context monitor
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
    │   ├── gpu_allocator.py   # Subtask 3 hybrid 16/48 attention & DeltaNet state accounting
    │   ├── latency_telemetry.py # Subtask 4 RequestClock & streaming TTFT latency instrumentation
    │   ├── context_admission.py # Subtask 4 model-aware token admission guard & fail-closed validation
    │   ├── runtime_policy.py  # Conservative CPU threading & microbatch flags policy
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

---

## ⚡ Subtask 3: KV Cache, Attention & GPU Memory Optimization

### 1. Hybrid 16/48 Architecture-Aware Memory Accounting (`gpu_allocator.py`)
Qwen3.8-27B employs a hybrid attention structure:
- **16 Full-Attention Layers:** Token-indexed KV cache. At 65,536 tokens with Q8_0 precision ($2 \times 16 \times 4 \times 256 \times 65536 \times 1.0625$ bytes), this occupies **2,176.0 MiB (~2.125 GiB)**.
- **48 Gated DeltaNet Recurrent Layers:** Matrix hidden state memory scaling as $\mathcal{O}(1)$ independent of context length ($n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}}^2 \times 2\text{ bytes}$).
- **Elimination of Zero-State Undercounting:** The allocator never assumes 0.0 MiB for recurrent state. It applies a mathematically verified baseline ($24.0\text{ MiB}$) tagged as `CONSERVATIVE_CALCULATED_ALLOWANCE` unless empirical runtime buffers are configured.

### 2. Flash Attention & CUDA Runtime Verification (`server_runner.py`)
- Retains `--flash-attn on` paired strictly with `--cache-type-k q8_0` and `--cache-type-v q8_0` (supported Turing SM75 kernel).
- The readiness probe inspects server `/props` directly to confirm `flash_attn: true` and active context window (`n_ctx: 65536`).
- Safely avoids aggressive prompt-cache shifting and persistent slot swapping that desynchronize recurrent state.

### 3. Context Capacity Safeguard at 85% (`chat.py`)
- Proactive token estimation tracks conversational growth toward the 65,536 limit.
- At **85% capacity (55,705 tokens)**, prints an explicit diagnostic warning allowing compaction before hitting hard context limits and risking truncated system instructions or tool schemas.

---

## ⚡ Subtask 4: CPU/RAM Coordination, Latency Telemetry & Model-Aware Context Guard

### 1. Model-Aware Token Admission Guard (`context_admission.py` & `auth_gateway.py`)
- Replaces naive character length approximations with exact model-level token validation via `llama-server`'s `/v1/chat/completions/input_tokens`.
- Strictly enforces:
  $$\text{Actual Input Tokens} + \text{Reserved Output Tokens} \le 65,536$$
- Rejects oversized payloads with HTTP 400 (`CONTEXT_LENGTH_EXCEEDED`) and handles unreachable upstream counters with fail-closed HTTP 503 (`TOKEN_COUNT_UNAVAILABLE`).
- Fast-paths raw inputs exceeding physical limits ($>250,000$ characters) to reject instantly without wasting GPU compute time.

### 2. Latency Telemetry & True TTFT Clock (`latency_telemetry.py` & `chat.py`)
- `RequestClock` measures user-observed completion wall time and streams Time-to-First-Token (**TTFT**) based strictly on the arrival of the first *content/tool_call delta* (not initial role headers).
- Extracts native `prompt_per_second`, `predicted_per_second`, `prompt_ms`, and `predicted_ms` directly from `llama-server` chunks.
- Resilient formatting handles missing/None timings gracefully without exceptions.

### 3. CPU Thread Allocation & Hardware Pinning (`server_runner.py` & `runtime_policy.py`)
- Evaluator decoding threads: `--threads 4` (matching Kaggle's 4 vCPU allocation without thread contention).
- Prompt ingestion threads: `--threads-batch 4` (dedicated batch processing).
- Preserves single dedicated slot (`--parallel 1`), microbatch size (`--ubatch-size 512`), and batch size (`--batch-size 2048`).
- Keeps all 64 layers resident in Dual-T4 VRAM (`--n-gpu-layers 99`) with zero compute layers offloaded to CPU.

### 4. Native Prompt-Prefix KV Cache Reuse
- Verified in live multi-turn conversations: turn 2 reuses 35 of 53 tokens from prefix cache, minimizing re-tokenization overhead on follow-up turns.

### 5. Explicit User Thinking Toggle (`/think on` / `/think off`)
- Replaces hardcoded keyword heuristics with full user control in `chat.py`.
- **Default: Thinking ON** (preserves deep Chain-of-Thought reasoning for complex coding, debugging, and analysis).
- **Interactive Commands:** Type `/think off` for direct responses (~1.7s TTFT) or `/think on` to re-enable deep reasoning.

---

## 🛡️ Subtask 5: API Gateway, Streaming & Remote Connection Reliability

### 1. Atomic Single-Slot Concurrency Scheduler (`SingleSlotScheduler`)
- Concurrency protection for `--parallel 1` inference slots: exactly one active caller admitted to GPU inference at any instant.
- Thread-safe bounded wait queue of depth 1 (`max_queue=1`).
- **Immediate Overload Rejection:** When slot and queue are occupied, subsequent requests are immediately rejected with **HTTP 429 (`SLOT_OVERLOADED`)** and `Retry-After: 5` header, eliminating unbounded queue delays.
- **Acquisition Timeout:** Callers waiting in the queue exceeding `queue_timeout` (5.0s) fail safely with **HTTP 503 (`SLOT_ACQUISITION_TIMEOUT`)**.
- **Atomic Release Guarantee:** Wrapped in strict `finally:` blocks ensuring the slot permit is released immediately upon request completion or client socket drop.

### 2. Lossless SSE Transport & UTF-8 / Tool-Call Argument Streaming
- Replaces intermediate JSON parsing and reconstructing with transparent chunked byte forwarding over `Transfer-Encoding: chunked`.
- **UTF-8 Multi-byte Preservation:** Raw byte stream forwarding prevents truncation or corruptions when 3-byte or 4-byte UTF-8 sequences cross network chunk boundaries.
- **Incremental Tool-Call JSON Stream:** Function argument fragments (e.g., `{"code":`, `"print(1)"`, `"}"`) stream verbatim without modification, allowing clients to reassemble arbitrary JSON payloads before parsing.
- **Reasoning Deltas:** Unmodified forwarding of native `<think>` blocks and reasoning tokens directly to client telemetry.

### 3. Client-Disconnect Cancellation & Upstream Fault Recovery
- Actively detects client connection aborts (`BrokenPipeError`, `ConnectionResetError`) during streaming.
- Guarantees immediate reclamation of the single-slot permit so following callers are not blocked by orphaned socket handles.
- Upstream server disconnects or restarts are caught cleanly, returning **HTTP 503 (`SERVICE_UNAVAILABLE`)** with `Retry-After: 5` header instead of hanging.

### 4. Route Whitelisting, Body Size Caps & Correlation Tracing
- **Route Whitelisting:** Strictly limits incoming traffic to `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/chat/completions/input_tokens`, and authenticated `/v1/shutdown`.
- **Payload Limits:** Fast-paths request bodies larger than 15 MB with **HTTP 413 (`PAYLOAD_TOO_LARGE`)**.
- **Request Tracing:** Generates and propagates unique `X-Request-ID` across every client interaction and error response for distributed telemetry.

---

## 🧪 Testing & Verification

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```
**Test Coverage (34/34 Passing):**
- Native `llama-server` command flag generation (`--parallel 1`, `--no-reasoning-preserve`, `--flash-attn on`, `--ctx-size 65536`, `--threads 4`, `--threads-batch 4`).
- Hardware VRAM and system host RAM inspection fallbacks.
- Bearer token security boundary & 401 gatekeeper.
- Per-GPU memory allocation & mathematical layer distribution check.
- Manifest dynamic metadata recording & independent capability transitions.
- Chat template source classification (`EMBEDDED_GGUF_TEMPLATE`).
- Exact hybrid 16/48 attention memory calculation (2,176.0 MiB at 64K).
- Conservative DeltaNet recurrent state allowance (never zero).
- Active 85% context threshold warning arithmetic.
- Context admission boundary pass ($63,488$ tokens + $2,048$ output).
- Context overflow rejection ($63,489$ tokens rejected).
- Fail-closed token counter network and missing payload protection.
- Native timing extraction and true content-token TTFT measurement.
- Non-streaming TTFT handling (no false values fabricated).
- Thread oversubscription rejection policy.
- Command policy validation.
- Single-slot scheduler immediate acquire and release.
- Single-slot scheduler atomic queue-full rejection (HTTP 429).
- Gateway route whitelist validation and path traversal rejection.
- Gateway 15 MB body size limit enforcement (HTTP 413).
- Lossless SSE multi-byte UTF-8 preservation & tool-call argument fragment reassembly.
- Client disconnect (`BrokenPipeError`) slot release guarantee.
- Upstream server unreachable clean HTTP 503 recovery.
