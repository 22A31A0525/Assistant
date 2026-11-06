# Personal Assistant

A modular, self-hosted AI Assistant architecture powered by remote GPU inference on Kaggle and custom local persona workflows.

---

## 📌 Project Architecture

The repository is structured around a root configuration and a clean, isolated `kaggle/` backend subsystem:

```text
Assistant/
├── .env                       # Local secrets & API tokens (git-ignored)
├── .gitignore                 # Exclusion rules for secrets, caches, and logs
├── README.md                  # Project overview & architectural guide
├── logs/                      # Runtime operational logs
├── assistant_persona/         # Persona profiles, system prompts & chat bindings
└── kaggle/                    # Dedicated Kaggle remote GPU backend package
    ├── __init__.py
    ├── cli.py                 # Unified Kaggle Hub CLI (python -m kaggle.cli)
    └── config/
        ├── auth_config/       # Multi-account credentials & active switcher
        │   ├── __init__.py
        │   ├── manager.py     # Account loader, synchronizer & validator
        │   └── models.py      # Account data models & slug generator
        ├── model_config/      # Curated 3-model catalog & hardware profiles
        │   ├── __init__.py
        │   ├── catalog.py     # Registry with fixed owner datasets & vision mmproj
        │   ├── common.py      # Shared CUDA 12 & runtime wheel datasets
        │   └── profile.py     # ModelProfile dataclass (text & multimodal)
        └── api_end_point_config/ # llama.cpp runtime flags & tunnel configuration
```

---

## 🚀 Features Implemented

### 1. Unified Kaggle Control Center
A single, centralized CLI to manage all remote GPU operations:
```bash
python -m kaggle.cli
```

### 2. Multi-Account Authentication Manager (`kaggle/config/auth_config/`)
- Dynamically parses active and registered accounts from `.env`.
- Safely synchronizes credentials to `~/.kaggle/kaggle.json` and `.env`.
- Cleans conflicting OAuth tokens to prevent CLI account lockups.
- Direct connectivity verification with Kaggle API.

### 3. Curated Model Catalog (`kaggle/config/model_config/`)
Configured specifically for Dual Tesla T4 GPUs (`LAYER` split mode) with exact Kaggle source mappings:

| Model | Capability | Primary Source / Dataset | Vision Projector (`mmproj`) |
|---|---|---|---|
| **Ornith-1.0-35B** | Pure Text / Reasoning | `rohitvijayadapa/ornith-1-0-35b` | None |
| **Qwen-3.6-35B** | Multimodal (Vision) | `saitejavenkatavinay/qwen36-35b-a3b-q4km` | `saitejavenkatavinay/qwen36-35b-mmproj` |
| **Qwen-3.8-27B** | Multimodal (Vision) | `saitejavenkatavinay/ornith-qwen38-27b` | `saitejavenkatavinay/qwen36-35b-mmproj` |

All models automatically link shared runtime dependencies (`rohitvijayadapa/ornith-runtime-wheels` & `rohitvijayadapa/ornith-runtime-cuda12-sm75`).
