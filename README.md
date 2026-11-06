# Personal Assistant

A modular, self-hosted AI Assistant architecture powered by remote GPU inference on Kaggle and custom local persona workflows.

---

## 📌 Current Project Status

The workspace has been initialized freshly with a clean layout and Git version control:

- **Root Environment**: Clean `.env` holding multi-account Kaggle credentials and tunnel configurations (protected via `.gitignore`).
- **Git Repository**: Initialized with secrets excluded from version control.
- **Kaggle Subsystem (`kaggle/`)**: Dedicated backend package folder structure established for remote LLM hosting.

---

## 🗂️ Project Directory Structure

```text
Assistant/
├── .env                       # Local secrets & API configurations (git-ignored)
├── .gitignore                 # Version control exclusion rules
├── README.md                  # Project overview & progress tracking
├── logs/                      # Runtime operational logs
├── assistant_persona/         # Persona profiles, system prompts & chat bindings
└── kaggle/                    # Dedicated Kaggle remote GPU backend package
    ├── __init__.py
    ├── config/
    │   ├── auth_config/       # Multi-account credentials & switcher logic
    │   ├── model_config/      # Model catalogs, quantization & VRAM profiles
    │   └── api_end_point_config/ # llama.cpp runtime flags & tunnel configuration
    ├── deploy_notebook/       # Kernel deployment templates & deployer CLI
    └── tracker/               # Kernel status monitor, log streamer & GPU stopper
```

---

## 🚀 What's Configured So Far

1. **Multi-Account Kaggle Authentication**
   - Configured in `.env` with primary and backup accounts.
   - Live connectivity verified via Kaggle CLI (`kaggle kernels list --mine`).

2. **Remote Host & Tunneling Blueprint**
   - Supports Cloudflare and Ngrok tunneling to expose remote `llama.cpp` server instances running on Kaggle GPUs (T4 / P100).

3. **Workspace Isolation**
   - The Kaggle subsystem is organized as a stand-alone Python package (`kaggle/`), ensuring clean separation from the local assistant's persona and logic.

---

## 📋 Next Steps

- [ ] **Step 1**: Migrate and optimize the authentication module (`kaggle/config/auth_config/`).
- [ ] **Step 2**: Implement the model catalog and VRAM hardware profiler (`kaggle/config/model_config/`).
- [ ] **Step 3**: Set up kernel deployment scripts and log streamer/GPU stopper (`kaggle/deploy_notebook/`, `kaggle/tracker/`).
- [ ] **Step 4**: Design the Assistant Persona module (`assistant_persona/`) to bind client interactions to the remote endpoint.
