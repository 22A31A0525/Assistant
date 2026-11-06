# Kaggle Subsystem Package

Provides remote GPU hosting and lifecycle management for personal assistant LLMs:

- **Unified CLI Entry**:
  ```bash
  python -m kaggle.cli
  ```
- **Modules**:
  - `config/auth_config/`: Multi-account credentials and active user switcher.
  - `config/model_config/`: Curated models (Ornith 35B text, Qwen 3.6 vision, Qwen 3.8 vision).
  - `config/api_end_point_config/`: Tunneling & Llama server settings.
  - `deploy_notebook/`: Kernel generation and deployment.
  - `tracker/`: Kernel status, tunnel log discovery, and quota stopper.
