"""Model Configuration Package."""
from kaggle.config.model_config.catalog import (
    ModelCatalog,
    get_ornith_35b,
    get_qwen36_35b,
    get_qwen38_27b,
)
from kaggle.config.model_config.profile import ModelProfile

__all__ = [
    "ModelProfile",
    "ModelCatalog",
    "get_ornith_35b",
    "get_qwen36_35b",
    "get_qwen38_27b",
]
