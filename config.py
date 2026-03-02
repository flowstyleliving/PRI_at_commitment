"""
Configuration for the PRI-at-commitment synthetic contradiction experiment.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class UncertaintyConfig:
    # PRI parameters
    pri_alpha: float = 0.1
    selected_prob_clamp: Tuple[float, float] = (1e-10, 1.0)
    cosine_epsilon: float = 1e-8

    # Delta-sigma / SVD numerical stability
    delta_sigma_jsd_epsilon: float = 1e-8
    svd_epsilon: float = 1e-8


DEFAULT_UNCERTAINTY_CONFIG = UncertaintyConfig()

# MLX model configurations used in this experiment.
MODEL_CONFIGS = [
    {
        "name": "llama_3.2_3b",
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "display_name": "Llama 3.2 3B Instruct",
        "model_type": "llama",
    },
    {
        "name": "mistral_7b",
        "path": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "display_name": "Mistral 7B Instruct",
        "model_type": "mistral",
    },
    {
        "name": "qwen_2.5_7b",
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "display_name": "Qwen 2.5 7B Instruct",
        "model_type": "qwen",
    },
]
