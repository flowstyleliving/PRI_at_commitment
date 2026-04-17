"""
Configuration for the PRI-at-commitment synthetic contradiction experiment.
"""

from dataclasses import dataclass, field
from typing import Tuple, Iterable


@dataclass
class UncertaintyConfig:
    # PRI parameters
    pri_alpha: float = 0.1
    selected_prob_clamp: Tuple[float, float] = (1e-10, 1.0)
    cosine_epsilon: float = 1e-8

    # Delta-sigma / SVD numerical stability
    delta_sigma_jsd_epsilon: float = 1e-8
    svd_epsilon: float = 1e-8

    # v3 null-space direction signal — ranks at which to compute null_ratio
    # and Fisher energy ε(r) = Σ_{i≤r} σ_i² / Σ_i σ_i².
    v3_rank_values: Iterable[int] = field(default_factory=lambda: (1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64))


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
