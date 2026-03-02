"""
Model-specific adapters for manual block-by-block forward passes.

Each adapter handles model structure introspection and hidden state extraction
without KV cache for Phase 1 correctness validation.

IMPORTANT: This implementation is designed for MLX-LM models which often require
mask, cache, and position inputs to transformer blocks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.llama import create_attention_mask
except Exception:
    create_attention_mask = None

import hidden_state_collector
import attention_contribution


class ModelAdapter(ABC):
    """
    Abstract base class for model-specific adapters.
    
    Adapters implement manual block-by-block forward passes to capture
    hidden states at each transformer layer during generation.
    """
    
    def __init__(
        self,
        model: Any,
        collector: hidden_state_collector.HiddenStateCollector,
        acr_collector: Optional[attention_contribution.ACRCollector] = None,
    ):
        """
        Initialize adapter with model and collector.
        
        Args:
            model: MLX model instance
            collector: Shared HiddenStateCollector instance
        """
        self.model = model
        self.collector = collector
        self.layers: List = []
        self.embed_tokens: Any = None
        self.norm: Any = None
        self.lm_head: Any = None
        self.acr_collector: Optional[attention_contribution.ACRCollector] = acr_collector
        self._acr_layer_cache: Dict[int, Optional[dict]] = {}
        self._prev_acr_input: Optional[mx.array] = None
        
        # Locate model components during initialization
        self._locate_components()
        self._validate_components()
        if self.acr_collector is not None:
            self.acr_collector.configure_for_layers(len(self.layers))
    
    @abstractmethod
    def _locate_components(self) -> None:
        """
        Introspect model structure to find components.
        
        Must set: self.layers, self.embed_tokens, self.norm, self.lm_head
        
        Raises:
            ValueError: If model structure is unknown or components not found
        """
        pass
    
    def _validate_components(self) -> None:
        """
        Validate that all required components were located.
        
        Raises:
            ValueError: If any component is missing
        """
        if self.embed_tokens is None:
            raise ValueError("embed_tokens not found in model")
        if self.norm is None:
            raise ValueError("norm layer not found in model")
        # Note: lm_head may not exist for models using weight tying
        if self.layers is None or len(self.layers) == 0:
            raise ValueError("No transformer layers found in model")
    
    def _extract_last_token_hidden(self, x: mx.array) -> mx.array:
        """
        Normalize shape to extract last-token vector.
        
        Handles various hidden state shapes:
        - [dim] -> return as-is
        - [seq, dim] -> return x[-1, :]
        - [batch, seq, dim] -> return x[0, -1, :]
        
        Args:
            x: Hidden state MLX array
            
        Returns:
            MLX array of shape [dim]
        """
        if x.ndim == 1:
            # Already [dim]
            return x
        elif x.ndim == 2:
            # [seq, dim] -> extract last token
            return x[-1, :]
        elif x.ndim == 3:
            # [batch, seq, dim] -> extract first batch, last token
            return x[0, -1, :]
        else:
            raise ValueError(f"Unexpected hidden state shape: {x.shape}")

    def _make_causal_mask(self, seq_len: int, dtype: mx.Dtype = mx.float16) -> Optional[mx.array]:
        """
        Create causal attention mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length
            dtype: Data type for mask (default float16 for Qwen/Phi-3 compatibility)
            
        Returns:
            Causal mask of shape [seq_len, seq_len] or None if not needed
        """
        # Create causal mask: upper triangular matrix of -inf
        # mask[i, j] = 0 if i >= j else -inf (can only attend to past)
        # Use float16 by default for Qwen/Phi-3 compatibility
        mask = mx.full((seq_len, seq_len), float('-inf'), dtype=dtype)
        mask = mx.triu(mask, k=1)  # Upper triangle above diagonal
        return mask

    def _make_attention_mask(self, x: mx.array, cache: Optional[Any]) -> Optional[mx.array]:
        """
        Create attention mask using MLX-LM helper when available.

        Falls back to a simple causal mask if the helper is unavailable.
        """
        if create_attention_mask is not None:
            try:
                return create_attention_mask(x, cache)
            except Exception:
                pass
        seq_len = x.shape[1] if x.ndim >= 2 else int(x.shape[0])
        return self._make_causal_mask(seq_len, dtype=x.dtype if hasattr(x, "dtype") else mx.float16)

    def _l2_norm(self, v: mx.array, epsilon: float = 1e-8) -> float:
        """
        Compute L2 norm of a vector and return as float.
        """
        return float(mx.sqrt(mx.sum(v * v) + epsilon).item())

    def _layer_supports_acr(self, layer: Any) -> bool:
        """
        Check whether a layer exposes the components needed for ACR.
        """
        return all(
            hasattr(layer, name)
            for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
        )

    def _forward_layer_with_optional_acr(
        self,
        layer_idx: int,
        layer: Any,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Forward a transformer layer and optionally record ACR.

        Falls back to robust layer call when ACR is disabled or unsupported.
        """
        if self.acr_collector is None:
            return self._call_layer_robust(layer, x, mask=mask, cache=cache)

        if not self._layer_supports_acr(layer):
            self.acr_collector.note_missing_layer(layer_idx, "missing_attn_components")
            return self._call_layer_robust(layer, x, mask=mask, cache=cache)

        # Residual stream entering the layer
        input_last = self._extract_last_token_hidden(x)
        if self.acr_collector.residual_mode == "delta" and self._prev_acr_input is not None:
            residual_vec = input_last - self._prev_acr_input
        else:
            residual_vec = input_last

        # Attention output before residual addition
        attn_out = layer.self_attn(layer.input_layernorm(x), mask, cache)
        attn_last = self._extract_last_token_hidden(attn_out)

        attn_norm = self._l2_norm(attn_last)
        residual_norm = self._l2_norm(residual_vec)
        self.acr_collector.record(layer_idx, attn_norm, residual_norm)
        self._prev_acr_input = input_last

        # MLP + residual (mirror layer forward)
        h = x + attn_out
        r = layer.mlp(layer.post_attention_layernorm(h))
        out = h + r
        return out
    
    def _call_layer_robust(
        self,
        layer: Any,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Call transformer layer with robust signature handling.
        
        MLX-LM layers often have different signatures:
        - layer(x, mask, cache)
        - layer(x, mask)
        - layer(x)
        
        And may return:
        - x (hidden state only)
        - (x, cache) (hidden state + new cache)
        
        Args:
            layer: Transformer layer module
            x: Input hidden states
            mask: Optional attention mask
            
        Returns:
            Output hidden states (unwrapped from tuple if necessary)
        """
        # Try different signatures in order of likelihood
        try:
            # Most common: layer(x, mask=mask, cache=cache)
            output = layer(x, mask=mask, cache=cache)
        except TypeError:
            try:
                # Fallback: layer(x, mask=mask)
                output = layer(x, mask=mask)
            except TypeError:
                try:
                    # Fallback: layer(x)
                    output = layer(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to call layer with any signature: {e}")
        
        # Unwrap tuple if layer returns (hidden_states, cache)
        if isinstance(output, tuple):
            return output[0]
        
        return output
    
    @abstractmethod
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Full prefix forward pass with hidden state collection.
        
        Performs block-by-block traversal, calling collector.record() after
        each transformer block. Without KV cache, recomputes full prefix.
        
        IMPORTANT: This method calls collector.start() to reset state for
        the current token generation step.
        
        Args:
            input_ids: MLX array of token IDs, shape [seq_len] or [1, seq_len]
            
        Returns:
            Logits for next token, MLX array of shape [vocab_size]
        """
        pass
    
    def next_token_logits(self, input_ids: mx.array) -> mx.array:
        """
        Wrapper to get next token logits.
        
        Args:
            input_ids: MLX array of token IDs
            
        Returns:
            Logits for next token, MLX array of shape [vocab_size]
        """
        return self.forward_prefix_with_collection(input_ids)


class LlamaAdapter(ModelAdapter):
    """
    Adapter for Llama 3.2 3B Instruct model.
    
    Model structure:
    - model.model.embed_tokens: Embedding layer
    - model.model.layers: List of transformer blocks
    - model.model.norm: Final RMS normalization
    - model.lm_head: Output projection to vocabulary
    """
    
    def _locate_components(self) -> None:
        """
        Locate Llama model components.
        
        Tries multiple attribute patterns to be robust.
        """
        # Try standard Llama structure first
        if hasattr(self.model, 'model'):
            self.embed_tokens = getattr(self.model.model, 'embed_tokens', None)
            self.layers = getattr(self.model.model, 'layers', None)
            self.norm = getattr(self.model.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            # Try flat structure
            self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Llama-specific forward pass with hidden state collection.
        
        Matches MLX-LM's LlamaModel.__call__ exactly for logit parity.
        
        Args:
            input_ids: Token IDs, shape [seq_len] or [1, seq_len]
            
        Returns:
            Logits for next token, shape [vocab_size]
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None
        
        # Normalize input shape to [batch=1, seq_len]
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]  # [seq_len] -> [1, seq_len]
        
        # Embed tokens: [1, seq_len, hidden_dim]
        x = self.embed_tokens(input_ids)
        
        # Create cache and mask (prefer MLX-LM helper; fallback to causal mask)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])
        
        # Pass through each transformer block
        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            
            # Extract and record last-token hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token: [1, seq_len, hidden_dim] -> [hidden_dim]
        last_token = self._extract_last_token_hidden(x)
        
        # Project to vocabulary using weight tying (as_linear)
        # MLX-LM Llama models use weight tying: embed_tokens.as_linear()
        logits = self.embed_tokens.as_linear(last_token)
        
        # Normalize logits shape to [vocab_size]
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        
        return logits


class QwenAdapter(ModelAdapter):
    """
    Adapter for Qwen 2.5 7B Instruct model.
    
    Qwen models may use different attribute names than Llama.
    """
    
    def _locate_components(self) -> None:
        """
        Locate Qwen model components with fallbacks.
        """
        # Try multiple attribute name patterns
        if hasattr(self.model, 'model'):
            # Try Qwen-specific names
            self.embed_tokens = (
                getattr(self.model.model, 'embed_tokens', None) or
                getattr(self.model.model, 'tok_embeddings', None) or
                getattr(self.model.model, 'wte', None)
            )
            self.layers = getattr(self.model.model, 'layers', None)
            self.norm = getattr(self.model.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        elif hasattr(self.model, 'transformer'):
            # Alternative: transformer attribute
            self.embed_tokens = (
                getattr(self.model.transformer, 'embed_tokens', None) or
                getattr(self.model.transformer, 'wte', None)
            )
            self.layers = getattr(self.model.transformer, 'layers', None) or getattr(self.model.transformer, 'h', None)
            self.norm = getattr(self.model.transformer, 'norm', None) or getattr(self.model.transformer, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            # Flat structure
            self.embed_tokens = getattr(self.model, 'embed_tokens', None) or getattr(self.model, 'tok_embeddings', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Qwen-specific forward pass with hidden state collection.
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None
        
        # Normalize input shape
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        
        seq_len = input_ids.shape[1]
        mask = self._make_causal_mask(seq_len)
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Pass through transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask)
            
            # Record hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token and project
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        
        # Normalize shape
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        
        return logits


class Phi3Adapter(ModelAdapter):
    """
    Adapter for Phi-3 Mini Instruct model.
    
    Phi-3 models may use different attribute names.
    """
    
    def _locate_components(self) -> None:
        """
        Locate Phi-3 model components with fallbacks.
        """
        # Try multiple attribute name patterns
        if hasattr(self.model, 'model'):
            self.embed_tokens = (
                getattr(self.model.model, 'embed_tokens', None) or
                getattr(self.model.model, 'wte', None)
            )
            self.layers = getattr(self.model.model, 'layers', None) or getattr(self.model.model, 'h', None)
            self.norm = getattr(self.model.model, 'norm', None) or getattr(self.model.model, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        elif hasattr(self.model, 'transformer'):
            self.embed_tokens = getattr(self.model.transformer, 'embed_tokens', None) or getattr(self.model.transformer, 'wte', None)
            self.layers = getattr(self.model.transformer, 'layers', None) or getattr(self.model.transformer, 'h', None)
            self.norm = getattr(self.model.transformer, 'norm', None) or getattr(self.model.transformer, 'ln_f', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
        else:
            self.embed_tokens = getattr(self.model, 'embed_tokens', None)
            self.layers = getattr(self.model, 'layers', None)
            self.norm = getattr(self.model, 'norm', None)
            self.lm_head = getattr(self.model, 'lm_head', None)
    
    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        """
        Phi-3-specific forward pass with hidden state collection.
        """
        # Reset collector for this token step
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None
        
        # Normalize input shape
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        
        seq_len = input_ids.shape[1]
        mask = self._make_causal_mask(seq_len)
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Pass through transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask)
            
            # Record hidden state
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract last token and project
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        
        # Normalize shape
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        
        return logits


class MistralAdapter(ModelAdapter):
    """
    Adapter for Mistral-style models.

    Mistral MLX-LM models generally follow Llama-like structure.
    """

    def _locate_components(self) -> None:
        if hasattr(self.model, "model"):
            self.embed_tokens = (
                getattr(self.model.model, "embed_tokens", None)
                or getattr(self.model.model, "tok_embeddings", None)
                or getattr(self.model.model, "wte", None)
            )
            self.layers = getattr(self.model.model, "layers", None) or getattr(self.model.model, "h", None)
            self.norm = getattr(self.model.model, "norm", None) or getattr(self.model.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        else:
            self.embed_tokens = (
                getattr(self.model, "embed_tokens", None)
                or getattr(self.model, "tok_embeddings", None)
                or getattr(self.model, "wte", None)
            )
            self.layers = getattr(self.model, "layers", None) or getattr(self.model, "h", None)
            self.norm = getattr(self.model, "norm", None) or getattr(self.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        x = self.embed_tokens(input_ids)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)

        x = self.norm(x)
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class SmolLMAdapter(ModelAdapter):
    """
    Adapter for SmolLM models (Llama-like in MLX).
    """

    def _locate_components(self) -> None:
        if hasattr(self.model, "model"):
            self.embed_tokens = (
                getattr(self.model.model, "embed_tokens", None)
                or getattr(self.model.model, "tok_embeddings", None)
                or getattr(self.model.model, "wte", None)
            )
            self.layers = getattr(self.model.model, "layers", None) or getattr(self.model.model, "h", None)
            self.norm = getattr(self.model.model, "norm", None) or getattr(self.model.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        elif hasattr(self.model, "transformer"):
            self.embed_tokens = (
                getattr(self.model.transformer, "embed_tokens", None)
                or getattr(self.model.transformer, "tok_embeddings", None)
                or getattr(self.model.transformer, "wte", None)
            )
            self.layers = getattr(self.model.transformer, "layers", None) or getattr(self.model.transformer, "h", None)
            self.norm = getattr(self.model.transformer, "norm", None) or getattr(self.model.transformer, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)
        else:
            self.embed_tokens = (
                getattr(self.model, "embed_tokens", None)
                or getattr(self.model, "tok_embeddings", None)
                or getattr(self.model, "wte", None)
            )
            self.layers = getattr(self.model, "layers", None) or getattr(self.model, "h", None)
            self.norm = getattr(self.model, "norm", None) or getattr(self.model, "ln_f", None)
            self.lm_head = getattr(self.model, "lm_head", None)

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        self.collector.start()
        if self.acr_collector is not None:
            self.acr_collector.start()
            self._prev_acr_input = None

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        x = self.embed_tokens(input_ids)
        cache = [None] * len(self.layers)
        mask = self._make_attention_mask(x, cache[0])

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            x = self._forward_layer_with_optional_acr(layer_idx, layer, x, mask=mask, cache=c)
            last_token_hidden = self._extract_last_token_hidden(x)
            self.collector.record(layer_idx, last_token_hidden)

        x = self.norm(x)
        last_token = self._extract_last_token_hidden(x)
        logits = self.lm_head(last_token)
        if logits.ndim == 2:
            logits = logits.squeeze(0)
        return logits


class LLaVAMiniAdapter(ModelAdapter):
    """
    Adapter stub for LLaVA-mini style multimodal models.

    This codebase is text-only (mlx-lm). LLaVA models are multimodal and
    typically require mlx-vlm. We expose this adapter to make the intended
    model type explicit, but it will raise with guidance at runtime.
    """

    def _locate_components(self) -> None:
        raise RuntimeError(
            "LLaVA-mini is a multimodal model. Use mlx-vlm for loading and "
            "inference; this text-only adapter is not implemented."
        )

    def forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array:
        raise RuntimeError(
            "LLaVA-mini is a multimodal model. Use mlx-vlm for loading and "
            "inference; this text-only adapter is not implemented."
        )


def create_adapter(
    model: Any,
    collector: hidden_state_collector.HiddenStateCollector,
    model_type: str = "llama",
    acr_collector: Optional[attention_contribution.ACRCollector] = None,
) -> ModelAdapter:
    """
    Factory function to create appropriate adapter for model.
    
    Args:
        model: MLX model instance
        collector: HiddenStateCollector instance
        model_type: One of "llama", "qwen", "phi3"
        
    Returns:
        Appropriate ModelAdapter subclass instance
        
    Raises:
        ValueError: If model_type is unknown
    """
    adapters = {
        "llama": LlamaAdapter,
        "qwen": QwenAdapter,
        "phi3": Phi3Adapter,
        "mistral": MistralAdapter,
        "smollm": SmolLMAdapter,
        "llava": LLaVAMiniAdapter
    }
    
    if model_type.lower() not in adapters:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {list(adapters.keys())}"
        )
    
    adapter_class = adapters[model_type.lower()]
    return adapter_class(model, collector, acr_collector=acr_collector)
