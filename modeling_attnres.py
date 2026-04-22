"""
Qwen3 with Block Attention Residuals (AttnRes).

Replaces standard additive residual connections with softmax attention over
previous block representations, as described in:
  "Attention Residuals" (Kimi Team, arXiv:2603.15031)

For pretrained model conversion, we use a **recency bias** approach:
a large learnable bias on the last element (partial_block) in the depth-
attention logits makes softmax put ~100% weight on it at init.  This means
block_attn_res(...) ≈ partial_block at init → the model is mathematically
equivalent to standard Qwen3.  During training the bias and proj weights
co-adapt, letting the model learn cross-block attention.
"""

import torch
import torch.nn as nn

# Re-use Qwen3 components directly from the installed transformers package.
# We only override DecoderLayer and Model; everything else is unchanged.
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    Qwen3PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


# ---------------------------------------------------------------------------
# Config extension
# ---------------------------------------------------------------------------

class Qwen3AttnResConfig(Qwen3Config):
    """Qwen3Config extended with AttnRes hyper-parameters."""

    model_type = "qwen3_attnres"

    def __init__(self, attnres_num_blocks: int = 8,
                 attnres_recency_bias_init: float = 3.0,
                 attnres_mode: str = "block",
                 attnres_gate_type: str = "bias",
                 **kwargs):
        # Remove legacy keys if present (from old checkpoints)
        kwargs.pop("attnres_init_bias", None)
        kwargs.pop("attnres_gate_init", None)
        super().__init__(**kwargs)
        self.attnres_num_blocks = attnres_num_blocks
        self.attnres_recency_bias_init = attnres_recency_bias_init
        # "block" = Block AttnRes (grouped), "full" = Full AttnRes (per-sublayer)
        self.attnres_mode = attnres_mode
        # Gate type: "bias" (recency bias on softmax logit),
        #            "sigmoid_scalar" (scalar sigmoid gate between residual & attnres),
        #            "sigmoid_vector" (input-dependent per-dim sigmoid gate)
        self.attnres_gate_type = attnres_gate_type


# ---------------------------------------------------------------------------
# Core Block-AttnRes operation
# ---------------------------------------------------------------------------

def capacity_check(weights: torch.Tensor, d: int, threshold: float = 0.01
) -> tuple[bool, int, int]:
    """
    CHECK operation: verify that the effective number of contributing
    blocks does not exceed the holographic capacity boundary √d.

    The capacity boundary k ≤ √d is a sharp phase transition from
    holographic computing theory (Plate 1995, Moore 2026). Above √d
    items in superposition, retrieval fidelity degrades to chance level
    regardless of how the weights are computed.

    Args:
        weights: softmax attention weights [N+1, B, T]
        d: hidden dimension
        threshold: minimum weight to count as a contribution

    Returns:
        (within_capacity, effective_k, capacity_limit)
    """
    # Count blocks with weight above threshold, averaged over batch/tokens
    effective_k = (weights > threshold).float().sum(dim=0).mean().item()
    capacity_limit = int(d ** 0.5)
    return effective_k <= capacity_limit, int(effective_k), capacity_limit


def block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,)
    norm: Qwen3RMSNorm,           # RMSNorm applied to keys before scoring
    recency_bias: nn.Parameter,   # scalar bias added to partial_block's logit
    return_entropy: bool = False, # if True, also return mean entropy of softmax weights
    enforce_capacity: bool = False,  # if True, prune to √d effective contributions
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Attend over all block representations + the current partial block.

    Returns a [B, T, D] tensor — the attended aggregation of depth history.
    If return_entropy=True, also returns a scalar entropy value.

    When enforce_capacity=True, applies the holographic capacity boundary
    k ≤ √d: if more than √d blocks contribute meaningfully, the lowest-
    weight blocks are pruned and weights renormalized. This prevents
    representation degradation from unchecked accumulation.

    Reference: "Attention Residuals Need a Capacity Boundary" (Moore 2026)
    """
    D = partial_block.shape[-1]

    # Stack everything: shape [N+1, B, T, D]
    V = torch.stack(blocks + [partial_block], dim=0)

    # Keys = normalised values
    K = norm(V)

    # Scalar logit per (block, batch, token) via the single learned query
    # proj.weight shape: (1, D) → squeeze to (D,)
    query = proj.weight.view(-1)                              # (D,)
    logits = torch.einsum("d, n b t d -> n b t", query, K)   # (N+1, B, T)

    # Recency bias: boost the last element (partial_block)
    logits[-1] = logits[-1] + recency_bias

    # Softmax across block dimension
    weights = logits.softmax(dim=0)                           # (N+1, B, T)

    # ── CHECK: Holographic capacity boundary ──
    # If effective contributors exceed √d, prune to top-√d.
    # This is a hard constraint from the capacity boundary k ≤ √d:
    # above this threshold, superposed representations degrade to
    # chance-level fidelity regardless of weight quality.
    if enforce_capacity:
        within_cap, eff_k, cap_limit = capacity_check(weights, D)
        if not within_cap and cap_limit < weights.shape[0]:
            # Keep only top-k blocks by mean weight across batch/tokens
            mean_weights = weights.mean(dim=(1, 2))          # (N+1,)
            _, top_indices = mean_weights.topk(cap_limit)
            mask = torch.zeros(weights.shape[0], device=weights.device)
            mask[top_indices] = 1.0
            weights = weights * mask.view(-1, 1, 1)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

    # Weighted sum of values
    h = torch.einsum("n b t, n b t d -> b t d", weights, V)  # (B, T, D)

    if return_entropy:
        # Entropy: -sum(w * log(w)), averaged over batch and tokens
        # Higher entropy = more diverse attention (good)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=0).mean()
        return h, entropy

    return h


# ---------------------------------------------------------------------------
# Modified decoder layer
# ---------------------------------------------------------------------------

class Qwen3AttnResDecoderLayer(GradientCheckpointingLayer):
    """
    Qwen3 decoder layer with Block AttnRes via recency-biased depth attention.

    At init, a large recency bias makes block_attn_res return partial_block
    exactly, so the model is mathematically identical to standard Qwen3.
    During training, proj weights learn to attend to earlier blocks while
    the bias co-adapts.

    Forward:
        h = block_attn_res(blocks, partial_block)   # ≈ partial_block at init
        attn_out = self_attn(layernorm(h))
        partial_block = partial_block + attn_out     # standard residual add
    """

    def __init__(self, config: Qwen3AttnResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Standard Qwen3 attention
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # AttnRes components — one (proj, norm) per sublayer.
        self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.attn_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gate type determines how AttnRes output is mixed with residual stream
        self.gate_type = getattr(config, "attnres_gate_type", "bias")
        bias_init = getattr(config, "attnres_recency_bias_init", 10.0)

        if self.gate_type == "sigmoid_scalar":
            # Scalar sigmoid gate: sigmoid(-2) ≈ 0.12 → small initial mixing
            self.attn_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.mlp_res_gate_logit = nn.Parameter(torch.tensor(-2.0))
            self.attn_res_bias = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_bias = nn.Parameter(torch.tensor(0.0))
        elif self.gate_type == "sigmoid_vector":
            # Input-dependent vector gate: per-dim, per-token gating
            self.attn_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.attn_res_gate_proj.weight)
            nn.init.constant_(self.attn_res_gate_proj.bias, -2.0)
            self.mlp_res_gate_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.mlp_res_gate_proj.weight)
            nn.init.constant_(self.mlp_res_gate_proj.bias, -2.0)
            self.attn_res_bias = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_bias = nn.Parameter(torch.tensor(0.0))
        elif self.gate_type == "learnable_alpha":
            # Simple learnable scalar: h = (1-α)*partial + α*attnres, init α=0
            self.attn_res_alpha = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_alpha = nn.Parameter(torch.tensor(0.0))
            self.attn_res_bias = nn.Parameter(torch.tensor(0.0))
            self.mlp_res_bias = nn.Parameter(torch.tensor(0.0))
        else:
            # Default "bias": no gate, AttnRes output used directly
            self.attn_res_bias = nn.Parameter(torch.tensor(bias_init))
            self.mlp_res_bias = nn.Parameter(torch.tensor(bias_init))

        # AttnRes mode: "block" or "full"
        self.attnres_mode = getattr(config, "attnres_mode", "block")

        # Block boundary: how many transformer layers per block (used in block mode)
        num_layers = config.num_hidden_layers
        num_blocks = getattr(config, "attnres_num_blocks", 8)
        self.layers_per_block = max(1, (num_layers + num_blocks - 1) // num_blocks)

    @property
    def is_block_boundary(self) -> bool:
        """True when this layer is the last in its block (0-indexed)."""
        return (self.layer_idx + 1) % self.layers_per_block == 0

    def _apply_gate(self, hidden_states, h_attn, sublayer: str):
        """Apply gating between residual stream and AttnRes output."""
        if self.gate_type == "sigmoid_scalar":
            logit = self.attn_res_gate_logit if sublayer == "attn" else self.mlp_res_gate_logit
            gate = torch.sigmoid(logit)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "sigmoid_vector":
            gate_proj = self.attn_res_gate_proj if sublayer == "attn" else self.mlp_res_gate_proj
            gate = torch.sigmoid(gate_proj(hidden_states))  # (B, T, D)
            return (1 - gate) * hidden_states + gate * h_attn
        elif self.gate_type == "learnable_alpha":
            alpha = self.attn_res_alpha if sublayer == "attn" else self.mlp_res_alpha
            return (1 - alpha) * hidden_states + alpha * h_attn
        else:
            # "bias" mode: no gate, AttnRes output used directly
            return h_attn

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        entropy_accum = kwargs.pop("entropy_accum", None)

        # ---- Attention sublayer ----
        if entropy_accum is not None:
            h_attn, ent = block_attn_res(blocks, partial_block,
                                         self.attn_res_proj, self.attn_res_norm,
                                         self.attn_res_bias, return_entropy=True)
            entropy_accum.append(ent)
        else:
            h_attn = block_attn_res(blocks, partial_block,
                                    self.attn_res_proj, self.attn_res_norm, self.attn_res_bias)
        h = self._apply_gate(partial_block, h_attn, "attn")

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(h),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        partial_block = partial_block + attn_out

        # Full mode: record post-attention state in history
        if self.attnres_mode == "full":
            blocks = blocks + [partial_block]

        # ---- MLP sublayer ----
        if entropy_accum is not None:
            h_attn, ent = block_attn_res(blocks, partial_block,
                                         self.mlp_res_proj, self.mlp_res_norm,
                                         self.mlp_res_bias, return_entropy=True)
            entropy_accum.append(ent)
        else:
            h_attn = block_attn_res(blocks, partial_block,
                                    self.mlp_res_proj, self.mlp_res_norm, self.mlp_res_bias)
        h = self._apply_gate(partial_block, h_attn, "mlp")

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        partial_block = partial_block + mlp_out

        # Record post-MLP state:
        # Full mode: always append (every sublayer output)
        # Block mode: only at block boundaries
        if self.attnres_mode == "full" or self.is_block_boundary:
            blocks = blocks + [partial_block]

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class Qwen3AttnResModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Block AttnRes via recency-biased depth attention."""

    config_class = Qwen3AttnResConfig

    def _init_weights(self, module):
        """Override to preserve AttnRes initialization."""
        super()._init_weights(module)
        if isinstance(module, Qwen3AttnResDecoderLayer):
            gate_type = getattr(self.config, "attnres_gate_type", "bias")
            if gate_type == "sigmoid_scalar":
                module.attn_res_gate_logit.data.fill_(-2.0)
                module.mlp_res_gate_logit.data.fill_(-2.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            elif gate_type == "sigmoid_vector":
                nn.init.zeros_(module.attn_res_gate_proj.weight)
                nn.init.constant_(module.attn_res_gate_proj.bias, -2.0)
                nn.init.zeros_(module.mlp_res_gate_proj.weight)
                nn.init.constant_(module.mlp_res_gate_proj.bias, -2.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            elif gate_type == "learnable_alpha":
                module.attn_res_alpha.data.fill_(0.0)
                module.mlp_res_alpha.data.fill_(0.0)
                module.attn_res_bias.data.fill_(0.0)
                module.mlp_res_bias.data.fill_(0.0)
            else:
                bias_init = getattr(self.config, "attnres_recency_bias_init", 10.0)
                module.attn_res_bias.data.fill_(bias_init)
                module.mlp_res_bias.data.fill_(bias_init)

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3AttnResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Block AttnRes state: list of completed block tensors + current partial
        # The token embedding acts as the first "block" (block 0).
        blocks: list[torch.Tensor] = [inputs_embeds]
        partial_block: torch.Tensor = inputs_embeds

        # Entropy accumulation for auxiliary loss
        entropy_lambda = kwargs.pop("entropy_lambda", 0.0)
        entropy_accum = [] if entropy_lambda > 0 else None

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                blocks, partial_block = self._gradient_checkpointing_func(
                    layer.__call__,
                    blocks,
                    partial_block,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                blocks, partial_block = layer(
                    blocks=blocks,
                    partial_block=partial_block,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    entropy_accum=entropy_accum,
                )

        hidden_states = self.norm(partial_block)

        # Compute mean entropy across all sublayers
        attnres_entropy = None
        if entropy_accum:
            attnres_entropy = torch.stack(entropy_accum).mean()

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        # Attach entropy as extra attribute
        out.attnres_entropy = attnres_entropy
        return out


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class Qwen3AttnResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Block AttnRes residuals."""

    config_class = Qwen3AttnResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.model = Qwen3AttnResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            # Entropy bonus: encourage diverse cross-layer attention
            entropy_lambda = kwargs.get("entropy_lambda", 0.0)
            attnres_entropy = getattr(outputs, "attnres_entropy", None)
            if entropy_lambda > 0 and attnres_entropy is not None:
                # Subtract entropy (negative because we want to maximize it)
                loss = loss - entropy_lambda * attnres_entropy

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
