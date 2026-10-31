"""Qwen2 decoder layer with the residual add before the MLP fused into the post-attention RMSNorm.

transformers adds the attention output to the residual stream in one fp32 pass and normalizes the
sum in another; add_rmsnorm (rmsnorm.py) does both in one launch and returns the new residual as
well. Everything else is the original layer: the same modules, called in the same order with the
same arguments.
"""

from .rmsnorm import add_rmsnorm, supported


def make_decoder_layer_forward(original, norm_cls):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        norm = self.post_attention_layernorm
        if not isinstance(norm, norm_cls) or not supported(hidden_states) or \
                norm.weight.shape != hidden_states.shape[-1:]:
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        residual, hidden_states = add_rmsnorm(residual, hidden_states, norm.weight,
                                              norm.variance_epsilon)
        return residual + self.mlp(hidden_states)

    return forward


def patch_qwen2_decoder_layer():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    cls.forward = make_decoder_layer_forward(cls.forward, modeling_qwen2.Qwen2RMSNorm)
