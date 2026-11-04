# Copyright 2026 Apple Inc.
# Licensed under the repository BSD-3-Clause license.
"""Experimental fixed-request-batch Gemma4 with functional KV I/O.

The caller owns independent global/sliding cache positions and writes returned
new KV only after inference completes. The model contains no dynamic KV writes.
This is deliberately a separate export contract from the shipping static engine.
"""
import torch
from torch import nn
from coreai_models.models.ios.gemma4_text import dequantize_per_tensor


class ReadonlyRowCache:
    """Preserve shared-layer semantics while returning only newly generated KV."""
    def register_kv_cache(self, key_cache, value_cache):
        self.past_k = key_cache
        self.past_v = value_cache
        self.new_k = [None] * key_cache.shape[0]
        self.new_v = [None] * value_cache.shape[0]
        self._keys = [key_cache[i] for i in range(key_cache.shape[0])]
        self._values = [value_cache[i] for i in range(value_cache.shape[0])]

    def update_and_fetch(self, layer_idx, offset, k, v, num_token_updates):
        self.new_k[layer_idx] = k
        self.new_v[layer_idx] = v
        self._keys[layer_idx] = torch.cat((self.past_k[layer_idx], k), dim=-1)
        self._values[layer_idx] = torch.cat((self.past_v[layer_idx], v), dim=-1)
        return self._keys[layer_idx], self._values[layer_idx]

    @property
    def k_cache(self):
        return self._keys

    @property
    def v_cache(self):
        return self._values

    def updates(self):
        return torch.stack(self.new_k), torch.stack(self.new_v)


class BatchedReadonlyGemma4(nn.Module):
    """Own an independently loaded Gemma4Extend; do not wrap a live engine's model.

    Masks cover [past capacity + query width]. Past slots contain only committed
    tokens, while appended query positions use a triangular mask. Sliding masks
    must exclude past tokens outside each query's own window, including within a
    multi-token query. RoPE is supplied per request at its absolute positions.
    """
    def __init__(self, extend):
        super().__init__()
        self.extend = extend
        self.emit_logits = True
        self.extend.prefill_mode = False
        self.extend.emit_hidden_states = True
        self.extend.global_cache = ReadonlyRowCache()
        self.extend.sliding_cache = ReadonlyRowCache()

    def forward(self, transformer_input, rope_cos, rope_sin, causal_mask,
                sliding_causal_mask, key_cache, value_cache, sliding_key_cache,
                sliding_value_cache, embedding_table, ple_embeddings):
        unused_offset = torch.zeros(1, dtype=torch.int32, device=transformer_input.device)
        _, hidden = self.extend(
            transformer_input, rope_cos, rope_sin, unused_offset, unused_offset,
            causal_mask, sliding_causal_mask, key_cache, value_cache,
            sliding_key_cache, sliding_value_cache, embedding_table, ple_embeddings,
        )
        if not self.emit_logits:
            # Intermediate prefill only commits KV. The final one-token tail
            # uses a decode graph to obtain logits, so this head is dead work.
            global_k, global_v = self.extend.global_cache.updates()
            sliding_k, sliding_v = self.extend.sliding_cache.updates()
            return global_k, global_v, sliding_k, sliding_v
        # Keep the tied vocabulary projection at physical batch 1. Broadcasting
        # [1,V,H] over request batch B sent the huge head matmul to GPU for B=2.
        # Fold request/token into one spatial sequence, then restore logical rows.
        b, q, h = hidden.shape
        table = embedding_table
        if table.dtype == torch.int8:
            table = dequantize_per_tensor(table, self.extend.emb_scale,
                                          self.extend.emb_zero_point, hidden.dtype)
        table = table.reshape(1, table.shape[0], h)
        projection = hidden.reshape(1, b * q, 1, h).transpose(-3, -1).reshape(1, 1, h, b * q)
        logits = (table @ projection).transpose(-1, -2).reshape(b, 1, q, table.shape[1])
        global_k, global_v = self.extend.global_cache.updates()
        sliding_k, sliding_v = self.extend.sliding_cache.updates()
        return logits, global_k, global_v, sliding_k, sliding_v


class FlatKVBatchedReadonlyGemma4(nn.Module):
    """Rank-4 external KV contract; leading axis combines storing-layer and row."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, transformer_input, rope_cos, rope_sin, causal_mask,
                sliding_causal_mask, key_cache, value_cache, sliding_key_cache,
                sliding_value_cache, embedding_table, ple_embeddings):
        b = transformer_input.shape[0]
        caches = [x.reshape(x.shape[0] // b, b, *x.shape[1:]) for x in
                  (key_cache, value_cache, sliding_key_cache, sliding_value_cache)]
        result = self.model(transformer_input, rope_cos, rope_sin,
                                      causal_mask, sliding_causal_mask, *caches,
                                      embedding_table, ple_embeddings)
        if self.model.emit_logits:
            logits, *updates = result
            return (logits, *(x.reshape(x.shape[0] * b, *x.shape[2:]) for x in updates))
        return tuple(x.reshape(x.shape[0] * b, *x.shape[2:]) for x in result)


class SpatialGeGLUMLP(nn.Module):
    """Keep token positions on the spatial axis instead of folding them into N."""
    def __init__(self, original):
        super().__init__()
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj

    def forward(self, x):
        x = x.transpose(-3, -1)
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(self.up_proj(x) * gate).transpose(-3, -1)
