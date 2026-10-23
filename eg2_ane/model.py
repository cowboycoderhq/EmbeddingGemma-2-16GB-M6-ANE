"""EmbeddingGemma 2 text tower in the Neural Engine layout.

Same weights and math as `SentenceTransformer("google/embeddinggemma-2").encode(text)`, rewritten for the ANE:
- (B, C, 1, S) activations, 1x1 convolutions, norms over the channel axis, attention one head at a time.
- The 262,144 x 512 token table is looked up on the host: the model takes inputs_embeds.
- Per-layer embeddings (PLE): one 512 -> 512 projection of inputs_embeds per layer, normalized.
- FP16 range: the residual stream is 1/RESID of its true value, every RMSNorm divides by its row maximum first, and
  the attention, MLP and PLE outputs are scaled up before their post norms.
- FP16 subnormals: the ANE flushes values below about 6e-5 to zero (apple/coreai-torch#138). Three changes keep
  small values out of that range: the PLE projection weights are scaled by PLE_IN, each norm divides by max/SQ
  (values in [-SQ, SQ] before the square), and the post-norm input scales above. In FP32 each change is exact,
  because the norms do not change with the scale of their input.
- Padding mask with torch.where (coreai-torch 0.4.2 converts the arithmetic + permute form wrongly,
  apple/coreai-torch#137). Sliding-window layers get the band |q - k| < 512 when the input is longer than 512.

GELU: the model uses the tanh approximation. GELU="sigmoid2" computes the same function as
x * sigmoid(x * (a + b x^2)), which is more accurate on the ANE (cosine 0.999997 vs 0.99998) and about 11% slower.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from eg2_ane.common import HIDDEN, PROMPTS, REPO, REVISION  # noqa: F401  (PROMPTS re-exported for the tests)

RESID = 8.0       # the residual stream is 1/RESID of its true value
ATTN_OUT = 16.0   # attention output scale before its post norm
FF_OUT = 256.0    # MLP output scale before its post norm
PLE_OUT = 16.0    # PLE block output scale before its post norm
PLE_IN = 1024.0   # PLE projection weight scale: its outputs (max ~0.05) are otherwise near the FP16 subnormal range
SQ = 8.0          # norms square x / (max / SQ)
NEG = -1e4
GELU = "tanh"     # MLP GELU form: "tanh" (native op) or "sigmoid2" (more accurate on the ANE, slower)


def gelu(x, form: str = "tanh"):
    """The tanh-approximation GELU. "sigmoid2" is the same function: 0.5 (1 + tanh(z)) = sigmoid(2 z)."""
    if form == "tanh":
        return nn.functional.gelu(x, approximate="tanh")
    if form == "sigmoid2":
        return x * torch.sigmoid(x * (1.5957691216057308 + 0.07135481627 * x * x))
    raise ValueError(form)


def load_sentence_transformer(repo: str | None = None, revision: str | None = REVISION):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(repo or REPO, revision=revision, device="cpu",
                             model_kwargs={"attn_implementation": "eager", "dtype": torch.float32})
    return st.eval().float()


def text_model(st):
    return st[0].auto_model.language_model


def embed_table(st, resid: float = RESID) -> np.ndarray:
    """Host-side lookup table: token embedding * embed_scale / resid, FP16 (262,144 x 512)."""
    emb = text_model(st).embed_tokens
    return (emb.weight.detach().float() * (float(emb.embed_scale) / resid)).numpy().astype(np.float16)


def tokenize(tokenizer, text: str, seq: int):
    """The sentence-transformers 6.2 call: no strip (a leading space changes the first tokens), truncation keeps the
    first seq - 1 tokens and the end token, padding at the end."""
    t = tokenizer(text, padding="max_length", truncation=True, max_length=seq, return_tensors="np")
    return t["input_ids"].astype(np.int32), t["attention_mask"].astype(np.int32)


def host_embeds(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(table[ids].transpose(0, 2, 1)[:, :, None, :])


def conv(lin: nn.Linear, scale: float = 1.0) -> nn.Conv2d:
    c = nn.Conv2d(lin.in_features, lin.out_features, 1, bias=lin.bias is not None)
    with torch.no_grad():
        c.weight.copy_(lin.weight[:, :, None, None] * scale)
        if lin.bias is not None:
            c.bias.copy_(lin.bias * scale)
    return c


class NormC(nn.Module):
    """RMSNorm over the channel axis of (B, C, 1, S), max-divided first. in_scale: the input is in_scale x its true
    value (eps is scaled to match); scale_out multiplies the result."""

    def __init__(self, dim: int, eps: float, weight: torch.Tensor | None, scale_out: float = 1.0,
                 in_scale: float = 1.0):
        super().__init__()
        self.eps = eps * in_scale ** 2
        w = torch.ones(dim) if weight is None else weight.detach().float()
        self.register_buffer("w", (w * scale_out)[None, :, None, None].float(), persistent=False)

    def forward(self, x):
        c = x.abs().amax(1, keepdim=True).clamp(min=1e-3) / SQ
        y = x / c
        return y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + (self.eps ** 0.5 / c) ** 2) * self.w


def rotate_half(x):
    h = x.shape[1] // 2
    return torch.cat((-x[:, h:], x[:, :h]), dim=1)


class PLE(nn.Module):
    """One layer's slice of the per-layer embedding projection, normalized. The input is inputs_embeds (1/resid of
    true); the output of the convolution is gain/resid x the true projection, and the norm removes the scale."""

    def __init__(self, proj: nn.Linear, rows: slice, scale: float, eps: float, norm, resid: float = RESID,
                 gain: float = PLE_IN):
        super().__init__()
        w = proj.weight.detach().float()[rows] * scale * gain
        self.register_buffer("w", w[:, :, None, None], persistent=False)
        self.norm = NormC(rows.stop - rows.start, eps, norm.weight if norm is not None else None,
                          in_scale=gain / resid)

    def forward(self, x):
        return self.norm(torch.nn.functional.conv2d(x, self.w))


class Layer(nn.Module):
    def __init__(self, layer, cfg, cos, sin, resid: float, ple: PLE):
        super().__init__()
        a, m = layer.self_attn, layer.mlp
        # read each layer from its own attention module: sliding layers have head_dim 256 and 2 KV heads, full layers
        # head_dim 512 and 1 KV head
        self.head_dim = a.head_dim
        self.groups = a.num_key_value_groups
        self.q_heads = a.q_proj.out_features // self.head_dim
        self.kv_heads = self.q_heads // self.groups
        self.scaling = a.scaling
        eps = cfg.rms_norm_eps
        self.ln_in = NormC(cfg.hidden_size, eps, layer.input_layernorm.weight)
        self.q, self.k, self.v = conv(a.q_proj), conv(a.k_proj), conv(a.v_proj)
        self.o = conv(a.o_proj, ATTN_OUT)
        self.qn = NormC(self.head_dim, eps, a.q_norm.weight)
        self.kn = NormC(self.head_dim, eps, a.k_norm.weight)
        self.vn = NormC(self.head_dim, eps, None)  # v_norm has no scale
        self.ln_post_attn = NormC(cfg.hidden_size, eps, layer.post_attention_layernorm.weight, 1.0 / resid, ATTN_OUT)
        self.ln_pre_ff = NormC(cfg.hidden_size, eps, layer.pre_feedforward_layernorm.weight)
        self.gate, self.up = conv(m.gate_proj), conv(m.up_proj)
        self.down = conv(m.down_proj, FF_OUT)
        self.ln_post_ff = NormC(cfg.hidden_size, eps, layer.post_feedforward_layernorm.weight, 1.0 / resid, FF_OUT)
        pb = layer.ple_block
        self.ple_gate = conv(pb.per_layer_input_gate)
        self.ple_proj = conv(pb.per_layer_projection, PLE_OUT)
        self.ln_post_ple = NormC(cfg.hidden_size, eps, pb.post_per_layer_input_norm.weight, 1.0 / resid, PLE_OUT)
        self.ple = ple
        self.resid = resid
        self.gelu_form = GELU
        self.register_buffer("layer_scalar", layer.layer_scalar.detach().float().reshape(1))
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rope(self, x):
        return x * self.cos + rotate_half(x) * self.sin

    def attn(self, q, k, v, mask_add):
        hd = self.head_dim
        kv = []  # per KV head: norm over head_dim, RoPE on the key
        for j in range(self.kv_heads):
            kv.append((self.rope(self.kn(k[:, j * hd:(j + 1) * hd])), self.vn(v[:, j * hd:(j + 1) * hd])))
        outs = []
        for i in range(self.q_heads):
            ki, vi = kv[i // self.groups]
            qi = self.rope(self.qn(q[:, i * hd:(i + 1) * hd]))
            s = torch.einsum("bchq,bchk->bkhq", qi, ki) * self.scaling + mask_add
            outs.append(torch.einsum("bkhq,bchk->bchq", s.softmax(dim=1), vi))
        return torch.cat(outs, dim=1)

    def forward(self, h, ple_in, mask_add):
        x = self.ln_in(h)
        h = h + self.ln_post_attn(self.o(self.attn(self.q(x), self.k(x), self.v(x), mask_add)))
        x = self.ln_pre_ff(h)
        h = h + self.ln_post_ff(self.down(gelu(self.gate(x), self.gelu_form) * self.up(x)))
        # the gate input must be at its true scale (GELU is not linear); the native GELU is accurate here
        x = gelu(self.ple_gate(h) * self.resid, "tanh") * ple_in
        h = h + self.ln_post_ple(self.ple_proj(x))
        return h * self.layer_scalar


class EG2TextANE(nn.Module):
    """inputs_embeds (B, 512, 1, S) FP16 from host_embeds(), attention_mask (B, S) int32 -> embedding (B, 768),
    L2-normalized."""

    def __init__(self, st, seq: int, resid: float = RESID, layers: int | None = None):
        super().__init__()
        tm = text_model(st)
        cfg = tm.config
        pos = torch.arange(seq)[None]
        dummy = torch.zeros(1, seq, cfg.hidden_size)
        rope = {}
        for lt in ("full_attention", "sliding_attention"):
            c, s = tm.rotary_emb(dummy, pos, lt)
            rope[lt] = (c[0].T[None, :, None, :].float(), s[0].T[None, :, None, :].float())
        q = torch.arange(seq)
        far = (q[None, :] - q[:, None]).abs() >= cfg.sliding_window
        self.use_band = bool(far.any())  # at seq <= 512 every sliding layer sees the full input
        self.register_buffer("band", (far.float() * NEG)[None, :, None, :])  # (1, K, 1, Q)
        proj = tm.ple.per_layer_model_projection
        per = cfg.hidden_size_per_layer_input
        n = layers or cfg.num_hidden_layers
        self.layers = nn.ModuleList(
            Layer(tm.layers[i], cfg, *rope[cfg.layer_types[i]], resid,
                  PLE(proj, slice(i * per, (i + 1) * per), cfg.hidden_size ** -0.5, cfg.rms_norm_eps,
                      tm.ple.per_layer_projection_norm, resid))
            for i in range(n))
        self.sliding = [t == "sliding_attention" for t in cfg.layer_types[:n]]
        self.norm = NormC(cfg.hidden_size, cfg.rms_norm_eps, tm.norm.weight)
        self.proj_out = conv(tm.embedding_projection)  # 512 -> 768

    def forward(self, inputs_embeds, attention_mask):
        m = attention_mask.to(inputs_embeds.dtype)[:, None, None, :]
        mask_add = torch.where(attention_mask[:, :, None, None] > 0, 0.0, NEG).to(inputs_embeds.dtype)
        mask_slide = mask_add + self.band if self.use_band else mask_add
        h = inputs_embeds
        for layer, sliding in zip(self.layers, self.sliding):
            h = layer(h, layer.ple(inputs_embeds), mask_slide if sliding else mask_add)
        h = self.norm(h)
        pooled = (h * m).sum(3, keepdim=True) / m.sum(3, keepdim=True).clamp(min=1.0)
        e = self.proj_out(pooled)
        e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return e[:, :, 0, 0]
