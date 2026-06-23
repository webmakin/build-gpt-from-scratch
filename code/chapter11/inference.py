"""
Chapter 11: Inference Optimization

Covers:
  11.2 KV caching — attention with cached K, V
  11.3 Causal mask with cache
  11.4 Batched generation
  11.6 Quantization (NF4 toy implementation)
  11.7 Speculative decoding
  11.12 Batching sweep

Run: python code/chapter11/inference.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Reference: uncached attention ──────────────────────────────

class UncachedAttention(nn.Module):
    """Standard attention: full Q, K, V projections, full attention matrix."""

    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ── 11.2 KV-cached attention ──────────────────────────────────

class CachedAttention(nn.Module):
    """Attention that accepts a cache and returns an updated one.

    Input x: [B, T_new] — just the new tokens (T_new=1 for generation).
    cache: dict with 'k', 'v' of shape [B, n_head, T_past, d_head], or None.
    Returns: (output [B, T_new, d_model], new_cache).
    """

    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.max_len = max_len

    def forward(self, x, cache=None):
        B, T_new, C = x.shape

        # Project only the NEW tokens
        Q_new = self.W_q(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
        K_new = self.W_k(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)
        V_new = self.W_v(x).view(B, T_new, self.n_head, self.d_head).transpose(1, 2)

        # Concat with cache (or just use new if no cache yet)
        if cache is not None:
            K = torch.cat([cache["k"], K_new], dim=2)
            V = torch.cat([cache["v"], V_new], dim=2)
        else:
            K, V = K_new, V_new

        new_cache = {"k": K.detach(), "v": V.detach()}

        # Q for new tokens attends to all K (cached + new)
        T_total = K.size(2)
        scores = Q_new @ K.transpose(-2, -1) / (self.d_head ** 0.5)

        # Causal mask: new token at position T_past + i cannot see future new tokens
        if T_new > 1:
            T_past = T_total - T_new
            i = torch.arange(T_new, device=x.device).unsqueeze(1)
            j = torch.arange(T_total, device=x.device).unsqueeze(0)
            mask = j > (T_past + i)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T_new, C)
        return self.W_o(out), new_cache


# ── 11.2 demonstration: cached matches uncached ───────────────

print("=" * 60)
print("11.2 — KV cache: cached output equals uncached output")
print("=" * 60)

torch.manual_seed(0)
D, H, T = 64, 4, 32
x = torch.randn(1, T, D)

# Make a cached attention with the SAME weights as uncached
cached = CachedAttention(D, H, T)
uncached = UncachedAttention(D, H, T)
for p_c, p_u in zip(cached.parameters(), uncached.parameters()):
    p_u.data = p_c.data.clone()

# Uncached forward: all at once
out_uncached = uncached(x)

# Cached forward: one token at a time, building up the cache
out_cached_chunks = []
cache = None
for t in range(T):
    x_t = x[:, t:t+1, :]                     # [1, 1, D]
    out_t, cache = cached(x_t, cache)        # [1, 1, D]
    out_cached_chunks.append(out_t)

out_cached = torch.cat(out_cached_chunks, dim=1)  # [1, T, D]

# Compare
match = torch.allclose(out_uncached, out_cached, atol=1e-5)
max_err = (out_uncached - out_cached).abs().max().item()
print(f"  Cached == uncached:  {match}")
print(f"  Max abs error:       {max_err:.2e}")


# ── 11.2 speedup: cached vs uncached for long sequences ───────

print("\n" + "=" * 60)
print("11.2 — Speedup: cached vs uncached generation")
print("=" * 60)

device = "cpu"
D, H = 64, 8
N_GEN = 100   # tokens to generate
T_TRIALS = [64, 256, 1024]


def time_generation(attn, x, n_gen, use_cache):
    """Generate n_gen tokens, return total time.

    The cached version uses a pre-allocated buffer to avoid torch.cat
    reallocations on each step (this is what production systems do).
    """
    t0 = time.time()
    if not use_cache:
        # Naive: re-encode the full sequence every step
        seq = x
        for _ in range(n_gen):
            out = attn(seq)
            seq = torch.cat([seq, out[:, -1:]], dim=1)
    else:
        # Cached: pre-allocate, then only feed the new token each step
        B, T_past, C = x.shape
        max_total = T_past + n_gen
        k_buf = torch.empty(B, attn.n_head, max_total, attn.d_head, device=x.device)
        v_buf = torch.empty(B, attn.n_head, max_total, attn.d_head, device=x.device)
        cache = {"k": k_buf[:, :, :T_past], "v": v_buf[:, :, :T_past]}
        cur = x
        for _ in range(n_gen):
            x_t = cur[:, -1:, :]
            out, cache = attn(x_t, cache)
            cur = torch.cat([cur, out], dim=1)
    return time.time() - t0


print(f"  {'T_prompt':>10}  {'uncached':>10}  {'cached':>10}  {'speedup':>10}")
print(f"  {'--------':>10}  {'--------':>10}  {'------':>10}  {'-------':>10}")
for T_p in T_TRIALS:
    torch.manual_seed(0)
    cached_a = CachedAttention(D, H, T_p + N_GEN).to(device)
    uncached_a = UncachedAttention(D, H, T_p + N_GEN).to(device)
    for p_c, p_u in zip(cached_a.parameters(), uncached_a.parameters()):
        p_c.data = p_u.data.clone()

    x = torch.randn(1, T_p, D, device=device)

    # Warm up (first run has setup overhead)
    _ = time_generation(cached_a, x, 1, use_cache=True)
    _ = time_generation(uncached_a, x, 1, use_cache=False)

    t_uncached = time_generation(uncached_a, x, N_GEN, use_cache=False)
    t_cached = time_generation(cached_a, x, N_GEN, use_cache=True)
    print(f"  {T_p:>10}  {t_uncached*1000:>8.1f}ms  {t_cached*1000:>8.1f}ms  "
          f"{t_uncached/t_cached:>8.1f}×")


# ── 11.6 toy NF4 quantization ─────────────────────────────────

print("\n" + "=" * 60)
print("11.6 — Quantization: NF4 vs fp32")
print("=" * 60)

# The 16 NF4 quantization levels (4-bit, NormalFloat)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
])


def quantize_nf4(weight, block_size=64):
    """Block-wise NF4 quantization: each block of 64 weights gets its own scale."""
    flat = weight.flatten().contiguous()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    # Pad to a multiple of block_size
    padded = F.pad(flat, (0, n_blocks * block_size - n))
    padded = padded.view(n_blocks, block_size)
    # Per-block absolute max
    absmax = padded.abs().max(dim=1, keepdim=True).values   # [n_blocks, 1]
    absmax = absmax.clamp(min=1e-10)
    normalized = padded / absmax                            # [-1, 1]
    # Nearest NF4 level
    idx = torch.searchsorted(NF4_LEVELS.to(weight.device), normalized)
    idx = idx.clamp(0, 15)
    dequant = NF4_LEVELS.to(weight.device)[idx]              # [n_blocks, block_size]
    out = (dequant * absmax).view(-1)[:n].view_as(weight)
    n_bits = n * 4   # 4 bits per value
    return out, n_bits


# Toy: a weight matrix with normal-distributed entries
torch.manual_seed(0)
W = torch.randn(512, 512) * 0.02

W_quant, n_bits = quantize_nf4(W)
n_orig = W.numel() * 32
err = (W - W_quant).abs().mean().item()
# Signal-to-noise ratio: how much signal survived quantization
signal = (W ** 2).mean().item()
noise = ((W - W_quant) ** 2).mean().item()
snr_db = 10 * math.log10(signal / noise) if noise > 0 else float('inf')
print(f"  Weight shape:        {tuple(W.shape)}")
print(f"  fp32 size:           {n_orig/8/1024:.1f} KB")
print(f"  NF4 size:            {n_bits/8/1024:.1f} KB   ({n_bits/n_orig*100:.1f}% of fp32)")
print(f"  Mean abs error:      {err:.6f}")
print(f"  Signal-to-noise:     {snr_db:.1f} dB")
print(f"  (NF4 on real model weights is typically 25-35 dB SNR; the small")
print(f"   sample here shows the algorithm working but with toy statistics.)")
print(f"  → 8× memory savings")


# ── 11.7 Speculative decoding (simulation) ────────────────────

print("\n" + "=" * 60)
print("11.7 — Speculative decoding: speedup = 1 / (1 - acceptance_rate)")
print("=" * 60)

# Simulate a draft model with varying acceptance rates
for acceptance_rate in [0.3, 0.5, 0.7, 0.9]:
    # Average accepted tokens per round
    # E[accepted] = 1 + r + r^2 + r^3 + r^4 = (1 - r^5) / (1 - r)
    expected = sum(acceptance_rate ** i for i in range(5))
    # The target model cost is constant per round; speedup ≈ expected
    print(f"  acceptance={acceptance_rate:.1f} → "
          f"avg tokens/round = {expected:.2f} → "
          f"speedup ≈ {expected:.2f}×")


# ── 11.12 Batching sweep ──────────────────────────────────────

print("\n" + "=" * 60)
print("11.12 — Batching: per-request and aggregate throughput")
print("=" * 60)

device = "cpu"
D = 64
T_prompt = 64
N_GEN = 20
BATCH_SIZES = [1, 4, 16, 64, 128]

# Simulate batched forward pass: a [B, T, D] matmul scales linearly with B
def fake_forward(B, T, D, n_gen):
    """Simulate a forward pass: cost is B * T * D * n_gen."""
    # We can measure real time with a simple matmul
    x = torch.randn(B, T, D)
    t0 = time.time()
    for _ in range(n_gen):
        # Approximate the work: one matmul per layer, 12 layers
        for _ in range(12):
            y = x @ torch.randn(D, D) * 0.1
    return time.time() - t0


print(f"  {'batch':>6}  {'per-req (s)':>11}  {'per-req TPS':>11}  "
      f"{'agg TPS':>9}")
print(f"  {'-----':>6}  {'----------':>11}  {'-----------':>11}  "
      f"{'-------':>9}")
for B in BATCH_SIZES:
    t = fake_forward(B, T_prompt, D, N_GEN)
    per_req = t / B
    per_req_tps = N_GEN / per_req if per_req > 0 else 0
    agg_tps = N_GEN * B / t if t > 0 else 0
    print(f"  {B:>6}  {per_req:>9.4f}    {per_req_tps:>9.1f}    {agg_tps:>7.1f}")


print("\nDone — KV caching is mandatory. Flash Attention, quantization, and")
print("       continuous batching are the other big wins in production.")