"""
Chapter 5: Self Attention — the heart of every transformer

Covers:
  5.2  Q, K, V projections
  5.3  Attention formula: softmax(QKᵀ / √d_k) V
  5.4  Visualizing the [T, T] attention matrix
  5.5  Causal masking (decoder-style, no peeking at the future)
  5.6  Full self-attention module from scratch
  5.7  Why we divide by sqrt(d_k) (softmax saturation)
  5.8  O(T²) memory cost
  5.9  PyTorch's optimized F.scaled_dot_product_attention

Run: python code/chapter05/self_attention.py
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 5.2 The three projections ───────────────────────────────────

print("=" * 60)
print("5.2 — Q, K, V projections")
print("=" * 60)

T, d_model, d_k = 5, 16, 16
X = torch.randn(T, d_model)

W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_k, bias=False)

Q, K, V = W_q(X), W_k(X), W_v(X)
print(f"X shape:    {X.shape}  ({T} tokens × {d_model} dims)")
print(f"Q, K shape: {Q.shape}  (queries / keys)")
print(f"V shape:    {V.shape}  (values)")


# ── 5.3 The attention formula ──────────────────────────────────

print("\n" + "=" * 60)
print("5.3 — Attention formula: softmax(QKᵀ / √d_k) V")
print("=" * 60)

scores = Q @ K.T                              # [T, T]
weights = F.softmax(scores / (d_k ** 0.5), dim=-1)
out = weights @ V                             # [T, d_k]

print(f"Scores shape:  {scores.shape}  (every query vs every key)")
print(f"Weights shape: {weights.shape}  (rows sum to 1)")
print(f"Row sums:      {weights.sum(dim=-1)}")
print(f"Output shape:  {out.shape}")


# ── 5.4 Visualize an attention matrix ──────────────────────────

print("\n" + "=" * 60)
print("5.4 — Visualize the attention matrix")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: a small hand-crafted attention pattern
    hand = torch.tensor([
        [0.60, 0.20, 0.10, 0.05, 0.03, 0.02],
        [0.10, 0.50, 0.20, 0.10, 0.05, 0.05],
        [0.05, 0.30, 0.40, 0.15, 0.05, 0.05],
        [0.02, 0.05, 0.30, 0.50, 0.08, 0.05],
        [0.02, 0.03, 0.05, 0.30, 0.50, 0.10],
        [0.01, 0.02, 0.03, 0.05, 0.40, 0.49],
    ])

    im = axes[0].imshow(hand.numpy(), cmap="viridis",
                        aspect="auto", vmin=0, vmax=0.7)
    axes[0].set_title("Hand-crafted pattern (each row → itself + prev)")
    axes[0].set_xlabel("attended-to position (key)")
    axes[0].set_ylabel("querying position (query)")
    for i in range(hand.size(0)):
        for j in range(hand.size(1)):
            axes[0].text(j, i, f"{hand[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if hand[i, j] < 0.4 else "black",
                         fontsize=8)
    plt.colorbar(im, ax=axes[0])

    # Right: real attention from a random untrained model
    torch.manual_seed(0)
    real_attn = SelfAttention(d_model=32, mask="causal") if False else None
    # Build weights ad-hoc so we can show a non-causal version
    Wq = nn.Linear(32, 32, bias=False)
    Wk = nn.Linear(32, 32, bias=False)
    Wv = nn.Linear(32, 32, bias=False)
    Xdemo = torch.randn(8, 32)
    Qd, Kd = Wq(Xdemo), Wk(Xdemo)
    real_w = F.softmax(Qd @ Kd.T / (32 ** 0.5), dim=-1).detach().numpy()

    im2 = axes[1].imshow(real_w, cmap="viridis", aspect="auto", vmin=0)
    axes[1].set_title("Random init, no causal mask (T=8)")
    axes[1].set_xlabel("attended-to position")
    axes[1].set_ylabel("querying position")
    for i in range(real_w.shape[0]):
        for j in range(real_w.shape[1]):
            axes[1].text(j, i, f"{real_w[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if real_w[i, j] < real_w.max() * 0.6 else "black",
                         fontsize=7)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("/tmp/attention_heatmap.png", dpi=120)
    print("Heatmap saved to: /tmp/attention_heatmap.png")
except ImportError as e:
    print(f"matplotlib not available, skipping plot: {e}")


# ── 5.5 Causal masking ─────────────────────────────────────────

print("\n" + "=" * 60)
print("5.5 — Causal mask")
print("=" * 60)

T = 6
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
print("Causal mask (True = blocked):")
print(causal_mask.int())

scores_demo = torch.randn(T, T)
masked = scores_demo.masked_fill(causal_mask, float("-inf"))
weights_demo = F.softmax(masked, dim=-1)

print("\nRow 2 attention (should only see positions 0–2):")
print(weights_demo[2].round(decimals=3).tolist())


# ── 5.6 Full self-attention module ──────────────────────────────

print("\n" + "=" * 60)
print("5.6 — SelfAttention module")
print("=" * 60)


class SelfAttention(nn.Module):
    """From-scratch self-attention with optional causal masking."""

    def __init__(self, d_model, d_k=None, d_v=None, mask=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.mask = mask    # "causal", "sliding:N", or None

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        # scores: [batch, seq_len, seq_len]

        if self.mask == "causal":
            T = x.size(1)
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))
        elif self.mask and self.mask.startswith("sliding:"):
            window = int(self.mask.split(":")[1])
            T = x.size(1)
            # Allow attending to positions [max(0, t-window) .. t].
            # idx layout: diff[t, s] = s - t (idx col s minus row t).
            # Block s > t (future: diff > 0) or s < t - window (diff < -window).
            idx = torch.arange(T, device=x.device)
            diff = idx.unsqueeze(0) - idx.unsqueeze(1)   # [T, T] = s - t
            block = (diff > 0) | (diff < -window)
            scores = scores.masked_fill(block, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ V


# Test
B, T, d = 2, 5, 16
x = torch.randn(B, T, d)

attn_causal = SelfAttention(d, mask="causal")
attn_full   = SelfAttention(d)
out_causal  = attn_causal(x)
out_full    = attn_full(x)

print(f"Input:           {x.shape}")
print(f"Causal output:   {out_causal.shape}")
print(f"Full output:     {out_full.shape}")

# Verify: with causal mask, output[t] only depends on x[0..t]
torch.manual_seed(42)   # reproducible weights
attn_causal_v = SelfAttention(d, mask="causal")
x_in = torch.randn(1, 5, d)
x_perturbed = x_in.clone()
x_perturbed[0, 4] = x_perturbed[0, 4] + 100.0   # perturb a future token
out_clean = attn_causal_v(x_in)
out_perturbed = attn_causal_v(x_perturbed)

print(f"\noutput[0, 0] unchanged after perturbing position 4: "
      f"{torch.allclose(out_clean[0, 0], out_perturbed[0, 0])}")
print(f"output[0, 4] changes after perturbing position 4:    "
      f"{not torch.allclose(out_clean[0, 4], out_perturbed[0, 4])}")


# ── 5.7 Why divide by sqrt(d_k) ────────────────────────────────

print("\n" + "=" * 60)
print("5.7 — Why divide by sqrt(d_k)")
print("=" * 60)

for d_k_test in [16, 64, 256, 1024]:
    torch.manual_seed(0)
    Q = torch.randn(1000, d_k_test)
    K = torch.randn(1000, d_k_test)
    raw = Q @ K.T                                 # [1000, 1000]
    scaled = raw / math.sqrt(d_k_test)
    raw_w   = F.softmax(raw, dim=-1)
    scale_w = F.softmax(scaled, dim=-1)
    # How concentrated is each row? (max weight per row, averaged)
    raw_max   = raw_w.max(dim=-1).values.mean().item()
    scale_max = scale_w.max(dim=-1).values.mean().item()
    print(f"d_k={d_k_test:4}  raw max-weight={raw_max:.3f}  "
          f"scaled max-weight={scale_max:.3f}")


# ── 5.8 O(T²) memory cost ─────────────────────────────────────

print("\n" + "=" * 60)
print("5.8 — O(T²) memory cost of attention")
print("=" * 60)

print(f"  T      |  scores matrix  |  float32 memory")
print(f"  -------+-----------------+----------------")
for T_test in [128, 512, 2048, 8192, 32768]:
    n_elem = T_test * T_test
    bytes_ = n_elem * 4
    if bytes_ < 1024 ** 2:
        mem = f"{bytes_ / 1024:.1f} KB"
    elif bytes_ < 1024 ** 3:
        mem = f"{bytes_ / 1024 ** 2:.1f} MB"
    else:
        mem = f"{bytes_ / 1024 ** 3:.2f} GB"
    print(f"  {T_test:5}  |  {n_elem:>13,}  |  {mem:>10}")


# ── 5.9 PyTorch's F.scaled_dot_product_attention ──────────────

print("\n" + "=" * 60)
print("5.9 — F.scaled_dot_product_attention (the production call)")
print("=" * 60)

B, H, T, d_k = 2, 4, 16, 32
Q = torch.randn(B, H, T, d_k)
K = torch.randn(B, H, T, d_k)
V = torch.randn(B, H, T, d_k)

# From-scratch version (causal)
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
weights = F.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
out_scratch = weights @ V

# PyTorch's built-in (with is_causal=True — uses the same causal mask)
out_native = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print(f"From-scratch output shape:  {out_scratch.shape}")
print(f"PyTorch native output shape: {out_native.shape}")
print(f"Numerically equal: {torch.allclose(out_scratch, out_native, atol=1e-5)}")

# Speed comparison on GPU isn't possible here (no CUDA),
# but on CPU the built-in has less Python overhead.
t0 = time.time()
for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
t1 = time.time()
print(f"100 calls of F.scaled_dot_product_attention: {(t1-t0)*1000:.1f}ms")


# ── 5.10 Sliding window mask ──────────────────────────────────

print("\n" + "=" * 60)
print("5.10 — Sliding window mask (attend to last N tokens)")
print("=" * 60)

sliding = SelfAttention(d_model=16, mask="sliding:3")
x_test = torch.randn(1, 8, 16)
out_sliding = sliding(x_test)

# Inspect the mask by running with and without — verify the pattern
with torch.no_grad():
    Q = sliding.W_q(x_test)
    K = sliding.W_k(x_test)
    scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
    T_x = x_test.size(1)
    idx = torch.arange(T_x)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)   # [T, T] = s - t
    block = (diff > 0) | (diff < -3)
    masked = scores.masked_fill(block, float("-inf"))
    sliding_w = F.softmax(masked, dim=-1)[0].numpy()  # [T, T]

print("Sliding window (window=3) attention pattern:")
print("(row t can attend to positions max(0, t-3) .. t)")
for i in range(T_x):
    row = " ".join(f"{sliding_w[i, j]:.2f}" for j in range(T_x))
    print(f"  pos {i}: {row}")


# ── 5.11 Permutation equivariance ─────────────────────────────

print("\n" + "=" * 60)
print("5.11 — Attention is permutation-equivariant")
print("=" * 60)

perm = torch.tensor([3, 0, 4, 1, 2])  # arbitrary shuffle
x_orig = torch.randn(1, 5, 16)
x_shuf = x_orig[:, perm]

attn_test = SelfAttention(d_model=16)   # no mask
out_orig = attn_test(x_orig)
out_shuf = attn_test(x_shuf)

# Output should be shuffled the same way
out_shuf_unperm = out_shuf[:, torch.argsort(perm)]
print(f"Shuffled output == unpermuted original: "
      f"{torch.allclose(out_orig, out_shuf_unperm, atol=1e-5)}")
print("(this is why positional embeddings are essential — see Chapter 4)")


print("\nDone — self-attention is the only operation in a transformer that")
print("       lets tokens exchange information. Everything else is per-position.")