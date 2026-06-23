"""
Chapter 6: Multi-Head Attention

Covers:
  6.3  MultiHeadAttention module (one big projection, reshape trick)
  6.4  Output projection W_o
  6.5  Parameter count: 4 × d_model² per layer
  6.6  Grouped-Query Attention (GQA)
  6.7  End-to-end forward pass, all-together

Run: python code/chapter06/multi_head_attention.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 6.3 Multi-head attention module ────────────────────────────

print("=" * 60)
print("6.3 — MultiHeadAttention")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention (GPT-2 style)."""

    def __init__(self, d_model, num_heads, max_seq_len=None, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if max_seq_len is not None:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            )

    def forward(self, x):
        # x: [B, T, d_model]
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, H, T, d_head]

        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)   # [B, H, T, T]
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        weights = F.softmax(scores, dim=-1)                       # [B, H, T, T]
        weights = self.attn_dropout(weights)

        out = weights @ V                                          # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # [B, T, d_model]
        out = self.W_o(out)
        out = self.resid_dropout(out)
        return out

    def return_attention(self, x):
        """Variant that also returns the attention weights for visualization."""
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out), weights


# Test
B, T, d_model, H = 2, 8, 64, 4
mha = MultiHeadAttention(d_model=d_model, num_heads=H, max_seq_len=128)
x = torch.randn(B, T, d_model)
out = mha(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}  (same shape as input — drop-in for self-attention)")
print(f"d_head: {mha.d_head}  (= d_model / H = {d_model} / {H})")


# ── 6.4 W_o matters ────────────────────────────────────────────

print("\n" + "=" * 60)
print("6.4 — W_o (output projection)")
print("=" * 60)
print(f"W_q, W_k, W_v:  each {d_model}×{d_model} = {d_model*d_model:>6,} params")
print(f"W_o:                  {d_model}×{d_model} = {d_model*d_model:>6,} params")
n = sum(p.numel() for p in mha.parameters())
print(f"Total:                              {n:>6,} params  (= 4 × d_model²)")


# ── 6.5 Parameter count sweep ─────────────────────────────────

print("\n" + "=" * 60)
print("6.5 — Parameter count across model sizes")
print("=" * 60)
print(f"  d_model  |  H  | d_head |  per-layer params  |   total (12 layers)")
print(f"  ---------+-----+--------+--------------------+----------------------")
for d, h in [(64, 4), (128, 4), (256, 4), (256, 8), (512, 8), (768, 12)]:
    d_head = d // h
    per_layer = 4 * d * d
    total = per_layer * 12
    print(f"  {d:>5}    | {h:>2}  |  {d_head:>3}   |     {per_layer:>10,}    |   {total:>14,}")


# ── 6.6 Grouped-Query Attention ───────────────────────────────

print("\n" + "=" * 60)
print("6.6 — Grouped-Query Attention (GQA)")
print("=" * 60)


class GroupedQueryAttention(nn.Module):
    """MHA where K and V are shared across groups of `group_size` heads."""

    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len=None, dropout=0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if max_seq_len is not None:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            )

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        # Repeat K, V across the group dimension
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.W_o(out))


# Compare parameter counts
d_model, num_heads = 512, 8
mha = MultiHeadAttention(d_model, num_heads, max_seq_len=128)
gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2, max_seq_len=128)
mqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1, max_seq_len=128)   # MQA

n_mha = sum(p.numel() for p in mha.parameters())
n_gqa = sum(p.numel() for p in gqa.parameters())
n_mqa = sum(p.numel() for p in mqa.parameters())

print(f"  d_model={d_model}, num_heads={num_heads}")
print(f"  MHA (H = {num_heads} KV heads): {n_mha:>9,} params  (full attention)")
print(f"  GQA (H = {num_heads} queries, {2} KV heads): {n_gqa:>9,} params  "
      f"({100*(1 - n_gqa/n_mha):.1f}% smaller)")
print(f"  MQA (H = {num_heads} queries, 1 KV head): {n_mqa:>9,} params  "
      f"({100*(1 - n_mqa/n_mha):.1f}% smaller)")

# Sanity check: forward pass works
x = torch.randn(2, 16, d_model)
out_gqa = gqa(x)
print(f"\nGQA output shape: {out_gqa.shape}")


# ── 6.7 Causal mask verification ──────────────────────────────

print("\n" + "=" * 60)
print("6.7 — Causal mask verification")
print("=" * 60)

# Each head should produce different attention (different W_k, W_v)
torch.manual_seed(0)
mha_test = MultiHeadAttention(d_model=32, num_heads=4, max_seq_len=16)
x_test = torch.randn(1, 8, 32)
out, weights = mha_test.return_attention(x_test)
# weights: [1, 4, 8, 8]
print(f"Weights shape: {weights.shape}  (batch=1, heads=4, T=8, T=8)")

# Per-row sums (should be 1)
print(f"Row sums: {weights[0, 0].sum(dim=-1).tolist()[:3]}...")

# Different heads should give different attention patterns
head_sims = []
for i in range(4):
    for j in range(i + 1, 4):
        # Compare per-row distributions
        a = weights[0, i].flatten()
        b = weights[0, j].flatten()
        sim = (a * b).sum().item()  # crude dot-product similarity
        head_sims.append(sim)
print(f"Average dot-product similarity between heads: {sum(head_sims)/len(head_sims):.3f}")
print("(non-zero but not 1.0 = heads are different)")


# ── 6.8 Visualize all heads ───────────────────────────────────

print("\n" + "=" * 60)
print("6.8 — Visualize all heads as a heatmap grid")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for h in range(4):
        ax = axes[h]
        attn_h = weights[0, h].detach().numpy()
        im = ax.imshow(attn_h, cmap="viridis", aspect="auto", vmin=0)
        ax.set_title(f"Head {h}")
        ax.set_xlabel("attended-to")
        ax.set_ylabel("querying")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Multi-head attention patterns (untrained model, causal mask)")
    plt.tight_layout()
    plt.savefig("/tmp/multi_head_attention.png", dpi=120)
    print("Heatmap saved to: /tmp/multi_head_attention.png")
except ImportError as e:
    print(f"matplotlib not available, skipping plot: {e}")


# ── 6.9 Train a tiny model and inspect head specialization ────

print("\n" + "=" * 60)
print("6.9 — Train a tiny model, inspect head specialization")
print("=" * 60)

# Use a tiny copy task: the model must learn that "the X Y the X Z" → predict "Y"
# This forces the model to use induction-head-like behavior.
def make_copy_data(n_samples=2000, seq_len=12, vocab_size=20, seed=0):
    torch.manual_seed(seed)
    X = torch.zeros(n_samples, seq_len, dtype=torch.long)
    Y = torch.full((n_samples, seq_len), -100, dtype=torch.long)   # -100 = ignore
    for i in range(n_samples):
        # Pattern: [a, b, a, c]  →  Y at position of 'c' is 'b'
        a = torch.randint(2, vocab_size, (1,)).item()
        b = torch.randint(2, vocab_size, (1,)).item()
        c = torch.randint(2, vocab_size, (1,)).item()
        # Random base
        X[i, 0] = 1
        # Two random "name-value" pairs
        X[i, 1] = a
        X[i, 2] = b
        X[i, 3] = a
        X[i, 4] = b
        X[i, 5] = a
        X[i, 6] = c
        # We want the model to output 'b' at position 7
        X[i, 7] = 0     # mask
        Y[i, 7] = b
        # Fill rest with random
        for j in range(8, seq_len):
            X[i, j] = torch.randint(0, vocab_size, (1,)).item()
    return X, Y


X, Y = make_copy_data(n_samples=500, seq_len=12, vocab_size=20)
print(f"Data: X={X.shape}, Y={Y.shape}  (loss only on Y != -100)")


# Tiny model: embed + 1 MHA layer + output projection
class TinyModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=4, max_seq_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, dropout=0.0)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T))
        x = self.attn(x)
        return self.head(x)


torch.manual_seed(42)
model = TinyModel(vocab_size=20, d_model=32, num_heads=4, max_seq_len=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(800):
    idx = torch.randint(0, len(X), (32,))
    logits = model(X[idx])
    # Only compute loss on positions where Y != -100
    mask = Y[idx] != -100
    loss = F.cross_entropy(
        logits.view(-1, 20), Y[idx].view(-1), ignore_index=-100
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 200 == 0:
        # Accuracy only on the masked-in positions
        preds = logits.argmax(-1)
        correct = ((preds == Y[idx]) & mask).sum().float()
        total = mask.sum().float()
        acc = (correct / total).item() if total > 0 else 0.0
        print(f"  step {step+1:4}  loss {loss.item():.3f}  acc {acc:.2f}")


# ── 6.10 Per-head attention distance ──────────────────────────

print("\n" + "=" * 60)
print("6.10 — Per-head attention distance (specialization signal)")
print("=" * 60)

# For each head, compute mean attention distance: sum over (t, s) of |t - s| * weight[t, s]
model.eval()
with torch.no_grad():
    sample = X[:1]                              # [1, T]
    out, weights = model.attn.return_attention(
        model.embed(sample) + model.pos(torch.arange(sample.size(1)))
    )
    # weights: [1, 4, 12, 12]
    T = sample.size(1)
    positions = torch.arange(T).float()
    # Distance matrix |t - s|
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()   # [T, T]
    # For each head, look at attention FROM position 7 (the query position)
    # — that's where the model has to look back to find the answer
    print(f"  head  |  pos 7 attends to (top-3)        |  pattern")
    print(f"  ------+----------------------------------+-----------------")
    for h in range(4):
        # Get the row for position 7 in this head
        attn_row = weights[0, h, 7].detach()
        top3 = attn_row.topk(3)
        positions_str = ", ".join(
            f"pos {p.item()} ({attn_row[p].item():.2f})" for p in top3.indices
        )
        # Mean distance of THIS head over ALL positions (more general signal)
        mean_dist = (weights[0, h] * dist).sum().item() / T
        # Sink = fraction of attention going to position 0
        sink = weights[0, h, :, 0].sum().item() / T
        interp = ("local" if mean_dist < T / 3 else
                  "global" if mean_dist > 2 * T / 3 else
                  "mixed")
        print(f"  {h:>3}   |  {positions_str:32} |  {interp} "
              f"(mean_dist={mean_dist:.1f}, sink={sink:.2f})")


print("\nDone — multi-head attention is the operation that gives transformers")
print("       their ability to learn many relationship types at once.")