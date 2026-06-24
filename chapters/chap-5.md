# Building GPT From Scratch

## Chapter 5

# Self Attention

> *"Attention is the only operation in a transformer that lets a token know about its neighbors. Everything else is just per-position arithmetic."*

---

## 5.1 A Database Lookup, Softened

Imagine you have a database of 50,000 entries, and you want to find the entries most relevant to a query. A hard lookup returns zero or one match. A **soft** lookup returns a weighted combination of *all* entries, where the weights say how relevant each one is.

That soft lookup is **attention**.

In a language model, the "database" is the sequence of token embeddings you're processing. The "query" is a particular token's hidden state. The "weights" tell you how much each position in the sequence should contribute to that token's updated representation.

This chapter builds that operation from scratch, then uses it inside a causal mask to make a GPT-style decoder block.

---

## 5.2 Three Projections: Query, Key, Value

Self-attention gives every token three different roles, by multiplying the same input embedding by three different learned matrices:

```
Q = X @ W_q          # "what am I looking for?"
K = X @ W_k          # "what do I contain?"
V = X @ W_v          # "what do I reveal if I'm attended to?"
```

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `X` | `[T, d_model]` | input token embeddings |
| `W_q` | `[d_model, d_k]` | query projection |
| `W_k` | `[d_model, d_k]` | key projection |
| `W_v` | `[d_model, d_v]` | value projection |
| `Q`, `K` | `[T, d_k]` | queries and keys (same dim) |
| `V` | `[T, d_v]` | values (often `d_v = d_model`) |
| `output` | `[T, d_v]` | context-aware token representations |

The names come from databases. A *query* asks a question. A *key* advertises what an entry is about. The dot product `q · k` measures how well a query matches a key — that is, how relevant one token is to another.

```python
import torch
import torch.nn as nn

T, d_model, d_k = 5, 16, 16
X = torch.randn(T, d_model)

W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_k, bias=False)

Q = W_q(X)        # [T, d_k]
K = W_k(X)        # [T, d_k]
V = W_v(X)        # [T, d_k]
print(Q.shape, K.shape, V.shape)
# torch.Size([5, 16]) torch.Size([5, 16]) torch.Size([5, 16])
```

The three matrices are independent and learned. After training, they specialize: `W_q` learns to project tokens into "question space," `W_k` into "answer-advertisement space," and `W_v` into "what to actually pass on if I'm selected."

---

## 5.3 The Attention Formula

Given queries `Q` and keys `K`, we compute every pairwise similarity:

\\[
\text{scores} = Q K^\top \in \mathbb{R}^{T \times T}
\\]

Then we scale and normalize:

\\[
\text{weights} = \text{softmax}\!\left(\frac{\text{scores}}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}
\\]

\\[
\text{output} = \text{weights} \cdot V \in \mathbb{R}^{T \times d_v}
\\]

That's it. The `1/sqrt(d_k)` divisor (Section 5.7) keeps the softmax from saturating when `d_k` is large.

```python
import torch.nn.functional as F

d_k = Q.size(-1)
scores = Q @ K.T                          # [T, T]
weights = F.softmax(scores / d_k**0.5, dim=-1)
out = weights @ V                         # [T, d_k]

print("Attention weights (rows sum to 1):")
print(weights.sum(dim=-1))
# tensor([1., 1., 1., 1., 1.])
print("Output shape:", out.shape)
# torch.Size([5, 16])
```

The output for each token is a **weighted sum of all other tokens' values**, with weights determined by query-key similarity. Every position now has access to information from every other position.

---

## 5.4 Reading the Attention Matrix

The `[T, T]` attention weight matrix is the most interpretable object in a transformer. It tells you, for every pair of positions, how much the first attends to the second.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Pretend we have some attention weights
T = 6
weights = torch.tensor([
    # attends to →
    [0.6, 0.2, 0.1, 0.05, 0.03, 0.02],   # token 0: mostly itself
    [0.1, 0.5, 0.2, 0.1, 0.05, 0.05],    # token 1
    [0.05, 0.3, 0.4, 0.15, 0.05, 0.05],  # token 2
    [0.02, 0.05, 0.3, 0.5, 0.08, 0.05],  # token 3
    [0.02, 0.03, 0.05, 0.3, 0.5, 0.1],   # token 4
    [0.01, 0.02, 0.03, 0.05, 0.4, 0.49], # token 5: mostly itself + token 4
])

plt.figure(figsize=(6, 5))
plt.imshow(weights.numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=0.7)
plt.colorbar(label="attention weight")
plt.xlabel("attended-to position (key)")
plt.ylabel("querying position (query)")
plt.title("Self-attention weights (5 tokens)")
for i in range(T):
    for j in range(T):
        plt.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                 color="white" if weights[i, j] < 0.4 else "black", fontsize=9)
plt.tight_layout()
plt.savefig("/tmp/attention_heatmap.png", dpi=120)
```

A trained model's attention matrix often has striking structure: a particular row might attend almost entirely to one column (a "head" specialized for a previous-token relationship), or form a diagonal band (local attention), or fan out to a single sink position ("attention sinks" — a Llama finding).

---

## 5.5 Causal Masking — Why GPT Can't See the Future

A language model is a **next-token predictor**: token at position `t` should only depend on tokens at positions `0..t`. If token 5 is allowed to attend to token 6, the model can simply copy the answer from the future and never learn to predict.

We enforce this with a **causal mask**: the upper triangle of the attention matrix is set to `-inf` before softmax, so those positions get zero weight.

```
Allowed attention (lower triangle + diagonal):
  1  0  0  0  0
  1  1  0  0  0
  1  1  1  0  0
  1  1  1  1  0
  1  1  1  1  1
```

```python
T = 5
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# True above the diagonal, False on/below
print(causal_mask)
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])

scores = torch.randn(T, T)
scores = scores.masked_fill(causal_mask, float("-inf"))
weights = F.softmax(scores, dim=-1)

print("Row 2 (token 2) attends to:")
print(weights[2])
# tensor([0.31, 0.45, 0.24, 0.00, 0.00])  ← tokens 3, 4 masked out
```

This is the key difference between a **decoder** (GPT) and an **encoder** (BERT). Encoders see the whole sequence; decoders only see the past.

---

## 5.6 The Full Self-Attention Module

Putting it all together:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None, mask=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        self.mask = mask    # "causal" or None

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)
        # [batch, seq_len, seq_len]

        if self.mask == "causal":
            T = x.size(1)
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return weights @ V


# Test
B, T, d_model = 2, 5, 16
x = torch.randn(B, T, d_model)

attn = SelfAttention(d_model, mask="causal")
out = attn(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")   # [B, T, d_model]
# torch.Size([2, 5, 16])
```

That's a complete self-attention layer — about 12 lines of meaningful code. In the next chapter we'll run *several* of these in parallel (multi-head); in chapter 7 we'll wrap it in a residual stream, layer norm, and feed-forward to make a full transformer block.

---

## 5.7 Why Divide by sqrt(d_k)?

The dot products `q · k` grow with the dimension. If `q` and `k` are unit-variance vectors, then `q · k` has variance `d_k`. For `d_k = 64`, the typical score magnitude is ±8. After softmax, that distribution becomes essentially one-hot — the model attends entirely to the highest-scoring position and learns nothing useful from the others.

Dividing by `sqrt(d_k)` rescales scores back to unit variance, keeping the softmax in a region where gradients flow.

```python
import math

for d_k in [16, 64, 256, 1024]:
    Q = torch.randn(1000, d_k)
    K = torch.randn(1000, d_k)
    raw = Q @ K.T
    print(f"d_k={d_k:4}  raw std={raw.std():6.2f}  "
          f"scaled std={raw.std() / math.sqrt(d_k):6.2f}")
# d_k=  16  raw std=  4.01  scaled std=  1.00
# d_k=  64  raw std=  8.00  scaled std=  1.00
# d_k= 256  raw std= 16.06  scaled std=  1.00
```

This is the "Scaled" in "Scaled Dot-Product Attention" from the original Transformer paper.

---

## 5.8 Complexity — The Catch

The attention matrix is `[T, T]`. Memory and compute both scale as **O(T²)** with sequence length. This is the central engineering constraint in modern LLMs:

| Sequence length | Attention matrix size | Float32 memory |
|---|---|---|
| 512 | 512² = 262K | 1 MB |
| 2048 | 2048² = 4.2M | 17 MB |
| 8192 | 8192² = 67M | 268 MB |
| 32,768 | 32,768² = 1.07B | 4.3 GB |
| 131,072 | 131,072² = 17B | 68 GB |

This quadratic blow-up is why:
- Inference uses **KV caching** (Chapter 11) — don't recompute the past
- Long-context models use **sliding window** or **sparse** attention
- **Flash Attention** (Chapter 11) processes attention without ever materializing the `[T, T]` matrix in HBM
- Llama 3.1's 128K context window was an engineering tour-de-force

For our SLM with `T = 256–1024`, O(T²) is totally fine.

---

## 5.9 Production: PyTorch's Built-in

You won't write the loop yourself in production. PyTorch's `F.scaled_dot_product_attention` does everything in this chapter — including causal masking, scaling, and (on supported GPUs) Flash Attention — in a single call:

```python
# Modern way
out = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=None,            # or a causal mask
    is_causal=True,            # built-in causal mask (faster)
    dropout_p=0.0,             # attention dropout
)
# out: [B, T, d_v]
```

When `is_causal=True`, PyTorch skips materializing the mask tensor entirely and uses an optimized kernel. The behavior is identical to our from-scratch version — but it runs 2–4× faster on GPU and is numerically stable.

In `code/chapter05/self_attention.py` we use this for the perf comparison, but the rest of the chapter is the from-scratch version because seeing the math is the point.

---

## Chapter Summary

- **Self-attention** lets every token gather information from every other token via a learned weighted sum.
- Three learned projections split each token's role into **Query** ("what I'm looking for"), **Key** ("what I offer"), and **Value** ("what I contribute if selected").
- The attention formula is `softmax(QKᵀ / √d_k) V` — three lines of math.
- **Causal masking** sets the upper triangle of the attention matrix to `-inf` so a token can only attend to past positions. This is what makes GPT a *next-token* predictor instead of a *fill-in-the-middle* model.
- The scaling factor `1/√d_k` keeps the softmax in a numerically healthy range.
- Attention is **O(T²)** in both compute and memory. For long sequences, this drives most of the engineering complexity in modern LLMs.

In Chapter 6, we run several attention "heads" in parallel and combine their outputs — **multi-head attention** — which lets the model attend to several different relationships at once.

---

## Exercises

1. **Hand-compute a 3-token attention.** Set `Q = [[1,0], [0,1], [1,1]]` and `K = V = Q`. Compute `softmax(QKᵀ/√2) V` by hand. Which token does position 0 attend to most? Does the answer change if you double `Q`? Why?
2. **Visualize causal attention.** Render the `[T, T]` mask for `T = 8` with `imshow`. Where is the "future"? Where is the "past"?
3. **No mask vs. causal mask.** Run the same input through `SelfAttention(mask=None)` and `SelfAttention(mask="causal")`. Print the two output matrices. The causal one should have the property that output row 0 depends only on input row 0, output row 1 on rows 0–1, etc. Verify.
4. **Softmax saturation.** Compute attention on random `Q, K ∈ ℝ^{1000 × 64}` with and without `1/√d_k`. How many entries in the softmax output are above 0.99? What about above 0.999?
5. **Memory profile.** Time and profile `SelfAttention(d_model=512)` on a sequence of length 1024 vs. 4096. How does memory scale? Where is the bottleneck — `Q@K.T` or the softmax?
6. **Attention is permutation-equivariant.** Shuffle the input tokens; the output tokens get shuffled the same way. Prove this with code. **What does this tell you about why we need positional embeddings (Chapter 4)?**
7. **Different mask shapes.** Implement a "sliding window" mask that allows each token to attend only to the previous 4 tokens. How does the attention pattern look? Compare to a full causal mask.

The full implementation lives in `code/chapter05/self_attention.py` — run it, modify the mask, and watch the attention patterns change.