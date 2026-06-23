# Building GPT From Scratch

## Chapter 6

# Multi-Head Attention

> *"One attention pattern isn't enough. The interesting relationships in language live in many directions at once."*

---

## 6.1 Why Multiple Heads?

A single self-attention layer computes one set of `Q, K, V` projections and produces one `[T, T]` attention matrix. That's one *kind* of relationship — whatever the layer learned to look for.

But the interesting relationships in language are plural:

- "What noun does this verb agree with?" — syntactic
- "What did 'it' refer to?" — coreference
- "What word will come next?" — predictive
- "Which tokens belong to the same phrase?" — chunking

A single attention pattern can't specialize in all of these at once. The "what word comes next" head would have to ignore "what noun does this agree with" head's work, even though both might be useful simultaneously.

**Multi-head attention** runs `H` independent self-attention operations ("heads") in parallel, each with its own learned `W_q, W_k, W_v`. The heads are free to specialize. Their outputs are concatenated and mixed with a final linear projection.

The output is the same shape as a single-head attention — `[B, T, d_model]` — so multi-head attention is a **drop-in replacement** for self-attention in any transformer block.

---

## 6.2 Splitting the Dimension

The key idea: instead of one big attention of dimension `d_model`, run `H` small attentions of dimension `d_head`, where `H × d_head = d_model`.

```
        d_model = 512, H = 8, d_head = 64
        ┌─────────────────────────────────────┐
        │           d_model = 512              │
        └─────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼────┐   ┌────▼────┐   ...   ┌────▼────┐
   │ head 1  │   │ head 2  │         │ head 8  │
   │ d=64    │   │ d=64    │         │ d=64    │
   └────┬────┘   └────┬────┘         └────┬────┘
        │             │                  │
        └──────────────┬──────────────────┘
                       │
                ┌──────▼──────┐
                │  Concat →   │
                │  Linear →   │
                │  d_model    │
                └─────────────┘
```

The trick: don't even bother with `H` separate `W_q` matrices. Use one big `W_q ∈ ℝ^{d_model × d_model}` and reshape the output into `[B, H, T, d_head]`. The math is identical, the code is shorter, and GPUs love it because the `H` dimension becomes a batch dimension that the matrix-multiply kernel can parallelize over.

```python
# Instead of H separate (d_model × d_head) matrices:
#   H * d_head * d_model = d_model² parameters
# Use one (d_model × d_model) matrix:
#   d_model² parameters  (same total, but fewer kernels to launch)

W_q = nn.Linear(d_model, d_model, bias=False)   # one big projection
Q = W_q(x)                                       # [B, T, d_model]
Q = Q.view(B, T, H, d_head).transpose(1, 2)      # [B, H, T, d_head]
```

The reshape is free (no data movement, just stride manipulation) on a contiguous tensor.

---

## 6.3 The Multi-Head Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention (GPT-2 style)."""

    def __init__(self, d_model, num_heads, max_seq_len=None, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # One big projection for all heads (math is equivalent to H small ones)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection mixes the heads back together
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (registered as a buffer so it moves with .to(device))
        if max_seq_len is not None:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            )

    def forward(self, x):
        # x: [B, T, d_model]
        B, T, C = x.shape

        # Project all heads at once
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        # Q, K, V: [B, H, T, d_head]

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)   # [B, H, T, T]

        # Causal mask
        scores = scores.masked_fill(
            self.causal_mask[:T, :T], float("-inf")
        )

        weights = F.softmax(scores, dim=-1)                       # [B, H, T, T]
        weights = self.attn_dropout(weights)

        # Weighted sum of values
        out = weights @ V                                          # [B, H, T, d_head]

        # Concatenate heads back: [B, H, T, d_head] -> [B, T, H, d_head] -> [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear mixes information across heads
        out = self.W_o(out)                                        # [B, T, d_model]
        out = self.resid_dropout(out)
        return out
```

The `.contiguous()` call is required because `transpose` returns a non-contiguous view and `view` needs contiguous memory. Calling `.contiguous()` does a real copy, so this is one of the few places in a transformer where an allocation is unavoidable.

---

## 6.4 Why `W_o` Matters

The output projection `W_o` is what lets the heads *cooperate*. Without it, each head's contribution lives in its own `d_head`-sized slice of the output, and the rest of the model has to read them all separately.

`W_o` mixes information across heads. After training, you can read off what the model considers "head specialization" by looking at which heads contribute most to which `W_o` rows. Often you'll find:

- A few heads do almost nothing (close to identity-zero output)
- Some heads specialize in attending to position `t-1` (the previous token)
- Some heads are "induction heads" — they look for a previous occurrence of the current token and copy what came after it
- Some heads are "attention sinks" — they always attend to position 0

This emergent specialization is a property of training, not architecture. Different training runs produce different head roles.

---

## 6.5 Parameter Count

A multi-head attention layer has:

| Component | Shape | Parameters |
|---|---|---|
| `W_q`, `W_k`, `W_v` | each `[d_model × d_model]` | `3 × d_model²` |
| `W_o` | `[d_model × d_model]` | `d_model²` |
| **Total** | | **`4 × d_model²`** |

For `d_model = 768` (GPT-2 small) that's ~2.36M parameters per attention layer. For a 12-layer model, that's ~28M — about a quarter of GPT-2 small's 124M total. The rest is in the feed-forward layers (next chapter) and embeddings.

**Memory sanity check.** The attention scores tensor `[B, H, T, T]` for `B=8, H=12, T=1024, float32` is 8 × 12 × 1024² × 4 bytes = 384 MB. That's a lot. Modern implementations (FlashAttention, Chapter 11) never materialize this tensor in HBM.

---

## 6.6 Grouped-Query Attention (GQA) and Multi-Query (MQA)

A bottleneck in inference is the **KV cache**: storing K and V for every previous token to avoid recomputing them at every generation step. The cache size is `B × H × T × d_head × 2` (for K and V) × 2 bytes (fp16) — a big fraction of memory for long sequences.

**Multi-Query Attention (MQA)** shares the same K and V across all heads — `H` queries but only 1 K and 1 V. This shrinks the cache by `H×` and is mostly free in quality.

**Grouped-Query Attention (GQA)** is a middle ground: share K and V across groups of `G` heads, so there are `H/G` distinct K/V pairs. Used in Llama 2 70B, Llama 3, Mistral, and most modern models. Quality is essentially identical to MHA, with MQA's memory savings.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len=None):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        # K and V project to num_kv_heads instead of num_heads
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        # Repeat K, V across the group dimension
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        # Now K, V have the same shape as Q; standard attention
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        # ... causal mask, softmax, weighted sum, etc.
```

For our Small Language Model we'll use plain MHA (simpler, plenty of capacity). We'll come back to GQA in Chapter 11 when we look at inference optimization.

---

## 6.7 The Math, All Together

For completeness, here is the entire forward pass of a multi-head attention layer in math:

```
Input:    X ∈ ℝ^{B × T × d_model}

Project:  Q = X W_q    ∈ ℝ^{B × T × d_model}
          K = X W_k    ∈ ℝ^{B × T × d_model}
          V = X W_v    ∈ ℝ^{B × T × d_model}

Reshape:  Q → [B, H, T, d_head]    (split last dim, swap to heads-first)
          K → [B, H, T, d_head]
          V → [B, H, T, d_head]

Scores:   S = Q K^T / √d_head        ∈ ℝ^{B × H × T × T}

Mask:     S[i, j] = -∞   for j > i   (causal)

Weights:  A = softmax(S)             ∈ ℝ^{B × H × T × T}     (rows sum to 1)

Context:  C = A V                    ∈ ℝ^{B × H × T × d_head}

Merge:    C → [B, T, H × d_head]     (concatenate heads)
          C → [B, T, d_model]        (via W_o)
```

That's it. 4 projections, 1 matrix multiply for scores, 1 softmax, 1 matrix multiply for context, 1 final projection. Everything else is reshaping and masking.

---

## Chapter Summary

- **Multi-head attention** runs `H` independent self-attention operations in parallel, each with its own `Q, K, V` projections.
- All heads' outputs are concatenated and mixed with a final `W_o` projection back to `d_model`.
- Each head has dimension `d_head = d_model / H`, and the heads operate on different *subspaces* of the residual stream.
- Implementation trick: use one big `[d_model × d_model]` projection and reshape into `[B, H, T, d_head]` — same math, fewer kernel launches.
- Heads **emerge** as specialists during training (induction heads, previous-token heads, attention sinks).
- **GQA / MQA** share K, V across heads to shrink the KV cache during inference.
- Multi-head attention is a **drop-in replacement** for self-attention. In the next chapter we wrap it with layer norm, residual connections, and a feed-forward network to make a complete transformer block.

---

## Exercises

1. **Head count tradeoff.** Set `d_model = 256` and compare `H ∈ {1, 2, 4, 8, 16}`. Does adding more heads always help? At what point do you see diminishing returns on a small training run?
2. **Inspect head specialization.** Train a small MHA on Tiny Shakespeare, then for each head compute the mean attention distance (the average |t - s| it attends to). Do some heads specialize in local vs. long-range attention?
3. **GQA training dynamics.** Train a small GQA model and a comparable MHA model from the same init. Does GQA converge at the same rate?
4. **Dropout placement.** Add `attn_dropout` to the weights and `resid_dropout` to the output. Which one matters more for regularization? (Hint: try setting each to 0.0 in turn.)
5. **Reshape exercise.** Given `Q = torch.randn(B, T, d_model)` and `num_heads = H`, write the reshape+transpose that produces `[B, H, T, d_head]`. Reverse it to recover the original shape. Print the shapes at every step.
6. **Ablate W_o.** Replace `W_o` with the identity (no projection). Does the model train? If it does, is the final loss better or worse?
7. **Visualize heads.** Train a small model, then plot the `[H, T, T]` attention matrix averaged over a few examples. Can you identify induction heads? (See the Anthropic "In-context Learning and Induction Heads" paper.)

The full implementation lives in `code/chapter06/multi_head_attention.py` — try `num_heads ∈ {1, 4, 8, 16}` and watch the parameter count and head-specialization behavior change.