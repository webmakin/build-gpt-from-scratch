# Building GPT From Scratch

## Chapter 7

# Transformer Blocks

> *"A transformer block is just two operations — attention and a feed-forward network — each wrapped in a residual stream and a normalization layer. That's it. That's the whole architecture."*

---

## 7.1 The Anatomy of a Block

Every transformer block has the same structure:

```
                ┌──────────────────────┐
                │  LayerNorm           │
                │  (pre-norm variant)  │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  Multi-Head          │
                │  Attention           │
                └──────────┬───────────┘
                           │
                  + ◄──────┘     (residual connection)
                  │
                ┌─▼────────────────┐
                │  LayerNorm       │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  Feed-Forward    │
                │  Network         │
                └────────┬─────────┘
                         │
                + ◄──────┘         (residual connection)
                │
              output
```

Two operations (attention, FFN), each with a residual bypass, each preceded by a normalization. That's the entire transformer. Everything else in the model — embeddings, the final output projection, the loss — is *outside* the block.

There are two ordering conventions in the wild:

**Post-norm (original Transformer, 2017):**

\\[
y = x + \text{Sublayer}(\text{LayerNorm}(x))
\\]

**Pre-norm (most modern models: GPT-3, Llama, Mistral):**

\\[
y = x + \text{LayerNorm}(\text{Sublayer}(x))
\\]

Both work. Pre-norm is easier to train without warmup. Post-norm can reach slightly better final quality if you have a good learning rate schedule. For our SLM we'll use **pre-norm** because it's more forgiving.

---

## 7.2 The Residual Stream

The residual connection — `y = x + Sublayer(x)` — is the most important architectural idea in deep learning. It's what lets us stack 12, 96, or 200 blocks without the gradients vanishing or the activations blowing up.

The metaphor: the residual stream is a **highway** that information flows along. Each block's sublayer is a **side road** that reads from the highway, does some processing, and writes its result back. The original signal is always preserved.

```python
# A block with vs without residual — the difference is dramatic.

# No residual:
y = Sublayer(x)              # each block fully replaces x

# With residual:
y = x + Sublayer(x)          # each block adds to x
```

Without residuals, you'd need careful initialization, layer norm placement, and learning rate warmup just to train 4 blocks. With residuals, you can train 200 blocks with the same hyperparameters.

The output of attention and FFN should be initialized to contribute **very little** to the residual stream at the start of training (small init). This way, the model starts as the identity function and gradually learns deviations from it.

```python
# Standard GPT-2 init: scale residual contributions by 1/sqrt(2 * n_blocks)
# This keeps the residual stream's variance stable across many blocks.
nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)
# ... and a more aggressive scaling at the final layer
```

---

## 7.3 Layer Normalization

Layer norm normalizes across the *embedding dimension* for each token independently. For each token's vector `x ∈ ℝ^{d_model}`:

```
mean = x.mean()
var  = x.var()
y    = (x - mean) / sqrt(var + eps)
y    = y * gamma + beta          # learnable scale + shift
```

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

In **pre-norm** blocks, LayerNorm is applied *inside* the residual:

\\[
y = x + \text{Sublayer}(\text{LayerNorm}(x))
\\]

In **post-norm** blocks, LayerNorm is applied *outside* the residual:
```
y = LayerNorm(x + Sublayer(x))
```

Pre-norm doesn't normalize the residual stream itself, only the input to each sublayer. This makes the residual stream's magnitude roughly constant across blocks, which is why deep pre-norm transformers train stably.

---

## 7.4 RMSNorm — A Faster Alternative

Modern LLMs (Llama, Mistral) use **RMSNorm** instead of LayerNorm. It drops the mean-centering step and the learnable shift:

\\[
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
\\]

Half the compute of LayerNorm, identical quality in practice, and ~10-20% faster on GPU. The math is so close to LayerNorm that the difference is essentially noise for models above a few hundred million parameters.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # mean of x^2 over the last dim, keepdim for broadcasting
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

Three lines of math, no `mean` or `var` calls. The cost difference compounds over a 96-block model.

---

## 7.5 The Feed-Forward Network

The FFN is the "thinking" step of each block. Attention is the "communication" step (tokens exchange information), FFN is the "computation" step (each token thinks about what it has).

The original Transformer used two linear layers with a ReLU between them:

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

The hidden dimension is typically `4 × d_model`. So for `d_model = 768`, the FFN has `768 → 3072 → 768` — a 4x expansion then projection back. This is where most of the model's parameters live (in a 12-block model, about 2/3 of params are in the FFNs).

Modern LLMs (Llama, PaLM) use **SwiGLU** instead of plain ReLU. It has three matrices instead of two, but the hidden dimension is scaled down by 2/3 to keep the parameter count constant:

\\[
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_\text{gate}) \odot (x W_\text{up})\big) W_\text{down}
\\]

The `W_gate` and `W_up` both expand from `d_model → (8/3) × d_model` (rounded to a multiple of 64 for hardware), and `W_down` projects back. The total parameter count per FFN is `3 × d_model × (8/3 × d_model) = 8 × d_model²` — same as a 4× expansion ReLU FFN's `2 × 4 × d_model² = 8 × d_model²`.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        # Llama-style: hidden_dim = (8/3) * d_model, rounded to multiple of 64
        if hidden_dim is None:
            hidden_dim = int(2 * d_model * 4 / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.W_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
```

`F.silu` is `x * sigmoid(x)` — the SiLU / Swish activation. The gating (multiplying by `silu(W_gate(x))`) is what gives SwiGLU its name and its quality edge over plain ReLU.

---

## 7.6 Putting It All Together

Here's the full block:

```python
class TransformerBlock(nn.Module):
    """Pre-norm transformer block with MHA and SwiGLU FFN."""

    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0, ffn="swiglu"):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln2 = RMSNorm(d_model)

        if ffn == "swiglu":
            self.ffn = SwiGLU(d_model)
        elif ffn == "relu":
            hidden = 4 * d_model
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_model),
            )
        else:
            raise ValueError(f"Unknown ffn: {ffn}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm: norm before sublayer
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

That's a complete transformer block. About 8 meaningful lines. The rest is plumbing.

**Parameter count for one block** with `d_model = 512, num_heads = 8`:

| Component | Parameters |
|---|---|
| `ln1`, `ln2` (RMSNorm) | 1,024 |
| `MultiHeadAttention` (4 × 512²) | 1,048,576 |
| `SwiGLU` (3 × 512 × 1365) | 2,097,152 |
| **Block total** | **~3.15M** |

So in a 12-block model, the FFNs are ~2× the attention parameters — consistent with the rule of thumb that FFNs hold most of the weights.

---

## 7.7 Stacking Blocks

A transformer is `N` blocks stacked, with the same `MultiHeadAttention` and `SwiGLU` shape at each layer (but different learned weights). Plus an embedding at the input and a final normalization + output projection at the end.

```python
class Transformer(nn.Module):
    """Pre-norm transformer language model."""

    def __init__(self, vocab_size, d_model, num_heads, num_blocks,
                 max_seq_len, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, max_seq_len, dropout)
            for _ in range(num_blocks)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (Chapter 4): input and output share the same matrix
        self.head.weight = self.token_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)   # [B, T, vocab_size]
```

That's the whole model. Embedding, stack of blocks, final norm, output projection. **This is GPT** (modulo the specific attention implementation, which we covered in chapters 5-6).

---

## 7.8 Counting Parameters

For our Small Language Model, let's pick:
- `vocab_size = 8192`
- `d_model = 384`
- `num_heads = 6`
- `num_blocks = 6`
- `max_seq_len = 512`

| Component | Parameters |
|---|---|
| Token + position embeddings | 8192 × 384 + 512 × 384 = ~3.36M |
| 6 × TransformerBlock | 6 × ~885K = ~5.31M |
| Final norm + tied head | 384 (RMSNorm only — head is shared) |
| **Total** | **~8.7M** |

Compared to:
- GPT-2 small: 124M parameters
- Llama 2 7B: 7,000M parameters
- Llama 3 70B: 70,000M parameters

Our SLM is 14× smaller than GPT-2 small, 800× smaller than Llama 7B. Big enough to learn interesting patterns, small enough to train on a laptop in an afternoon.

---

## 7.9 Gradient Flow and Why Pre-Norm Matters

The residual connection makes the gradient flow trivial. If `y = x + f(x)`, then `∂y/∂x = I + ∂f/∂x`. The identity term means the gradient can always flow through the network unchanged, regardless of how badly `f` is behaving.

The problem with **post-norm** is that the residual stream's magnitude can grow or shrink as you stack blocks, because each block's output gets added to the stream. After 12 blocks of plain addition, the stream might be 12× larger (or smaller) than the input, and the layer norm has to fight harder to keep things in range. Deep post-norm transformers are notoriously hard to train without careful warmup.

**Pre-norm** breaks the cycle: each block's residual addition is `x + Sublayer(LayerNorm(x))`, and the norm guarantees that `Sublayer(LayerNorm(x))` is well-scaled. The residual stream's magnitude stays roughly constant through the entire stack. You can train a 200-block pre-norm transformer with the same learning rate you used for a 6-block one.

The benefit shows up most clearly *during training* — not as an obvious per-step gradient-norm difference at initialization, but as stability over thousands of steps without a learning-rate warmup. Post-norm models often diverge in the first few hundred steps unless you ramp the LR up gradually. Pre-norm models train cleanly with a constant LR from step 1.

---

## 7.10 Initialization

The original GPT-2 init is `std=0.02` for all linear layers and embeddings. The "scale residual contributions" trick is to multiply the output of each residual block's sublayer by `1/sqrt(2 * n_blocks)` so that the residual stream's variance stays stable.

```python
def init_weights(module, n_blocks, std=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    # Scale residual outputs by 1/sqrt(2*n_blocks)
    if isinstance(module, TransformerBlock):
        module.attn.W_o.weight.data *= (1 / (2 * n_blocks) ** 0.5)
        module.ffn.W_down.weight.data *= (1 / (2 * n_blocks) ** 0.5)
```

This is what GPT-2 used. Llama uses `std=0.02` without the residual scaling (relies on RMSNorm). Both work.

---

## Chapter Summary

- A **transformer block** is `attention + FFN`, each wrapped in a residual connection and a layer norm.
- **Pre-norm** (`x = x + Sublayer(LayerNorm(x))`) trains more stably than **post-norm** at depth.
- **Residual connections** are the highway that lets gradients flow unchanged through hundreds of blocks.
- **LayerNorm** normalizes across the embedding dim per token; **RMSNorm** is a cheaper drop-in (~10% faster).
- The **FFN** is typically a 4× expansion. Modern models use **SwiGLU** (3 matrices, gated) instead of plain ReLU.
- A complete transformer is: embedding → N blocks → final norm → output projection (often tied to the input embedding).

In Chapter 8, we put all of this together into the full GPT model and prepare it for training.

---

## Exercises

1. **Block ablation.** Replace RMSNorm with LayerNorm in your block. Replace SwiGLU with ReLU FFN. Replace pre-norm with post-norm. Train each variant. Which change matters most?
2. **Residual scaling.** Try `init` with and without the `1/sqrt(2n)` residual scaling. Does it help with a 12-block model? With a 4-block model?
3. **FFN dimension.** Set `ffn_hidden = {2, 4, 8} × d_model`. At what point does increasing FFN size stop helping?
4. **Gradient norms.** Add a hook that records `||∂L/∂x||` at the input of every block during training. Plot them. In a pre-norm model, the *spread* (max/min ratio) should stay smaller than in a post-norm model as depth grows.
5. **Skip the FFN.** Replace the SwiGLU FFN with the identity. The model becomes "attention only." How much does the loss increase on Tiny Shakespeare?
6. **Skip the attention.** Replace the MHA with the identity. The model becomes a deep feed-forward network. How does the loss compare?
7. **Deeper vs wider.** Train a 12-block, 256-dim model and a 6-block, 384-dim model with the same parameter count. Which generalizes better?

The full implementation lives in `code/chapter07/transformer_block.py` — it's the complete block + a 6-layer mini-transformer you can train end-to-end on Tiny Shakespeare.