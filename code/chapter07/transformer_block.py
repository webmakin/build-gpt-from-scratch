"""
Chapter 7: Transformer Blocks

Covers:
  7.3  LayerNorm (reference implementation)
  7.4  RMSNorm (faster drop-in)
  7.5  SwiGLU FFN
  7.6  TransformerBlock (pre-norm)
  7.7  Full Transformer model with weight tying
  7.8  Parameter count breakdown
  7.9  Gradient flow: pre-norm vs post-norm
  7.10 GPT-2 init with residual scaling

Run: python code/chapter07/transformer_block.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 7.3 LayerNorm (reference) ──────────────────────────────────

print("=" * 60)
print("7.3 — LayerNorm (reference, for understanding)")
print("=" * 60)


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


# Quick equivalence check vs nn.LayerNorm
ref = nn.LayerNorm(64)
mine = LayerNorm(64)
# Copy the same gamma=1, beta=0
mine.gamma.data = ref.weight.data.clone()
mine.beta.data = ref.bias.data.clone()

x = torch.randn(2, 8, 64)
print(f"LayerNorm matches PyTorch: {torch.allclose(ref(x), mine(x), atol=1e-5)}")


# ── 7.4 RMSNorm (the production choice) ────────────────────────

print("\n" + "=" * 60)
print("7.4 — RMSNorm (used in Llama, Mistral, Qwen)")
print("=" * 60)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


rms = RMSNorm(64)
out = rms(x)
print(f"RMSNorm output shape: {out.shape}")
print(f"Output RMS over last dim: {out.pow(2).mean(dim=-1).sqrt().mean():.3f}  "
      f"(should be ~1.0)")


# ── 7.5 SwiGLU FFN ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("7.5 — SwiGLU FFN")
print("=" * 60)


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU activation. Used in Llama, PaLM, Qwen."""

    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # Llama convention: (8/3) × d_model, rounded to multiple of 64
            hidden_dim = int(2 * d_model * 4 / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.hidden_dim = hidden_dim

        self.W_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


ffn = SwiGLU(d_model=64)
n_ffn = sum(p.numel() for p in ffn.parameters())
print(f"SwiGLU hidden_dim: {ffn.hidden_dim}  (Llama rounding of (8/3) × 64 = 170 → 192)")
print(f"SwiGLU parameters: {n_ffn:,}")
print(f"  = 3 × d_model × hidden_dim = 3 × 64 × {ffn.hidden_dim} = {3 * 64 * ffn.hidden_dim:,}")


# ── 7.6 TransformerBlock (pre-norm) ────────────────────────────

print("\n" + "=" * 60)
print("7.6 — TransformerBlock")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """Minimal MHA — same as chapter 6, inlined here for self-containment."""

    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
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
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_head ** 0.5)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.W_o(out))


class TransformerBlock(nn.Module):
    """Pre-norm block: MHA + SwiGLU FFN, each with residual + RMSNorm."""

    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


block = TransformerBlock(d_model=64, num_heads=4, max_seq_len=32)
x = torch.randn(2, 16, 64)
out = block(x)
print(f"Block input:  {x.shape}")
print(f"Block output: {out.shape}  (shape preserved)")
print(f"Block parameters: {sum(p.numel() for p in block.parameters()):,}")


# ── 7.7 Full Transformer model ─────────────────────────────────

print("\n" + "=" * 60)
print("7.7 — Full Transformer model")
print("=" * 60)


class Transformer(nn.Module):
    """Pre-norm causal Transformer LM with weight tying."""

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
        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


model = Transformer(vocab_size=100, d_model=64, num_heads=4, num_blocks=4, max_seq_len=32)
ids = torch.randint(0, 100, (2, 16))
logits = model(ids)
print(f"Input ids:    {ids.shape}")
print(f"Output logits: {logits.shape}  (batch × seq × vocab)")


# ── 7.8 Parameter count breakdown ─────────────────────────────

print("\n" + "=" * 60)
print("7.8 — Parameter count breakdown")
print("=" * 60)

# Our SLM target config
vocab_size, d_model, num_heads, num_blocks, max_seq_len = 8192, 384, 6, 6, 512

slm = Transformer(vocab_size, d_model, num_heads, num_blocks, max_seq_len)
n_total = sum(p.numel() for p in slm.parameters())
n_emb   = slm.token_emb.weight.numel() + slm.pos_emb.weight.numel()
n_blocks_total = sum(p.numel() for p in slm.blocks.parameters())
n_lnf   = slm.ln_f.gamma.numel()
n_tied_head = 0   # shared with token_emb, no extra params

print(f"SLM config: vocab={vocab_size}, d={d_model}, H={num_heads}, "
      f"L={num_blocks}, T={max_seq_len}")
print(f"  Token + position embeddings:  {n_emb:>11,}")
print(f"  {num_blocks} × TransformerBlock: {n_blocks_total:>11,}")
print(f"    (avg per block: {n_blocks_total // num_blocks:,})")
print(f"  Final RMSNorm (γ only):       {n_lnf:>11,}")
print(f"  Output head:                  {n_tied_head:>11,}  (tied to token_emb)")
print(f"  Total unique parameters:      {n_total:>11,}")

# Reference points
print(f"\nFor comparison:")
print(f"  GPT-2 small:    124,000,000")
print(f"  Llama 2 7B:  7,000,000,000")
print(f"  Our SLM:        {n_total:,}  "
      f"({n_total/124e6:.2f}× smaller than GPT-2 small, "
      f"{n_total/7e9:.5f}× smaller than Llama 7B)")


# ── 7.9 Gradient flow: pre-norm vs post-norm ──────────────────

print("\n" + "=" * 60)
print("7.9 — Gradient flow: pre-norm vs post-norm")
print("=" * 60)


class PostNormBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout=0.0)
        self.ffn = SwiGLU(d_model)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


def measure_grad_norms(pre_norm: bool, n_blocks: int, d_model: int = 64,
                       num_heads: int = 4, max_seq_len: int = 32):
    """Build a stack of blocks, do one fwd/bwd, and record gradient norm at
    the input of each block."""
    Block = TransformerBlock if pre_norm else PostNormBlock
    blocks = nn.ModuleList([
        Block(d_model, num_heads, max_seq_len) for _ in range(n_blocks)
    ])

    x = torch.randn(2, 16, d_model, requires_grad=True)
    for blk in blocks:
        x = blk(x)
    x.sum().backward()

    # The input gradient at each block
    norms = []
    for blk in blocks:
        # Get the gradient on the FIRST parameter of the block (input projection W_q)
        if pre_norm:
            g = blk.attn.W_q.weight.grad
        else:
            g = blk.attn.W_q.weight.grad
        norms.append(g.norm().item())
    return norms


print(f"{'depth':>5}  {'pre-norm (W_q grads)':>22}  {'post-norm (W_q grads)':>22}")
print(f"{'-----':>5}  {'----------------------':>22}  {'-----------------------':>22}")
# Note: this experiment shows gradient norms at *initialization*, after a
# single fwd/bwd. The real pre-norm advantage shows up over training —
# post-norm models typically need learning rate warmup to avoid early
# divergence, while pre-norm trains stably with a constant LR.
for n_blocks in [2, 6, 12, 24]:
    pre = measure_grad_norms(pre_norm=True, n_blocks=n_blocks)
    post = measure_grad_norms(pre_norm=False, n_blocks=n_blocks)
    pre_spread = max(pre) / min(pre)
    post_spread = max(post) / min(post)
    print(f"L={n_blocks:>3}  mean={sum(pre)/len(pre):>7.2f}  "
          f"spread={pre_spread:.2f}     "
          f"mean={sum(post)/len(post):>7.2f}  spread={post_spread:.2f}")
print()
print("Both grow with depth, with similar spreads at initialization.")
print("Pre-norm's real training advantage is *over time*: it lets you skip")
print("LR warmup and train deep stacks without divergence. The at-init")
print("spread advantage is small and noisy.")


# ── 7.10 GPT-2 init with residual scaling ─────────────────────

print("\n" + "=" * 60)
print("7.10 — GPT-2 init with residual scaling")
print("=" * 60)


def gpt2_init(module, n_blocks, base_std=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=base_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=base_std)
    elif isinstance(module, TransformerBlock):
        # Scale residual-output projections by 1/sqrt(2 * n_blocks)
        scale = 1.0 / math.sqrt(2 * n_blocks)
        module.attn.W_o.weight.data *= scale
        module.ffn.W_down.weight.data *= scale


slm2 = Transformer(vocab_size, d_model, num_heads, num_blocks, max_seq_len)
slm2.apply(lambda m: gpt2_init(m, n_blocks))

# Verify
w_o_norms = [b.attn.W_o.weight.norm().item() for b in slm2.blocks]
ffn_norms = [b.ffn.W_down.weight.norm().item() for b in slm2.blocks]
print(f"W_o norms after init:        {[f'{n:.2f}' for n in w_o_norms]}")
print(f"FFN W_down norms after init: {[f'{n:.2f}' for n in ffn_norms]}")
print(f"Expected scale: 1/sqrt(2 × {n_blocks}) = {1/math.sqrt(2*n_blocks):.4f}")


# ── 7.11 Train a tiny model end-to-end ────────────────────────

print("\n" + "=" * 60)
print("7.11 — Train a tiny transformer on a tiny task")
print("=" * 60)

# Toy task: learn to reverse short sequences
torch.manual_seed(0)
seq_len = 8
vocab = 16
n_train = 1000

X = torch.randint(1, vocab, (n_train, seq_len))
Y = torch.flip(X, dims=[1])   # reverse each sequence

# Shift Y: predict the *next* token given current and past
# For a reversal task, this is essentially predicting position t = (T-1-t) of input
Y_input  = torch.cat([torch.zeros(n_train, 1, dtype=torch.long), Y[:, :-1]], dim=1)

tiny = Transformer(vocab_size=vocab, d_model=32, num_heads=4, num_blocks=2, max_seq_len=seq_len)
opt = torch.optim.Adam(tiny.parameters(), lr=3e-3)

for step in range(500):
    idx = torch.randint(0, n_train, (32,))
    logits = tiny(X[idx])                            # [B, T, V]
    loss = F.cross_entropy(logits.reshape(-1, vocab), Y[idx].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 100 == 0:
        acc = (logits.argmax(-1) == Y[idx]).float().mean().item()
        print(f"  step {step+1:4}  loss {loss.item():.3f}  acc {acc:.2f}")

# Sample a few predictions
print("\nSample predictions (input → model output):")
tiny.eval()
with torch.no_grad():
    for i in range(3):
        x_in = X[i:i+1]
        pred = tiny(x_in).argmax(-1)[0].tolist()
        truth = Y[i].tolist()
        print(f"  in:  {x_in[0].tolist()}")
        print(f"  pred: {pred}")
        print(f"  true: {truth}\n")


# ── 7.12 Compare RMSNorm vs LayerNorm speed ──────────────────

print("=" * 60)
print("7.12 — RMSNorm vs LayerNorm speed")
print("=" * 60)

x = torch.randn(32, 512, 1024)
ln = nn.LayerNorm(1024).eval()
rn = RMSNorm(1024).eval()
# Match gamma
rn.gamma.data = ln.weight.data.clone()

# Warm up
for _ in range(10):
    _ = ln(x); _ = rn(x)

import time
t0 = time.time()
for _ in range(1000):
    _ = ln(x)
t1 = time.time()
t2 = time.time()
for _ in range(1000):
    _ = rn(x)
t3 = time.time()

print(f"LayerNorm:  {(t1-t0)*1000:6.1f}ms for 1000 calls")
print(f"RMSNorm:    {(t3-t2)*1000:6.1f}ms for 1000 calls")
print()
print("(In production, torch.nn.RMSNorm and FlashAttention-style fused")
print(" kernels give a real 10-30% speedup. The naive implementation")
print(" here is unoptimized — the numbers above measure the math, not")
print(" the optimized kernel.)")


print("\nDone — the transformer block is the unit of every modern LM.")
print("       Stack N of them, add embeddings and a head, and you have GPT.")