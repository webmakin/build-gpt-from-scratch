"""
Chapter 4: Embeddings — Lookup Tables and Learned Vectors

Covers:
  4.2  One-hot vs direct lookup
  4.3  nn.Embedding as a table lookup
  4.6  Token + positional embedding module
  4.7  Sinusoidal position encoding (the original Transformer)
  4.8  Embedding scale by sqrt(d_model)
  4.9  Tied input/output embeddings
  4.10 t-SNE visualization of trained embeddings

Run: python code/chapter04/embeddings.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 4.2 One-hot lookup vs direct index ──────────────────────────

print("=" * 60)
print("4.2 — One-hot lookup vs direct index")
print("=" * 60)

V, d = 50000, 256
E = torch.randn(V, d)         # pretend this is the embedding table

# Slow way: one-hot @ E
one_hot = torch.zeros(V)
one_hot[2338] = 1.0
result_slow = one_hot @ E

# Fast way: row lookup
result_fast = E[2338]

print(f"Slow result shape: {result_slow.shape}")
print(f"Fast result shape: {result_fast.shape}")
print(f"Equal: {torch.allclose(result_slow, result_fast)}")


# ── 4.3 nn.Embedding is a lookup table ─────────────────────────

print("\n" + "=" * 60)
print("4.3 — nn.Embedding as a lookup table")
print("=" * 60)

embed = nn.Embedding(num_embeddings=V, embedding_dim=d)
token_ids = torch.tensor([1014, 2338, 7731])
vectors = embed(token_ids)
print(f"Input ids:  {token_ids.shape}  (3 tokens)")
print(f"Output vec: {vectors.shape}  (3 tokens × 256 dims)")
print(f"First 5 dims of id 1014: {vectors[0, :5].detach()}")

# Variance across the vocabulary — high-variance dims are most "useful"
variances = embed.weight.var(dim=0)
top5 = variances.topk(5).indices
bot5 = variances.topk(5, largest=False).indices
print(f"Highest-variance dims: {top5.tolist()}")
print(f"Lowest-variance dims:  {bot5.tolist()}")


# ── 4.6 Token + positional embedding module ────────────────────

print("\n" + "=" * 60)
print("4.6 — TokenAndPositionalEmbedding")
print("=" * 60)


class TokenAndPositionalEmbedding(nn.Module):
    """Token embedding + learned position embedding (GPT-2 style)."""

    def __init__(self, vocab_size, d_model, max_seq_len=512, scale_by_sqrt_d=False):
        super().__init__()
        self.d_model = d_model
        self.scale_by_sqrt_d = scale_by_sqrt_d
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)

    def forward(self, ids):
        # ids: [batch, seq_len]
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)            # [0, 1, ..., T-1]
        tok_vec = self.token(ids)                           # [B, T, d]
        if self.scale_by_sqrt_d:
            tok_vec = tok_vec * math.sqrt(self.d_model)     # GPT-2 trick
        pos_vec = self.position(pos).unsqueeze(0)           # [1, T, d]
        return tok_vec + pos_vec                            # [B, T, d]


# Test
emb_module = TokenAndPositionalEmbedding(
    vocab_size=V, d_model=d, max_seq_len=512, scale_by_sqrt_d=True
)
ids = torch.tensor([[1014, 2338, 7731, 1014, 8421]])   # batch=1, seq=5
out = emb_module(ids)
print(f"Input:  {ids.shape}  (batch=1, seq=5)")
print(f"Output: {out.shape}  (batch=1, seq=5, d_model=256)")

# Verify: output == token_embed + pos_embed (broadcast over batch)
with torch.no_grad():
    pos = torch.arange(ids.shape[1])
    scale = math.sqrt(emb_module.d_model) if emb_module.scale_by_sqrt_d else 1.0
    expected = emb_module.token(ids) * scale + emb_module.position(pos).unsqueeze(0)
print(f"Output matches lookup: {torch.allclose(out, expected, atol=1e-4)}")

# Verify: same token at different positions gets different output
print(f"'The' at pos 0 vs pos 3 differ: "
      f"{not torch.allclose(out[0, 0], out[0, 3])}")


# ── 4.7 Sinusoidal position encoding ───────────────────────────

print("\n" + "=" * 60)
print("4.7 — Sinusoidal position encoding")
print("=" * 60)


def sinusoidal_position_encoding(max_seq_len, d_model):
    """Original Transformer position encoding (sin/cos at varying frequencies)."""
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


pe = sinusoidal_position_encoding(max_seq_len=512, d_model=256)
print(f"Shape: {pe.shape}  (512 positions × 256 dims)")
print(f"Position 0:  {pe[0, :8].tolist()}")
print(f"Position 1:  {pe[1, :8].tolist()}")
print(f"Position 100: {pe[100, :8].tolist()}")

# A nice property: dot product of two position vectors depends on
# their distance, not their absolute position (within the same sin/cos band)
def positional_dot(distance, d_model):
    """If PE(p) · PE(p+k) depended only on k, this would be constant across p."""
    pe = sinusoidal_position_encoding(max_seq_len=2000, d_model=d_model)
    p, k = 100, distance
    return (pe[p] * pe[p + k]).sum().item()


print(f"\nPositional dot product (d=256) at distance k=10:")
for p in [10, 100, 500, 1000, 1500]:
    print(f"  PE({p}) · PE({p + 10}) = {positional_dot(10, 256):.4f}")


# ── 4.9 Tied input/output projection ───────────────────────────

print("\n" + "=" * 60)
print("4.9 — Weight tying (input embed = output projection)")
print("=" * 60)


class TiedLMHead(nn.Module):
    """Reuse the embedding table as the output projection."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)

    def forward(self, hidden_states):
        # hidden_states: [B, T, d_model]
        # logits: [B, T, vocab_size]
        return hidden_states @ self.token.weight.T


tied = TiedLMHead(vocab_size=V, d_model=d)
n_tied = sum(p.numel() for p in tied.parameters())

untied = nn.Sequential(
    nn.Linear(d, V, bias=False)   # separate output projection
)
n_untied = sum(p.numel() for p in untied.parameters())

print(f"Tied parameters:    {n_tied:>12,}")
print(f"Untied parameters:  {n_tied + n_untied:>12,}  "
      f"(embedding + separate output projection)")
print(f"Savings:            {n_untied:>12,}  "
      f"({100 * n_untied / (n_tied + n_untied):.1f}% of total)")

hidden = torch.randn(2, 16, d)
logits = tied(hidden)
print(f"\nHidden states: {hidden.shape}  →  Logits: {logits.shape}")


# ── 4.10 Train embeddings on a tiny task ───────────────────────

print("\n" + "=" * 60)
print("4.10 — Training embeddings on a synthetic bigram task")
print("=" * 60)

# A small synthetic corpus with clusters: animals, verbs, foods.
# The model should learn that "cat" and "dog" have similar vectors.
corpus = """
the cat sat on the mat
the dog sat on the mat
the cat ate the fish
the dog ate the fish
the cat chased the mouse
the dog chased the mouse
the cat drank the milk
the dog drank the milk
the cat slept on the mat
the dog slept on the mat
the boy ate the pizza
the boy ate the burger
the boy ate the pasta
the boy ate the sushi
the boy drank the milk
the boy drank the juice
the boy drank the water
the boy drank the coffee
""".lower().split()

# Build vocab
vocab = sorted(set(corpus))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
V_small = len(vocab)
print(f"Vocab size: {V_small}  →  {vocab[:5]}...")

# Build training pairs (input, target) — predict next word
pairs = [(stoi[w1], stoi[w2]) for w1, w2 in zip(corpus, corpus[1:])]
xs = torch.tensor([p[0] for p in pairs])
ys = torch.tensor([p[1] for p in pairs])

# Tiny model: embedding → hidden → output projection.
# The hidden layer lets the embedding be the *only* thing learning about
# token identity (the projection can also be learned independently).
torch.manual_seed(0)
d_small = 16
emb = nn.Embedding(V_small, d_small)
out_proj = nn.Linear(d_small, V_small, bias=False)
params = list(emb.parameters()) + list(out_proj.parameters())
opt = torch.optim.Adam(params, lr=0.05)

for step in range(1500):
    h = emb(xs)                          # [N, d]
    logits = out_proj(h)                  # [N, V]
    loss = F.cross_entropy(logits, ys)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (step + 1) % 250 == 0:
        print(f"  step {step+1:4}  loss {loss.item():.3f}")

print(f"\nFinal loss: {loss.item():.3f}  "
      f"(irreducible ≈ {math.log(3):.3f} for 3-way choices)")


# ── 4.10b Inspect the learned embeddings ───────────────────────

print("\n" + "=" * 60)
print("4.10b — Inspecting learned embeddings")
print("=" * 60)

# For each word, find its nearest neighbors by cosine similarity
W = emb.weight.detach()
W_norm = F.normalize(W, dim=1)


def nearest_neighbors(word, k=4):
    if word not in stoi:
        return f"  {word!r} not in vocab"
    i = stoi[word]
    sims = (W_norm @ W_norm[i]).cpu()
    top = sims.topk(k + 1).indices[1:]   # skip self
    return [(itos[j.item()], sims[j].item()) for j in top]


for w in ["cat", "dog", "ate", "milk", "pizza", "mat", "the"]:
    print(f"\nNearest to {w!r}:")
    for neighbor, sim in nearest_neighbors(w):
        print(f"  {neighbor:8}  sim={sim:.3f}")


# ── 4.10c t-SNE visualization (optional, needs sklearn + matplotlib) ──

print("\n" + "=" * 60)
print("4.10c — t-SNE visualization (optional)")
print("=" * 60)

try:
    import sys
    if "--no-tsne" in sys.argv:
        raise ImportError("disabled by --no-tsne flag")

    import matplotlib
    matplotlib.use("Agg")     # non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # PCA down to 2D as a fast initializer (sklearn t-SNE default is slow)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    init = pca.fit_transform(W.numpy())

    tsne = TSNE(n_components=2, perplexity=min(5, V_small - 1),
                random_state=42, init=init, max_iter=300)
    coords = tsne.fit_transform(W.numpy())

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=80, alpha=0.7, c="steelblue")
    for i, word in itos.items():
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.title("Trained token embeddings (t-SNE)")
    plt.tight_layout()
    out_path = "/tmp/embeddings_tsne.png"
    plt.savefig(out_path, dpi=120)
    print(f"t-SNE plot saved to: {out_path}")

except ImportError as e:
    print(f"skipped (install scikit-learn + matplotlib, "
          f"or remove --no-tsne to enable): {e}")


print("\nDone — every operation above appears inside GPT's embedding stack.")