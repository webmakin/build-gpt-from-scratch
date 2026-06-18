"""
Chapter 2: Mathematics of Language Models — Hands-On

Covers: scalars, vectors, matrices, tensors, dot products,
matrix multiplication, softmax, gradients, broadcasting,
and a minimal scaled dot-product attention block.

Run: python code/chapter02/math_foundations.py
"""

import torch
import torch.nn.functional as F


# ── 2.2 Scalars, Vectors, Matrices, Tensors ────────────────────

print("=" * 50)
print("2.2 — Scalars, Vectors, Matrices, Tensors")
print("=" * 50)

scalar = torch.tensor(3.14)
print(f"Scalar: {scalar}, shape: {scalar.shape}")

vector = torch.tensor([0.2, -0.5, 1.3, 0.0])
print(f"Vector: {vector}, shape: {vector.shape}")

matrix = torch.tensor([[0.2, -0.5, 1.3, 0.0],
                       [0.8, 0.1, 0.1, 0.9],
                       [0.0, 0.0, 0.5, 0.5]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")

batch = torch.randn(2, 3, 4)
print(f"Batch tensor shape: {batch.shape}  → [batch, seq_len, d_model]")


# ── 2.3 Dot Product ────────────────────────────────────────────

print("\n" + "=" * 50)
print("2.3 — Dot Product")
print("=" * 50)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

dot_manual = (a * b).sum()
dot_builtin = torch.dot(a, b)
print(f"Manual dot:  {dot_manual}")
print(f"Built-in dot: {dot_builtin}")
print(f"Match: {torch.allclose(dot_manual, dot_builtin)}")


# ── 2.4 Matrix Multiplication ──────────────────────────────────

print("\n" + "=" * 50)
print("2.4 — Matrix Multiplication")
print("=" * 50)

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

C = A @ B
print(f"A @ B:\n{C}")

# Build attention scores: Q @ K^T
d_head = 4
T = 3
Q = torch.randn(T, d_head)
K = torch.randn(T, d_head)
V = torch.randn(T, d_head)

scores = Q @ K.T
print(f"\nQ shape: {Q.shape}")
print(f"K.T shape: {K.T.shape}")
print(f"Scores shape: {scores.shape}  → [T, T]")
print(f"Scores:\n{scores}")


# ── 2.5 Softmax ────────────────────────────────────────────────

print("\n" + "=" * 50)
print("2.5 — Softmax")
print("=" * 50)

def softmax(x, dim=-1):
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

logits = torch.tensor([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probs:  {probs}")
print(f"Sum:    {probs.sum():.4f}")

# Attention weights
attn_weights = softmax(scores / (d_head ** 0.5), dim=-1)
attn_output = attn_weights @ V
print(f"\nAttention weights:\n{attn_weights}")
print(f"Row sums: {attn_weights.sum(dim=-1)}")
print(f"Output shape: {attn_output.shape}")


# ── 2.6 Gradients ──────────────────────────────────────────────

print("\n" + "=" * 50)
print("2.6 — Gradients (one step of SGD)")
print("=" * 50)

x = torch.tensor([2.0])
w = torch.tensor([3.0], requires_grad=True)

y_pred = x * w
y_true = torch.tensor([5.0])
loss = (y_pred - y_true) ** 2

loss.backward()
print(f"Loss: {loss.item():.4f}")
print(f"∂L/∂w: {w.grad.item():.4f}")
# ∂L/∂w = 2*(2*3 - 5)*2 = 2*(6-5)*2 = 4

lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
    w.grad.zero_()

print(f"Updated w: {w.item():.4f}  (was 3.0, moved toward 2.5)")


# ── 2.7 Broadcasting ───────────────────────────────────────────

print("\n" + "=" * 50)
print("2.7 — Broadcasting")
print("=" * 50)

matrix = torch.ones(3, 4)
bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = matrix + bias
print(f"Matrix + bias:\n{result}")

# Causal mask broadcasting
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
scores_batch = torch.randn(2, 4, 3, 3)  # [B=2, H=4, T=3, T=3]
masked = scores_batch.masked_fill(mask, float('-inf'))
print(f"\nCausal mask:\n{mask.squeeze().int()}")
print(f"Broadcast from {list(mask.shape)} → {list(masked.shape)}")


# ── 2.8 Complete Scaled Dot-Product Attention ──────────────────

print("\n" + "=" * 50)
print("2.8 — Scaled Dot-Product Attention")
print("=" * 50)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Minimal attention — the heart of every GPT model."""
    d_head = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V

B, H, T, d = 2, 4, 8, 16
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)
V = torch.randn(B, H, T, d)

output = scaled_dot_product_attention(Q, K, V)
print(f"Input:  Q,K,V each [{B}, {H}, {T}, {d}]")
print(f"Output: {list(output.shape)}")

# With causal mask
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
output_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"Output (causal): {list(output_causal.shape)}")

print("\nDone — every operation above appears inside GPT.")
