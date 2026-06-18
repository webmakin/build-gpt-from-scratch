# Building GPT From Scratch

## Chapter 2

# Mathematics of Language Models

> *"The unreasonable effectiveness of neural networks comes down to one operation: multiply, add, repeat."*

---

## 2.1 Why Math Matters

You could treat PyTorch as a black box and build a working GPT by copying code. But when your loss explodes, your gradients vanish, or your attention patterns collapse, you'll be stuck staring at tensor shapes with no idea what went wrong.

Every operation in a transformer — attention, normalization, feed-forward — is a handful of linear algebra operations chained together. Once you see the math, the architecture becomes obvious.

This chapter builds the mathematical vocabulary you'll need, with PyTorch code alongside every concept. We won't prove theorems. We'll compute.

---

## 2.2 Scalars, Vectors, Matrices, Tensors

A **scalar** is a single number.

```python
import torch

temperature: float = 0.7          # Python float
temperature_t = torch.tensor(0.7) # 0-d tensor (scalar)

print(temperature_t.shape)  # torch.Size([])
```

A **vector** is a 1D list of numbers. In deep learning, a vector typically represents a single data point: a word embedding, a hidden state, or a bias term.

```python
# A token embedding of size 4
embedding = torch.tensor([0.2, -0.5, 1.3, 0.0])
print(embedding.shape)  # torch.Size([4])
```

A **matrix** is a 2D grid. Rows are often samples; columns are features. The attention scores in a transformer are a matrix. The weight matrices in every linear layer are matrices.

```python
# 3 tokens, each with a 4-dimensional embedding
sequence = torch.tensor([
    [0.2, -0.5, 1.3, 0.0],   # token 1
    [0.8,  0.1, 0.1, 0.9],   # token 2
    [0.0,  0.0, 0.5, 0.5],   # token 3
])
print(sequence.shape)  # torch.Size([3, 4])
```

A **tensor** generalizes to 3+ dimensions. In a transformer, a batch of sequences is a 3D tensor: `[batch, seq_len, d_model]`.

```python
# Batch of 2 sequences, each with 3 tokens, embedding dim 4
batch = torch.randn(2, 3, 4)
print(batch.shape)  # torch.Size([2, 3, 4])
```

### Shape conventions in this book

| Symbol      | Meaning                  | Typical size    |
|-------------|--------------------------|-----------------|
| B           | Batch size               | 8–64            |
| T           | Sequence length (tokens) | 128–2048        |
| C / d_model | Embedding dimension      | 128–768         |
| V           | Vocabulary size           | 4,096–50,000    |
| H           | Number of attention heads| 4–12            |
| d_head      | Dimension per head        | d_model / H     |

---

## 2.3 Dot Product — The Engine of Attention

The dot product measures how much two vectors "agree." It multiplies corresponding elements and sums them.

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

dot = (a * b).sum()           # element-wise multiply, then sum
print(dot)                    # tensor(32.)
# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Equivalent:
dot2 = torch.dot(a, b)
print(dot2)                   # tensor(32.)
```

**Why this matters for transformers:** Self-attention computes a dot product between every query vector and every key vector. If two token embeddings point in similar directions, their dot product is large — meaning those tokens "attend" to each other.

---

## 2.4 Matrix Multiplication — Attention at Scale

Matrix multiplication is the workhorse. `C = A @ B` computes every dot product between rows of A and columns of B.

```python
A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])
B = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0],
])

C = A @ B
print(C)
# tensor([[19., 22.],
#         [43., 50.]])
# C[0][0] = 1*5 + 2*7 = 19
# C[0][1] = 1*6 + 2*8 = 22
# C[1][0] = 3*5 + 4*7 = 43
# C[1][1] = 3*6 + 4*8 = 50
```

### The attention formula in matrix form

Self-attention computes:

```
scores = Q @ K^T        # [T, T] — every query dot every key
weights = softmax(scores / sqrt(d_head))  # normalize
output  = weights @ V   # weighted sum of values
```

Let's build a minimal version:

```python
d_head = 4
T = 3  # 3 tokens

Q = torch.randn(T, d_head)  # queries
K = torch.randn(T, d_head)  # keys
V = torch.randn(T, d_head)  # values

scores = Q @ K.T            # [3, 3] — raw attention scores
print("Scores shape:", scores.shape)
print("Scores:\n", scores)
```

Every step in a transformer reduces to `@`, `+`, and a few element-wise operations.

---

## 2.5 Softmax — Converting Scores to Probabilities

Raw dot products can be any real number. Softmax maps them to probabilities that sum to 1, amplifying the largest values.

```python
def softmax(x, dim=-1):
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

logits = torch.tensor([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)
# tensor([0.6590, 0.2424, 0.0986])
print(probs.sum())  # tensor(1.0000)
```

After softmax, the attention scores become a proper probability distribution: each token's attention weights across all other tokens sum to 1.

```python
attn_weights = softmax(scores / (d_head ** 0.5), dim=-1)
output = attn_weights @ V  # [3, 4]

print("Attention weights sum per row:", attn_weights.sum(dim=-1))
# tensor([1.0000, 1.0000, 1.0000])
```

---

## 2.6 Gradients — How Models Learn

A neural network is a function `f(x, W)` where `W` are the parameters. Training means adjusting `W` to minimize a loss `L`.

The **gradient** `∂L/∂W` tells us which direction to move each parameter to reduce the loss.

```python
# A tiny "model": predict y from x with one weight
x = torch.tensor([2.0])
w = torch.tensor([3.0], requires_grad=True)  # track gradients

y_pred = x * w                    # forward pass
y_true = torch.tensor([5.0])
loss = (y_pred - y_true) ** 2     # MSE: (6-5)^2 = 1

loss.backward()                    # compute gradients
print(f"∂L/∂w = {w.grad}")        # ∂L/∂w = 2*(x*w - y)*x = 2*(6-5)*2 = 4
# tensor([4.])

# Move w in the opposite direction of the gradient
learning_rate = 0.1
with torch.no_grad():
    w -= learning_rate * w.grad
    w.grad.zero_()  # reset for next iteration

print(f"Updated w: {w.item():.2f}")  # 3.0 - 0.1*4 = 2.6
```

Every weight in GPT — 124 million of them in GPT-2 small — is updated this way, billions of times during training.

---

## 2.7 Broadcasting — When Shapes Don't Match

PyTorch automatically expands dimensions when shapes "mostly" align. This is called broadcasting and it's everywhere in transformer code.

```python
# Add a bias vector to every row of a matrix
matrix = torch.ones(3, 4)        # [3, 4]
bias   = torch.tensor([1, 2, 3, 4])  # [4]

result = matrix + bias            # bias broadcast to [3, 4]
print(result)
# tensor([[2., 3., 4., 5.],
#         [2., 3., 4., 5.],
#         [2., 3., 4., 5.]])
```

The rules: align dimensions from the right. If one is 1 (or missing), it's stretched to match. If neither is 1 and they don't match, it's an error.

```python
# Causal mask: broadcast [1, 1, T, T] mask over [B, H, T, T] scores
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()  # [3, 3]
mask = mask.unsqueeze(0).unsqueeze(0)                    # [1, 1, 3, 3]

scores = torch.randn(2, 4, 3, 3)  # [batch=2, heads=4, seq=3, seq=3]
scores = scores.masked_fill(mask, float('-inf'))
print("Mask shape:", mask.shape, "→ broadcast to", scores.shape)
```

---

## 2.8 Putting It All Together — A Minimal Attention Block

Everything we've covered combined into one function:

```python
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, heads, seq_len, d_head]
    Returns: [batch, heads, seq_len, d_head]
    """
    d_head = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return weights @ V


# Test it
B, H, T, d = 2, 4, 8, 16  # batch, heads, tokens, dim
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)
V = torch.randn(B, H, T, d)

output = scaled_dot_product_attention(Q, K, V)
print(output.shape)  # torch.Size([2, 4, 8, 16])
```

This function — 5 lines of math — is the heart of every GPT model.

---

## Chapter Summary

- **Scalars, vectors, matrices, tensors** are the data structures of deep learning. Every tensor has a shape; knowing the shape at each step is half the battle.
- **Dot products** measure similarity between vectors — they drive attention.
- **Matrix multiplication** (`@`) scales dot products to entire sequences at once.
- **Softmax** converts raw scores into a probability distribution.
- **Gradients** (`backward()`) tell us how to update parameters to reduce loss.
- **Broadcasting** lets shapes that almost match work together without explicit loops.

In Chapter 3, we'll use these operations to build a Byte Pair Encoding tokenizer, converting raw text into the integer token IDs that feed into our model.

---

## Exercises

1. Write a function `cosine_similarity(a, b)` that computes `(a·b) / (||a|| * ||b||)`. Use it to find which two token embeddings are most similar in a randomly initialized embedding matrix.
2. Implement a small linear regression model `y = x @ W + b` with gradient descent. Train it on synthetic data and plot the loss curve.
3. Compute the attention weights for a sequence of 4 tokens. Before softmax, manually verify that the diagonal entries (self-attention scores) are not necessarily the largest. Why might a token attend more to another token than to itself?
4. The scaling factor `1/sqrt(d_head)` in attention prevents the softmax from saturating. Write a script that computes softmax on random vectors of size 16, 64, 256, and 1024 with and without scaling. What happens to the distribution as dimension grows?
