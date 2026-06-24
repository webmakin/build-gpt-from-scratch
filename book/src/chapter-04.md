# Building GPT From Scratch

## Chapter 4

# Embeddings

> *"An integer on its own is meaningless. An integer that points to a row in a learned table is a coordinate in meaning-space."*

---

## 4.1 The Problem With Token IDs

In Chapter 3 we turned text into integers. A sentence like:

```
"The cat sat"  →  [1014, 2338, 7731]
```

Each integer is just an *index*. The model sees `1014` and has no idea it means "The." It sees `7731` and has no idea it means "sat."

Treating token IDs as raw numbers is a dead end. The model could try to learn that "IDs near each other are similar words" — but they're not. In GPT-2's vocabulary, `1014` ("The") and `7731` ("sat") are neighbors, but `7732` is "Ġsat" (a space followed by "sat") and `7730` is "ĠSat" (a space + capitalized). The integer order carries no semantic meaning.

We need a way to give every token its own vector — a list of numbers that *describes* it — and to make those vectors **learnable**. Two similar words should have similar vectors. Two unrelated words should have vectors that point in different directions.

That vector is called an **embedding**.

---

## 4.2 From One-Hot to Dense

The simplest idea: represent each token as a one-hot vector of vocabulary length, then multiply by a matrix.

```
Vocabulary size V = 50,000
Embedding dimension d = 256

Token "cat" (id = 2338) →
    [0, 0, ..., 0, 1, 0, ..., 0]    (one-hot, length 50,000)
                                 ↑
                                 position 2338

Multiply by embedding matrix E (shape [V, d]) →
    row 2338 of E                   (length 256)
```

That works, but it's wasteful. Multiplying a 50,000-element one-hot vector by a 50,000×256 matrix just to pick out one row is a lot of arithmetic for a lookup. In practice, **we skip the multiplication and do a direct index** — same result, much faster.

```python
import torch

# Embedding table: V rows, d columns
E = torch.randn(50000, 256)   # initialized randomly

# "Lookup" — equivalent to one-hot @ E
one_hot = torch.zeros(50000)
one_hot[2338] = 1.0
result_slow = one_hot @ E            # [256]

result_fast = E[2338]                # [256]   ← this is the lookup
print(torch.allclose(result_slow, result_fast))  # True
```

The fast version is what every modern deep learning framework uses internally.

---

## 4.3 The Embedding Table Is a Lookup

The embedding matrix is a rectangular table with one row per token. To embed a token, look up its row. The whole process is a **table lookup** — nothing more.

```python
embedding = torch.nn.Embedding(
    num_embeddings=50000,   # vocabulary size
    embedding_dim=256,      # vector size per token
)

# Look up tokens by ID
token_ids = torch.tensor([1014, 2338, 7731])  # ["The", "cat", "sat"]
vectors = embedding(token_ids)

print(vectors.shape)   # torch.Size([3, 256])
print(vectors[0, :5])  # first 5 dims of "The"
# tensor([ 0.0234, -0.1102,  0.3341,  0.0892, -0.2210], grad_fn=<SliceBackward0>)
```

That's it. The "embedding" for a token is just a row from a learnable matrix.

Why is this useful? Because the table is **trainable**. During training, the model updates every row of this table using backpropagation. Tokens that appear in similar contexts end up with similar rows. The first few dimensions might learn to encode "is this a verb?", the next few "is it about animals?", and so on — without us telling the model what to look for.

---

## 4.4 What Do the Dimensions Mean?

Here's the catch: by themselves, the dimensions of a learned embedding **don't have a name**. They're whatever the model found useful. After training, dim 17 might be "degree of abstraction" and dim 89 might be "is the word capitalized" — but only the model knows.

We can probe them. Take a trained embedding matrix and find the dimension that best correlates with, say, "word length" or "is it a verb." The answer is often surprising — there's a famous result that the direction `king − man + woman ≈ queen` works in some embeddings, because the model learned a "gender" axis.

But for a *Small Language Model* trained from scratch on a small corpus, the dimensions won't be that clean. They'll be a mess of overlapping signals. That's fine — the model uses all 256 of them together as a fingerprint.

```python
# Inspect: which dimension has the largest variance across the vocab?
variances = embedding.weight.var(dim=0)
print("Top-5 highest-variance dims:", variances.topk(5).indices)
print("Top-5 lowest-variance dims: ", variances.topk(5, largest=False).indices)
```

High-variance dimensions are the ones the model relies on most.

---

## 4.5 Embeddings Inside a Transformer

In a GPT model, embeddings appear in three places:

1. **Token embedding** — turns the integer ID into a vector. `E[token_id]`
2. **Positional embedding** — adds position information (Section 4.6)
3. **Output projection** — turns the final hidden state back into a probability distribution over the vocabulary (this is the "language model head" — covered in Chapter 8)

For now, focus on the token embedding. In code:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        # ids: [batch, seq_len]
        # returns: [batch, seq_len, d_model]
        return self.table(ids)


# Test
embed = TokenEmbedding(vocab_size=50000, d_model=256)
ids = torch.tensor([[1014, 2338, 7731]])   # one sequence of 3 tokens
out = embed(ids)
print(out.shape)  # torch.Size([1, 3, 256])
```

The input to a GPT is **always** a sequence of token IDs. The first operation is always an embedding lookup. Everything that follows — attention, feed-forward, layer norm — operates on these vectors.

---

## 4.6 Positional Information — A Missing Ingredient

Here's a subtle problem. Consider these two sentences:

```
"The cat ate the fish."
"The fish ate the cat."
```

Token IDs:
```
[1014, 2338, 7731, 1014, 8421, 13]
[1014, 8421, 7731, 1014, 2338, 13]
```

The token sets are the same. The **order** is what makes the meaning different. But our embedding lookup `E[token_id]` produces the *same* vector for "cat" in both positions. The model has no way to know which "cat" came first.

**Self-attention** (the next chapter) is also order-blind by default. A bag of vectors, no matter how clever the attention, cannot tell "cat at position 2" from "cat at position 5."

We need to inject **positional information** into the embeddings. The standard solution: add a learned vector for each position.

```python
class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)

    def forward(self, ids):
        # ids: [batch, seq_len]
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)         # [0, 1, ..., T-1]
        tok_vec = self.token(ids)                        # [B, T, d_model]
        pos_vec = self.position(pos)                     # [T, d_model]
        return tok_vec + pos_vec                         # [B, T, d_model]


# Test
embed = TokenAndPositionalEmbedding(vocab_size=50000, d_model=256, max_seq_len=512)
ids = torch.tensor([[1014, 2338, 7731]])
out = embed(ids)
print(out.shape)  # torch.Size([1, 3, 256])
```

The model now has a unique "coordinate" for `(token, position)` pairs. "Cat at position 2" gets a different vector than "cat at position 5" — even though the token embedding is the same, the position embedding added on top is different.

---

## 4.7 Sinusoidal Positional Encodings (the Original)

The original Transformer paper ("Attention Is All You Need", 2017) used *fixed* sinusoidal positions instead of learned ones:

\\[
\text{PE}(pos, 2i)   = \sin\!\left(\frac{pos}{10000^{2i / d_\text{model}}}\right)
\\]

\\[
\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i / d_\text{model}}}\right)
\\]


Why? Two reasons that turned out to be partially wrong but interesting:

1. **Generalization to longer sequences.** Sinusoidal positions extend naturally to any sequence length; learned positions cap out at whatever `max_seq_len` you trained with.
2. **Relative position reasoning.** Because `sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`, the dot product `PE(pos) · PE(pos+k)` depends only on `k`, not on `pos`. The model can learn "attend to the token 5 back" without knowing the absolute position.

Modern LLMs (Llama, Mistral, Qwen) use a different approach: **Rotary Position Embeddings (RoPE)**, which we'll cover in Chapter 5 or 7. For now, the key point: *some* positional information must be added, and the choice of scheme matters.

For your SLM, **learned positional embeddings are fine**. They're simpler, work well at small scale, and are what the original GPT-2 used.

```python
import math

def sinusoidal_position_encoding(max_seq_len, d_model):
    """The original Transformer position encoding."""
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


pe = sinusoidal_position_encoding(max_seq_len=512, d_model=256)
print(pe.shape)            # torch.Size([512, 256])
print(pe[0, :4])           # position 0:  sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1
# tensor([0., 1., 0., 1.])
print(pe[100, :4])         # position 100:  sin/cos of various frequencies
```

---

## 4.8 Embedding Scale — A Small but Real Trick

GPT-2 scales token embeddings by `sqrt(d_model)` before adding positional encodings:

```python
self.token = nn.Embedding(vocab_size, d_model)

def forward(self, ids):
    tok_vec = self.token(ids) * math.sqrt(self.d_model)
    pos_vec = self.position(torch.arange(T))
    return tok_vec + pos_vec
```

Why? Token embeddings are initialized with small variance (default `std=1`), while learned positional embeddings are also small. Multiplying by `sqrt(d_model)` (e.g. `sqrt(768) ≈ 27.7`) makes the token signal dominate early in training, which empirically helps convergence.

In Llama and other newer models this trick isn't used, but it costs nothing and is worth knowing.

---

## 4.9 Tying Input and Output Embeddings

The output layer of a language model is also a matrix: it projects the final hidden state `[B, T, d_model]` into a probability distribution over the vocabulary `[B, T, V]`. That's another `[V, d_model]` matrix.

If you tie the two matrices together — use the **same** matrix for input embedding and output projection — you save `V × d_model` parameters (often 30% of the total!) and you sometimes get slightly better generalization. The intuition: a token's input vector and its output probability distribution should be "compatible" representations of the same concept.

```python
class TiedOutputProjection(nn.Module):
    """Reuse the embedding table as the output projection."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # No separate output matrix — we transpose the embedding in forward()

    def forward(self, hidden_states):
        # hidden_states: [B, T, d_model]
        # output: [B, T, vocab_size]
        return hidden_states @ self.embedding.weight.T
```

Tying is used by GPT-2, Llama, and most production models. The math works because both matrices are `[V, d_model]`, just transposed for the output.

---

## 4.10 Visualizing Embeddings

After training, you can plot embeddings with **t-SNE** or **UMAP** to see what the model learned. Tokens that ended up close in the embedding space should be semantically related.

```python
# Quick t-SNE of an embedding matrix (requires scikit-learn)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# Take a subset of vocab (the most common tokens)
ids_to_plot = list(range(2000))
vectors = embedding.weight.detach()[ids_to_plot].numpy()
coords = tsne.fit_transform(vectors)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.6)
plt.title("Token embeddings (t-SNE projection)")
plt.show()
```

If training worked, you should see clusters of similar words — verbs together, nouns together, names together, etc. The fact that this structure emerges *without anyone telling the model what categories to use* is one of the most striking results in modern NLP.

---

## Chapter Summary

- An **embedding** is a learned vector for a discrete token, stored as a row in a table.
- `nn.Embedding(vocab_size, d_model)` is a lookup table; `embedding(ids)` is just fancy indexing.
- Token embeddings alone are **order-blind** — "cat at position 2" and "cat at position 5" look identical to the model.
- We add **positional information** by adding a position embedding to each token embedding.
- The original Transformer used **sinusoidal** positions; GPT-2 used **learned** positions; modern LLMs use **RoPE** (covered later).
- **Weight tying** between input embedding and output projection saves parameters and often improves generalization.

In Chapter 5, we use these embeddings as the input to **self-attention** — the operation that lets every token look at every other token and decide what's relevant.

---

## Exercises

1. **Lookup vs. one-hot.** Time `embedding(token_id)` for `token_id ∈ [0, 50000)` versus `(one_hot_vec @ embedding.weight)`. The first should be **O(1)**, the second **O(V × d)**. Confirm with `timeit`.
2. **Embedding arithmetic.** Train a tiny embedding on a sentiment dataset. Compute `vec("good") - vec("bad")` and `vec("happy") - vec("sad")`. Is the angle between them small (suggesting a common "positivity" axis)?
3. **Sinusoidal vs learned.** Train a small model with each. Does one converge faster on Tiny Shakespeare? (Hint: use the same hyperparameters, swap only the position encoding.)
4. **Visualize.** Run t-SNE on the embedding from `code/chapter04/embeddings.py` after training. Paste the result in an issue. Are there visible clusters?
5. **Weight tying.** Add weight tying to a tiny model and compare parameter count and validation loss against the untied version. Is the savings worth it?
6. **Dimension scaling.** Set `d_model ∈ {32, 64, 128, 256, 512}` and train each. Plot validation loss vs. `d_model`. Where do you see diminishing returns?

The full implementation lives in `code/chapter04/embeddings.py` — it trains a small embedding on synthetic data and visualizes the result. Run it, modify it, see what the model learns.