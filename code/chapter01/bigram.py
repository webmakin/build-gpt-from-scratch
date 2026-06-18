"""
Chapter 1: What Is a Language Model? — Bigram Model

A minimal statistical language model that predicts the next word
based only on the immediately preceding word (bigram counts).

Run: python code/chapter01/bigram.py
"""

from collections import defaultdict
import random


# ── Training data ──────────────────────────────────────────────

text = """
I like pizza
I like pasta
I like coffee
You like pizza
"""

words = text.lower().split()


# ── Build bigram frequency table ───────────────────────────────

def build_bigrams(tokens, n_context=1):
    """Count how often each token follows a given context of n_context tokens."""
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - n_context):
        context = tuple(tokens[i : i + n_context])
        next_token = tokens[i + n_context]
        model[context][next_token] += 1
    return model


bigrams = build_bigrams(words, n_context=1)

print("Bigram counts for 'like':", dict(bigrams[("like",)]))
# → {'pizza': 2, 'pasta': 1, 'coffee': 1}


# ── Convert counts to probabilities ────────────────────────────

def to_probabilities(counts):
    """Convert a {token: count} dict into {token: probability}."""
    total = sum(counts.values())
    return {token: count / total for token, count in counts.items()}


def predict_next(context, model):
    """Given a context tuple, return probability distribution over next tokens."""
    return to_probabilities(model[context])


probs = predict_next(("like",), bigrams)
print("Probabilities after 'like':", probs)
# → {'pizza': 0.5, 'pasta': 0.25, 'coffee': 0.25}


# ── Sample from the distribution (Exercise 3) ──────────────────

def sample_next(context, model):
    """Sample the next token according to the predicted probabilities."""
    counts = model[context]
    tokens = list(counts.keys())
    weights = list(counts.values())
    return random.choices(tokens, weights=weights, k=1)[0]


# ── Generate a short sequence (Exercise 3 extension) ────────────

def generate(start_word, model, length=5):
    """Generate text by repeatedly sampling the next word."""
    tokens = [start_word]
    for _ in range(length):
        context = (tokens[-1],)
        if context not in model:
            break
        tokens.append(sample_next(context, model))
    return " ".join(tokens)


print("\nGenerated:", generate("i", bigrams, length=4))


# ── Extend to trigrams (Exercise 2) ────────────────────────────

trigrams = build_bigrams(words, n_context=2)
print("\nTrigram counts for ('i', 'like'):", dict(trigrams[("i", "like")]))

print("Trigram probabilities:", predict_next(("i", "like"), trigrams))
