"""
Chapter 3: Building a Tokenizer — Byte Pair Encoding from Scratch

Covers: bytes, UTF-8, BPE training, encoding/decoding, regex pre-tokenization
(skipping the GPT-2 regex for clarity), special tokens, and a `tiktoken`
comparison at the end.

Run: python code/chapter03/tokenizer.py
"""

from collections import Counter


# ── 3.3 Bytes and UTF-8 ─────────────────────────────────────────

print("=" * 60)
print("3.3 — Bytes and UTF-8")
print("=" * 60)

for s in ["A", "é", "你", "🙂"]:
    b = s.encode("utf-8")
    print(f"  {s!r:6} → {len(b)} bytes: {list(b)}")


# ── 3.4 CharTokenizer (the dumbest possible tokenizer) ──────────

print("\n" + "=" * 60)
print("3.4 — CharTokenizer")
print("=" * 60)


class CharTokenizer:
    """Vocabulary = every unique character that appeared in the training text."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


char_tok = CharTokenizer("hello world")
print(f"Vocabulary size: {char_tok.vocab_size}")
print(f"Vocabulary:      {char_tok.chars}")

ids = char_tok.encode("hello")
print(f"encode('hello') = {ids}")
print(f"decode(ids)     = {char_tok.decode(ids)!r}")

# Sanity check: round trip
test = "hello world"
assert char_tok.decode(char_tok.encode(test)) == test
print("Round trip: OK")


# ── 3.5–3.6 BPE training ────────────────────────────────────────

print("\n" + "=" * 60)
print("3.5–3.6 — Byte Pair Encoding (BPE)")
print("=" * 60)


def get_pair_counts(ids):
    """Count every adjacent (a, b) pair in a list of token IDs."""
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with `new_id`. Returns new list."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# Tiny corpus (UTF-8 bytes)
text = (
    "the cat sat on the mat. the cat ate the rat. "
    "the rat sat on the cat. the cat and the rat sat on the mat."
)
ids = list(text.encode("utf-8"))

print(f"Initial sequence length: {len(ids)} bytes")
print(f"Initial vocab size:      256 (raw bytes)")

num_merges = 30
merges = {}  # (a, b) -> new_id, ordered by creation time (Python 3.7+)

print(f"\nRunning {num_merges} merges:")
for i in range(num_merges):
    pair_counts = get_pair_counts(ids)
    if not pair_counts:
        break
    pair, count = pair_counts.most_common(1)[0]
    new_id = 256 + i
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    print(f"  Merge {i+1:2}: pair={pair} → new_id={new_id} "
          f"(count={count}, seq_len={len(ids)})")

print(f"\nFinal vocab size: {256 + len(merges)}")
print(f"Final sequence length: {len(ids)} "
      f"(compression: {len(ids)/len(list(text.encode('utf-8'))):.2%})")


# ── 3.7 Encode / decode ─────────────────────────────────────────

print("\n" + "=" * 60)
print("3.7 — Encoding and decoding")
print("=" * 60)

# Re-build inverse vocab: itos[id] = byte sequence for that token
itos = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():
    itos[new_id] = itos[a] + itos[b]


def encode(text, merges):
    """Replay merges in order. Returns list of token IDs."""
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids, itos):
    """Convert IDs back to a UTF-8 string using the byte-mapping vocab."""
    out = bytearray()
    for i in ids:
        out.extend(itos[i])
    return out.decode("utf-8", errors="replace")


for sample in ["the cat", "the rat", "a new sentence!", "🙂 emoji"]:
    enc = encode(sample, merges)
    dec = decode(enc, itos)
    print(f"  {sample!r:30} → {len(enc)} tokens → {dec!r}")
    assert dec == sample, f"Round trip failed: {dec!r} != {sample!r}"

print("All round trips: OK")


# ── 3.9 Special tokens ──────────────────────────────────────────

print("\n" + "=" * 60)
print("3.9 — Special tokens")
print("=" * 60)

# Reserve a few IDs at the top of the vocab for control signals
SPECIAL_BOS = 256 + num_merges
SPECIAL_EOS = 256 + num_merges + 1
SPECIAL_PAD = 256 + num_merges + 2

itos[SPECIAL_BOS] = b"<|bos|>"
itos[SPECIAL_EOS] = b"<|eos|>"
itos[SPECIAL_PAD] = b"<|pad|>"


def encode_with_special(text, merges, bos=False, eos=False):
    """Encode text, optionally wrapping in BOS/EOS tokens."""
    ids = encode(text, merges)
    if bos:
        ids = [SPECIAL_BOS] + ids
    if eos:
        ids = ids + [SPECIAL_EOS]
    return ids


def decode_with_special(ids, itos):
    """Decode, stripping special tokens (they don't survive byte-decode cleanly)."""
    out = bytearray()
    for i in ids:
        if i in itos:
            out.extend(itos[i])
    return out.decode("utf-8", errors="replace")


chat_ids = encode_with_special("hi", merges, bos=True, eos=True)
print(f"Chat encode:  {chat_ids}")
print(f"Chat decode:  {decode_with_special(chat_ids, itos)!r}")
print(f"  BOS id = {SPECIAL_BOS}, EOS id = {SPECIAL_EOS}")


# ── 3.8 Production tokenizers with `tiktoken` ──────────────────

print("\n" + "=" * 60)
print("3.8 — Production tokenizers (tiktoken)")
print("=" * 60)

try:
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4")

    samples = [
        "Hello, world!",
        "你好世界 🙂",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "https://example.com/api/v1/users?id=42",
    ]

    print(f"\nGPT-4 tokenizer (vocab size = {enc.max_token_value + 1}):\n")
    for s in samples:
        ids = enc.encode(s)
        chars = len(s.encode("utf-8"))
        print(f"  {len(ids):3} tokens / {chars:3} bytes  "
              f"(compression: {chars/len(ids):.2f} bytes/token)  {s[:50]!r}")

    # Inspect individual tokens
    print("\nToken-by-token breakdown of 'Hello, world!':")
    for i in enc.encode("Hello, world!"):
        print(f"  {i:6} → {enc.decode([i])!r}")

except ImportError:
    print("\n`tiktoken` not installed — skipping production comparison.")
    print("Install with: pip install tiktoken")


# ── Vocabulary size sweep (Exercise 2) ──────────────────────────

print("\n" + "=" * 60)
print("Exercise 2 — Vocabulary size sweep")
print("=" * 60)

# A more realistic mini-corpus: distinct words so the sweep has something
# to chew on at every merge count.
sweep_corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks at the fox. the fox runs away from the dog. "
    "machine learning is a subset of artificial intelligence. "
    "neural networks learn representations from data. "
    "transformers use self-attention to model sequences. "
    "tokenizers convert text into integer sequences. "
    "byte pair encoding is a simple but powerful algorithm. "
    "the cat sat on the mat and the rat sat on the cat."
)
sweep_test = (
    "the quick brown fox jumps over the lazy dog. "
    "transformers use self-attention to model sequences."
)

print(f"Training corpus: {len(sweep_corpus)} characters "
      f"({len(sweep_corpus.encode('utf-8'))} bytes)")
print(f"Test string:     {len(sweep_test)} characters "
      f"({len(sweep_test.encode('utf-8'))} bytes)")
print(f"\n  merges  |  tokens  |  bytes/token  |  vocab size")
print(f"  -------+----------+---------------+------------")

for n_merges in [0, 50, 100, 250, 500, 1000, 2000]:
    work_ids = list(sweep_corpus.encode("utf-8"))
    work_merges = {}
    for i in range(n_merges):
        counts = get_pair_counts(work_ids)
        if not counts:
            break
        pair = counts.most_common(1)[0][0]
        new_id = 256 + i
        work_ids = merge(work_ids, pair, new_id)
        work_merges[pair] = new_id

    test_enc = encode(sweep_test, work_merges)
    test_bytes = len(sweep_test.encode("utf-8"))
    print(f"  {n_merges:5}  |  {len(test_enc):6}  |  "
          f"{test_bytes/len(test_enc):.2f}         |  "
          f"{256 + len(work_merges)}")


print("\nDone — you now have a working BPE tokenizer.")