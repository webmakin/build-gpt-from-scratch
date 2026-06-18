# Contributing to Build GPT From Scratch

Thanks for your interest in contributing. This project is an educational guide — every line of code is meant to be read and understood, not just executed.

## Ways to Contribute

- **Fix an error.** If a math derivation, code example, or explanation is wrong, open an issue or PR.
- **Improve an explanation.** If something is confusing, suggest a clearer version.
- **Add an exercise.** Each chapter ends with exercises — new ones are always welcome.
- **Translate a chapter.** Making this material accessible in other languages is hugely valuable.
- **Benchmark an implementation.** Performance numbers (speed, memory, FLOPs) with different batch sizes / sequence lengths add real value.
- **Add a diagram.** Visual explanations of attention, backprop, or tensor shapes help everyone.

## Ground Rules

- **Code is for reading first.** Variable names should be descriptive, not clever. Comments explain *why*, not *what*.
- **Keep it self-contained.** Minimize external dependencies. The goal is to understand every line, not to wrap a library.
- **One chapter per PR.** Makes review manageable.
- **Run the code.** Every code block in a chapter should also exist as a runnable `.py` file in `code/chapterNN/`.

## Setup

```bash
git clone git@github.com:webmakin/build-gpt-from-scratch.git
cd build-gpt-from-scratch
pip install -r requirements.txt
```

## Chapter Structure

Each chapter lives in two places:
- `chapters/chap-N.md` — the prose, math, and inline code
- `code/chapterNN/` — runnable standalone scripts

When you add or modify code in a chapter, keep both in sync.

## Pull Request Checklist

- [ ] Code runs without errors (`python code/chapterNN/filename.py`)
- [ ] Chapter markdown and code file are in sync
- [ ] No new dependencies unless discussed in an issue first
- [ ] Status table in `STATUS.md` is updated if a chapter reaches a new milestone
