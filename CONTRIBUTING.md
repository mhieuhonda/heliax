# Contributing to Heliax

Heliax is an early, performance-focused library. Contributions should improve correctness, clarity, or measured performance.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
python examples/benchmark.py
```

Please include:

- a focused test for behavior changes;
- a small benchmark or note when changing a numerical kernel;
- clear documentation for public API changes;
- no unrelated dependency or formatting churn.

## Reporting security issues

Do not open a public issue for a suspected vulnerability. Use the repository owner’s private security contact.
