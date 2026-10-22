# Contributing

Thanks for your interest.

## Before you contribute

- This project is dual-licensed (academic free / commercial paid). Contributions
  are accepted only under [`CLA.md`](CLA.md), so the author can continue to offer
  both licenses.
- Do not upload third-party audio, datasets, or model weights unless their
  license explicitly allows redistribution and you add attribution.
- Open an issue first for large changes.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
pytest -q
```

- Keep the code dependency-light (NumPy/SciPy only where needed).
- Add a test for behavior changes.
- Do not change the reported numbers without updating `docs/RESULTS.md`.
- Do not add a "SOTA" claim; this is a traditional baseline.

## Pull requests

- One topic per PR.
- Include a short description, test commands, and before/after results.
- By submitting a PR you agree to the CLA.
