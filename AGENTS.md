# Repository Guidelines

## Project Structure & Module Organization
`BenNevis/` contains the Python package:
- `core/` (training, dataset, losses, language utilities)
- `models/` (RNNP, Transformer, Conformer, Whisper, Wav2Vec2)
- `run/` (training/prediction/alignment entry scripts)
- `utils/`, `samplers/`, and `bin/` (support code and helper scripts)

`tests/` holds unit tests (mirrors package areas, e.g., `tests/core/`, `tests/utils/`).
`egs/` contains end-to-end ASR recipes (YesNo, TIMIT, LibriSpeech).
`tools/` provides environment/Kaldi setup scripts.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: install package in editable mode with dev tools.
- `pre-commit install`: install local formatting/lint hooks.
- `pytest tests/`: run unit tests.
- `pytest tests/ --cov=BenNevis --cov-report=term-missing --cov-report=html`: run tests with coverage.
- `black BenNevis/ && isort BenNevis/`: format code and imports.
- `flake8 BenNevis/`: run lint checks.
- `python -m build`: build distribution artifacts.

## Coding Style & Naming Conventions
- Python 3.8+; follow PEP 8 with a 120-char line limit.
- Formatting/linting are standardized with `black`, `isort` (black profile), and `flake8`.
- Test discovery expects `test_*.py`, `Test*` classes, and `test_*` functions.
- Use clear, descriptive names (`snake_case` for functions/variables, `PascalCase` for classes).

## Testing Guidelines
- Framework: `pytest` with `pytest-cov`.
- Place tests under `tests/`, near the related module area.
- Include both success and failure/edge-case coverage.
- If dependencies like `k2` are unavailable, write tests to skip cleanly when appropriate.

## Commit & Pull Request Guidelines
- Prefer concise Conventional Commit-style subjects seen in history (`fix: ...`, `feat: ...`, `docs: ...`).
- Keep the first line short and imperative; add details in the body when needed.
- PRs should use `.github/pull_request_template.md`: include change type, linked issue (`Fixes #...`), test evidence, and docs/changelog updates when relevant.
- For recipe or UI-facing changes, include reproducible commands and output snippets (or screenshots when applicable).

## Configuration Notes
- Kaldi and `k2` are external prerequisites for full ASR workflows; see `tools/put_kaldi.sh` and recipe `run.sh` files in `egs/`.
