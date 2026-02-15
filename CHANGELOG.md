# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Package configuration with `pyproject.toml` for proper Python packaging
- Dependency management with `requirements.txt` and `requirements-dev.txt`
- Version information in `BenNevis/__init__.py`
- GitHub Actions CI/CD pipeline for automated testing and linting
- Basic unit tests for core modules
- `CONTRIBUTING.md` with development guidelines
- `CITATION.cff` for academic citation
- Pre-commit hooks configuration
- GitHub issue and PR templates
- This CHANGELOG file

### Changed
- Updated README.md with improved installation instructions
- Enhanced `.gitignore` for better coverage

## [0.1.0] - 2024-01-01

### Added
- Initial release of BenNevis
- End-to-end ASR training with DWFST support
- Multiple topology support (CTC, MMI-CTC, state-based models)
- PyTorch Distributed Data Parallel (DDP) training support
- Integration with k2 for WFST operations
- Kaldi-style data preparation pipeline
- Support for multiple model architectures:
  - RNNP (RNN-Transducer)
  - Conformer
  - Transformer
  - Wav2Vec2
  - Whisper
- WandB logger integration
- Example recipes for YesNo, TIMIT, and LibriSpeech datasets
- Flexible graph loss computation
- Dynamic batch sampling
- Alignment tools

[Unreleased]: https://github.com/zeyuzhao/BenNevis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zeyuzhao/BenNevis/releases/tag/v0.1.0
