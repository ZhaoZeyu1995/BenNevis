# Contributing to BenNevis

Thank you for your interest in contributing to BenNevis! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BenNevis.git
   cd BenNevis
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/zeyuzhao/BenNevis.git
   ```

## Development Setup

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (recommended)
- Kaldi (for data preparation and decoding)

### Installation

1. Install Kaldi and link it:
   ```bash
   cd tools
   ./put_kaldi.sh /path/to/kaldi
   ```

2. Create and activate virtual environment:
   ```bash
   cd tools
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install PyTorch (adjust CUDA version as needed):
   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install k2 (match PyTorch and CUDA versions):
   ```bash
   pip install k2==1.24.4.dev20231220+cuda11.8.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html
   ```

5. Install BenNevis in development mode:
   ```bash
   cd ..
   pip install -e ".[dev]"
   ```

6. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Describe the bug clearly with steps to reproduce
- Include system information (OS, Python version, PyTorch version, etc.)
- Provide error messages and logs

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement and its benefits
- Provide examples if possible

### Code Contributions

1. Create a new branch from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

5. Format your code:
   ```bash
   black BenNevis/
   isort BenNevis/
   ```

6. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. Push to your fork and create a pull request

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Maximum line length: 120 characters
- Use `black` for code formatting
- Use `isort` for import sorting
- Use `flake8` for linting

### Code Format

Run these commands before committing:

```bash
# Format code
black BenNevis/
isort BenNevis/

# Check linting
flake8 BenNevis/
```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Include type hints where appropriate
- Update README.md if adding new features

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add detailed description if necessary

Example:
```
Add transformer encoder implementation

- Implement multi-head attention mechanism
- Add positional encoding
- Include layer normalization
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=BenNevis --cov-report=html

# Run specific test file
pytest tests/test_dataset.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Test both success and failure cases
- Mock external dependencies when appropriate

Example test structure:
```python
import pytest
from BenNevis.core.dataset import Dataset

def test_dataset_initialization():
    """Test Dataset can be initialized correctly."""
    # Test implementation
    pass

def test_dataset_invalid_input():
    """Test Dataset raises error with invalid input."""
    with pytest.raises(ValueError):
        # Test implementation
        pass
```

## Pull Request Process

1. **Update Documentation**: Ensure README.md and docstrings are updated

2. **Update CHANGELOG**: Add an entry to CHANGELOG.md under "Unreleased"

3. **Ensure Tests Pass**: All existing tests must pass, add new tests for new features

4. **Code Quality**: Run `black`, `isort`, and `flake8`

5. **Clear Description**: Provide a clear description of changes in the PR

6. **Link Issues**: Reference related issues (e.g., "Fixes #123")

7. **Review**: Be responsive to feedback and make requested changes

8. **Squash Commits**: Maintainers may ask you to squash commits before merging

### PR Title Format

- Use clear, descriptive titles
- Start with a type prefix: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

Examples:
- `feat: Add conformer model implementation`
- `fix: Correct loss calculation in GraphLoss`
- `docs: Update installation instructions`

## Questions?

If you have questions, feel free to:
- Open an issue on GitHub
- Contact the maintainer at zeyuhongwu1995@gmail.com

Thank you for contributing to BenNevis! 🏔️
