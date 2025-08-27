# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration and testing.

## Workflows

### 1. Tests (`test.yml`)
**Trigger:** Automatically on every push to main/develop and all pull requests to main

**What it does:**
- Runs tests on Python 3.10, 3.11, and 3.12
- Performs linting with black and ruff
- Type checking with mypy
- Runs all unit and integration tests
- Generates coverage reports
- Security audit for dependencies

**Required for merge:** Yes (when branch protection is enabled)

### 2. Manual Test Run (`manual-test.yml`)
**Trigger:** Manual trigger from Actions tab

**What it does:**
- Allows running tests manually with custom parameters
- Choose Python version
- Run specific test file or all tests
- Configurable verbosity

**Use cases:**
- Debug specific test failures
- Test with specific Python version
- Quick test runs during development

## Quick Start

### Running Tests Locally

```bash
# Install the package with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_analysis_end_to_end.py -v

# Run with coverage
pytest tests/ --cov=hound --cov-report=term-missing
```

### Linting and Formatting

```bash
# Format code
black .

# Check formatting (CI mode)
black --check .

# Run linter
ruff check .

# Fix linting issues
ruff check --fix .
```

## Test Structure

```
tests/
├── test_analysis_end_to_end.py    # End-to-end integration tests
├── test_*.py                       # Other test files
└── fixtures/                       # Test fixtures (if any)
```

## Key Features

### Multi-Version Testing
- Tests run on Python 3.10, 3.11, and 3.12
- Ensures compatibility across versions

### Coverage Reporting
- Coverage reports uploaded to Codecov (if configured)
- Local coverage with `pytest --cov`

### Security Scanning
- Automatic dependency vulnerability scanning
- Uses pip-audit for security checks

### Test Artifacts
- Test results saved as XML artifacts
- Available for download from workflow runs

## Troubleshooting

### Common Issues

1. **Import errors in tests**
   - Ensure package is installed with `pip install -e .`
   - Check PYTHONPATH is set correctly

2. **Tests pass locally but fail in CI**
   - Check for hardcoded paths
   - Verify all dependencies are listed
   - Check for OS-specific code

3. **Workflow not triggering**
   - Verify branch names match
   - Check workflow syntax
   - Ensure file is in `.github/workflows/`

## Configuration Files

- `pyproject.toml` - Package configuration and test settings
- `requirements.txt` - Production dependencies
- `.github/workflows/*.yml` - GitHub Actions workflows
- `.github/BRANCH_PROTECTION.md` - Branch protection setup guide

## Contributing

1. Write tests for new features
2. Ensure all tests pass locally
3. Run linting and formatting
4. Create pull request
5. Wait for CI checks to pass
6. Request review

## Status Badges

Add these to your main README.md:

```markdown
![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Tests/badge.svg)
```

## Support

For issues with CI/CD, check:
- GitHub Actions tab for workflow runs
- Workflow logs for specific errors
- This README for common solutions