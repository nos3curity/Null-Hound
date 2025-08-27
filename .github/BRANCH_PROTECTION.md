# GitHub Branch Protection Rules Setup

This document describes how to configure branch protection rules to make tests mandatory for all pull requests.

## Setting Up Branch Protection

### 1. Navigate to Repository Settings
1. Go to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click on "Branches" under "Code and automation"

### 2. Add Branch Protection Rule

Click "Add rule" and configure the following:

#### Branch name pattern
- Enter: `main` (or your default branch name)

#### Protection Settings

**Required status checks:**
- ✅ **Require status checks to pass before merging**
- ✅ **Require branches to be up to date before merging**
- Select these status checks (they will appear after the first workflow run):
  - `test (3.10)`
  - `test (3.11)`
  - `test (3.12)`
  - `test-integration`
  - `all-checks-passed`

**Require a pull request before merging:**
- ✅ **Require a pull request before merging**
- ✅ **Require approvals** (recommended: 1-2 approvals)
- ✅ **Dismiss stale pull request approvals when new commits are pushed**
- ✅ **Require review from CODEOWNERS** (if using CODEOWNERS file)

**Additional settings (recommended):**
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits** (optional but recommended)
- ✅ **Include administrators** (enforce rules for admins too)
- ✅ **Restrict who can push to matching branches** (optional)

### 3. Save Protection Rules

Click "Create" or "Save changes" to apply the protection rules.

## GitHub Actions Workflow Details

The workflow (`.github/workflows/test.yml`) includes:

### Test Matrix
- Runs tests on Python 3.10, 3.11, and 3.12
- Ensures compatibility across different Python versions

### Test Jobs

1. **`test`** - Main test suite
   - Runs linting (black, ruff)
   - Type checking (mypy)
   - Unit tests with coverage
   - Uploads test results and coverage reports

2. **`test-integration`** - Integration tests
   - Runs end-to-end tests
   - Tests module imports
   - Only runs on pull requests

3. **`security`** - Security audit
   - Checks for vulnerable dependencies
   - Uses pip-audit for security scanning

4. **`all-checks-passed`** - Final check
   - Ensures all required jobs passed
   - Single status check for branch protection

## Workflow Triggers

Tests run automatically on:
- **Push** to `main` or `develop` branches
- **Pull requests** to `main` branch
- **PR events**: opened, synchronize, reopened

## Local Testing

Before pushing, you can run tests locally:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black --check .
ruff check .

# Run type checking
mypy hound/

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_analysis_end_to_end.py -v

# Run with coverage
pytest tests/ --cov=hound --cov-report=term-missing
```

## Troubleshooting

### Tests not appearing in branch protection
- Push the workflow file to your default branch first
- Run the workflow at least once (manually or via PR)
- The status checks will then appear in the protection rule settings

### Workflow not triggering
- Ensure the workflow file is in `.github/workflows/` directory
- Check that the file has `.yml` or `.yaml` extension
- Verify branch names in the workflow match your repository

### Tests failing on CI but passing locally
- Check Python version differences
- Ensure all dependencies are in `pyproject.toml` or `requirements.txt`
- Look for environment-specific issues (paths, permissions, etc.)

## Benefits

With these protection rules:
- ✅ No broken code can be merged to main
- ✅ All tests must pass before merging
- ✅ Code quality is enforced (linting, formatting)
- ✅ Security vulnerabilities are caught early
- ✅ Consistent code across the team

## Additional Recommendations

1. **Add a CODEOWNERS file** to automatically request reviews from specific team members
2. **Set up Codecov** for detailed coverage reports and PR comments
3. **Add status badges** to your README.md:

```markdown
![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO)
```

4. **Configure Dependabot** for automatic dependency updates with PR creation