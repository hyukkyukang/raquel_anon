# RAQUEL Tests

This directory contains unit tests for the RAQUEL project.

## Test Structure

```
test/
├── __init__.py
├── README.md                # This file
├── training/                # Tests for training modules
├── aligned_db/              # Aligned DB pipeline tests
├── metrics/                 # Metric tests
└── template/                # Template system tests
```

## Running Tests

### Prerequisites

Install testing dependencies:

```bash
pip install -r requirements.txt
# Optional extras for local runs
pip install pytest pytest-cov
```

`requirements.txt` already includes the training stack (`torch`, `transformers`, `lightning`, `peft`, etc.). Tests that need optional ML runtimes now skip automatically when those packages are unavailable.

### Run All Tests

```bash
# Using the test runner
python run_tests.py

# Using pytest (recommended)
pytest test/ -v

# Using unittest
python -m unittest discover -s test -p "test_*.py" -v
```

### Run Specific Test Modules

```bash
# Run only import tests
pytest test/training/test_imports.py -v

# Run only data tests
pytest test/training/test_data.py -v

# Run only utils tests
pytest test/training/test_utils.py -v
```

### Run Specific Test Classes

```bash
# Run a specific test class
pytest test/training/test_imports.py::TestTrainingImports -v

# Run a specific test method
pytest test/training/test_utils.py::TestUnlearningMethodParsing::test_parse_valid_methods -v
```

## Test Coverage

### Current Test Coverage

- **Import Tests** (`test_imports.py`):
  - All module imports (data, datamodules, models, losses, callbacks, utils)
  - Unlearning method definitions and descriptions
  - Main training module structure

- **Data Tests** (`test_data.py`):
  - CustomDataCollator functionality
  - Dataset loading from JSON
  - Tokenization with label masking
  - IDK dataset creation

- **Utility Tests** (`test_utils.py`):
  - Model directory utilities
  - Unlearning method parsing
  - Method requirement detection (IDK, reference model)
  - Base model directory name generation
  - Method descriptions validation
  - Current NPO semantics: no IDK dataset required, but a reference model is required

### Test Statistics

```bash
# Get test count
pytest test/ --collect-only | grep "test session starts" -A 1

# Run with coverage (if pytest-cov installed)
pytest test/ --cov=src.training --cov-report=html
```

## Writing New Tests

### Test Template

```python
"""Test description."""

import unittest


class TestFeatureName(unittest.TestCase):
    """Test suite for FeatureName."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_specific_behavior(self):
        """Test a specific behavior."""
        # Arrange
        # Act
        # Assert
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
```

### Testing Guidelines

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clear Names**: Use descriptive test method names (e.g., `test_parse_invalid_method_raises_error`)
3. **AAA Pattern**: Follow Arrange-Act-Assert pattern
4. **Mock External Dependencies**: Use mocking for file I/O, network calls, etc.
5. **Test Edge Cases**: Include tests for edge cases and error conditions

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest test/ -v --cov=src.training
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Fast execution
- No external dependencies (use mocking)
- Current location: `test/training/`

### Integration Tests (Future)
- Test interactions between components
- May use temporary files/databases
- Future location: `test/integration/`

### End-to-End Tests (Future)
- Test complete training workflows
- Slow execution
- May require GPU
- Future location: `test/e2e/`

## Troubleshooting

### ModuleNotFoundError for `lightning`, `transformers`, or `torch`

Install the project requirements:
```bash
pip install -r requirements.txt
```

### Tests Pass Locally But Fail in CI

- Check Python version compatibility
- Ensure all dependencies are in requirements files
- Check for environment-specific code paths

### Import Errors

Make sure you're running tests from the project root:
```bash
cd /home/user/RAQUEL
pytest test/
```

## Contributing

When adding new functionality to `src/training/`, please add corresponding tests:

1. Create tests in the appropriate `test_*.py` file
2. Run tests locally: `pytest test/training/ -v`
3. Ensure all tests pass before submitting PR
4. Aim for >80% code coverage for new code

## Future Enhancements

- [ ] Integration tests for DataModules with real tokenizers
- [ ] Integration tests for LightningModules
- [ ] End-to-end training tests (short epochs)
- [ ] Performance benchmarking tests
- [ ] Memory leak detection tests
- [ ] Multi-GPU testing
- [ ] Code coverage reporting in CI
