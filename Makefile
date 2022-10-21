PYTHON    = python
PYTEST    = pytest
PYLINT    = pylint
BLACK     = black

SOURCE_DIR = deeprob
TESTS_DIR  = tests


.PHONY: all clean

# Perform tests and print static code quality
all: pytest pylint

# Run black (check only)
black_check:
	$(BLACK) "$(SOURCE_DIR)" --check --diff --color

# Run black (format files)
black:
	$(BLACK) "$(SOURCE_DIR)"

# Run tests with HTML coverage output
pytest:
	$(PYTEST) --cov "$(SOURCE_DIR)" --cov-report=html

# Run tests for Codecov
pytest_codecov:
	$(PYTEST) --cov "$(SOURCE_DIR)" --cov-report=xml

# Print static code quality to stdout
pylint:
	$(PYLINT) "$(SOURCE_DIR)"

# Upload the PIP package
pip_upload: pip_package
	$(PYTHON) -m twine upload dist/*

# Build the PIP package
pip_package:
	$(PYTHON) -m build .

# Clean files
clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf deeprob_kit.egg-info dist
