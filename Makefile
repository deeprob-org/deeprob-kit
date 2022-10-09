PYTHON    = python
PYLINT    = pylint
PYTEST    = pytest
SETUP_SRC = setup.py

SOURCE_DIR    = deeprob
TESTS_DIR     = tests
BENCHMARK_DIR = benchmark


.PHONY: all clean

# Print static code quality, perform tests and build PIP package
all: pylint pytest pip_package

# Print static code quality to stdout
pylint:
	$(PYLINT) "$(SOURCE_DIR)"

# Run tests with HTML coverage output
pytest:
	$(PYTEST) "$(TESTS_DIR)" --cov "$(SOURCE_DIR)" --cov-report=html

# Run tests for Codecov
pytest_codecov:
	$(PYTEST) "$(TESTS_DIR)" --cov "$(SOURCE_DIR)" --cov-report=xml

# Run benchmarks
benchmarks:
	for SCRIPT in $(wildcard $(BENCHMARK_DIR)/run_*.py); do PYTHONPATH=. $(PYTHON) $$SCRIPT; done

# Upload the PIP package
pip_upload: pip_package
	$(PYTHON) -m twine upload dist/*

# Build the PIP package
pip_package: $(SETUP_SRC)
	$(PYTHON) $(SETUP_SRC) sdist bdist_wheel

# Clean files
clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf dist build deeprob_kit.egg-info
