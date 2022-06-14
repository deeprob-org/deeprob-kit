PYTHON    = python
PYLINT    = pylint
UNITTEST  = unittest
COVERAGE  = coverage
SETUP_SRC = setup.py

SOURCE_DIR    = deeprob
TEST_DIR      = test
BENCHMARK_DIR = benchmark


.PHONY: all clean

# Print static code quality and coverage information to stdout
all: pylint_cli coverage_cli

# Clean all
clean: clean_coverage clean_pip

# Print static code quality to stdout
pylint_cli:
	$(PYLINT) "$(SOURCE_DIR)"

# Print coverage information to stdout
coverage_cli: unit_tests
	$(COVERAGE) report

# Run unit tests
unit_tests:
	$(COVERAGE) run --source "$(SOURCE_DIR)" -m $(UNITTEST) discover --start-directory "$(TEST_DIR)"

# Run benchmarks
benchmarks:
	for SCRIPT in $(wildcard $(BENCHMARK_DIR)/run_*.py); do PYTHONPATH=. $(PYTHON) $$SCRIPT; done

# Upload the PIP package
pip_upload: pip_package
	$(PYTHON) -m twine upload dist/*

# Build the PIP package
pip_package: clean_pip $(SETUP_SRC)
	$(PYTHON) $(SETUP_SRC) sdist bdist_wheel

# Clean tests and coverage related files
clean_coverage:
	rm -rf .coverage

# Clean PIP package related files
clean_pip:
	rm -rf dist build deeprob_kit.egg-info
