PYTHON       = python
UNITTEST     = unittest
COVERAGE     = coverage
SETUP_SOURCE = setup.py

BENCHMARK_DIR = benchmark

COVERAGE_FLAGS = --source deeprob
UNITTEST_FLAGS = --verbose --start-directory test

.PHONY: clean

# Print Coverage information on stdout
coverage_cli: unit_tests
	$(COVERAGE) report

# Run Unit Tests
unit_tests:
	$(COVERAGE) run $(COVERAGE_FLAGS) -m $(UNITTEST) discover $(UNITTEST_FLAGS)

# Run benchmarks
benchmarks:
	export PYTHONPATH=. && \
	$(PYTHON) $(BENCHMARK_DIR)/clt_queries.py && \
	$(PYTHON) $(BENCHMARK_DIR)/spn_queries.py

# Upload the PIP package
pip_upload: pip_package
	$(PYTHON) -m twine upload dist/*

# Build the PIP package
pip_package: $(SETUP_SOURCE)
	$(PYTHON) $< sdist bdist_wheel

# Clean files and directories
clean:
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf deeprob_kit.egg-info
