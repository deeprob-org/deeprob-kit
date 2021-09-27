PYTHON       = python
UNITTEST     = unittest
COVERAGE     = coverage
SETUP_SOURCE = setup.py

COVERAGE_FLAGS = --source deeprob
UNITTEST_FLAGS = --verbose

.PHONY: pip_clean

show_coverage: unit_tests
	$(COVERAGE) report -m

unit_tests:
	$(COVERAGE) run $(COVERAGE_FLAGS) -m $(UNITTEST) $(UNITTEST_FLAGS)

pip_package: $(SETUP_SOURCE)
	$(PYTHON) $< sdist bdist_wheel

pip_upload: pip_package
	$(PYTHON) -m twine upload dist/*

pip_clean:
	rm -rf dist
	rm -rf build
	rm -rf deeprob_kit.egg-info
