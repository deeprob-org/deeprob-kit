SPHINX_OPTS   ?=
SPHINX_BUILD  ?= sphinx-build
SPHINX_APIDOC ?= sphinx-apidoc

SPHINX_BUILD_DIR  = build
SPHINX_SOURCE_DIR = source
API_SOURCE_DIR    = $(SPHINX_SOURCE_DIR)/api
LIB_SOURCE_DIR    = ../deeprob

.PHONY: clean

html: api
	$(SPHINX_BUILD) -M html "$(SPHINX_SOURCE_DIR)" "$(SPHINX_BUILD_DIR)" $(SPHINX_OPTS)

api:
	$(SPHINX_APIDOC) -o "$(API_SOURCE_DIR)" "$(LIB_SOURCE_DIR)"

clean:
	rm -rf "$(SPHINX_BUILD_DIR)" "$(API_SOURCE_DIR)"
