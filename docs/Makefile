SPHINX_BUILD      = sphinx-build
SPHINX_APIDOC     = sphinx-apidoc
SPHINX_BUILD_DIR  = _build
SPHINX_SOURCE_DIR = .
SPHINX_API_DIR    = api
API_SOURCE_DIR    = ../deeprob

.PHONY: clean

# Make the HTML static site
sphinx_html: sphinx_api
	$(SPHINX_BUILD) -M html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)

# Make the API pages
sphinx_api:
	$(SPHINX_APIDOC) -o $(SPHINX_API_DIR) $(API_SOURCE_DIR)

# Clean directories
clean:
	rm -rf $(SPHINX_API_DIR)
	rm -rf $(SPHINX_BUILD_DIR)
