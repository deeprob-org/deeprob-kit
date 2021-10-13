SPHINX_BUILD      = sphinx-multiversion
SPHINX_BUILD_DIR  = _build/html
SPHINX_SOURCE_DIR = .

.PHONY: clean

# Make the HTML static site
sphinx_html:
	$(SPHINX_BUILD) $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)

# Clean directories
clean:
	rm -rf $(SPHINX_BUILD_DIR)
