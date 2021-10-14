SPHINX_BUILD      = sphinx-multiversion
SPHINX_BUILD_DIR  = _build
SPHINX_SOURCE_DIR = .
SPHINX_INDEX_FILE = index.html

.PHONY: clean

# Make the HTML static site
sphinx_html:
	$(SPHINX_BUILD) $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)
	cp $(SPHINX_SOURCE_DIR)/$(SPHINX_INDEX_FILE) $(SPHINX_BUILD_DIR)

# Clean directories
clean:
	rm -rf $(SPHINX_BUILD_DIR)

