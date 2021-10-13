## Documentation

The documentation is generated automatically by Sphinx, using sources stored in the `docs` directory
(with a slightly modified [*Read-the-Docs*](https://readthedocs.org/) theme).
Sooner or later we will make it available also online, probably hosted using GitHub pages. 

If you wish to build the documentation yourself, you will need to install the dependencies listed in `requirements.txt`
and execute the Makefile script as following:
```bash
# Clean existing documentation (optional)
make clean

# Build source code API documentation
make sphinx_api

# Build HTML documentation
make sphinx_html
```
The output HTML documentation can be found inside `_build/html` directory.
