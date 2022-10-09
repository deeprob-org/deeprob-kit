# Documentation

The source code documentation is hosted using GitHub Pages at
[deeprob-kit.readthedocs.io/en/latest](https://deeprob-kit.readthedocs.io/en/latest/).

The documentation is generated automatically by Sphinx, using sources stored in the `docs` directory
(with a [*Read-the-Docs*](https://readthedocs.org/) theme).

If you wish to build the documentation yourself, you will need to install the dependencies listed in `requirements.txt`.
Finally, it's possible to execute the Makefile script as following:
```shell
# Clean existing documentation (optional)
make clean
# Build HTML documentation
make html
```
The output HTML documentation can be found inside the `_build` directory.
