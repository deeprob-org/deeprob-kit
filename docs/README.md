## Documentation

The source code documentation is hosted using GitHub Pages at
[deeprob-org.github.io/deeprob-kit](https://deeprob-org.github.io/deeprob-kit/).

The documentation is generated automatically by Sphinx, using sources stored in the `docs` directory
(with a slightly modified [*Read-the-Docs*](https://readthedocs.org/) theme).
In particular, the documentation is versioned, i.e. there is a documentation page for the main branch and for every tag
(or release) of the library.

If you wish to build the documentation yourself, you will need to install the dependencies listed in `requirements.txt`.
Moreover, in order to build versioned documentation, it is necessary to install a
[fork of sphinx-multiversion](https://github.com/Holzhaus/sphinx-multiversion/pull/62) as following:
```shell
pip install git+https://github.com/samtygier-stfc/sphinx-multiversion.git@prebuild_command
```

Finally, it's possible to execute the Makefile script as following:
```shell
# Clean existing documentation (optional)
make clean
# Build HTML documentation
make
```
The output HTML documentation, for any local branch and tag (or release), can be found inside the `_build`
directory.
