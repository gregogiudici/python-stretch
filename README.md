# pystretch
================

|      CI              | status |
|----------------------|--------|
| pip builds           | [![Pip Action Status][actions-pip-badge]][actions-pip-link] |
| wheels               | [![Wheel Action Status][actions-wheels-badge]][actions-wheels-link] |

[actions-pip-link]:        https://github.com/gregogiudici/python-signalsmith-stretch/actions?query=workflow%3APip
[actions-pip-badge]:       https://github.com/gregogiudici/python-signalsmith-stretch/workflows/Pip/badge.svg
[actions-wheels-link]:     https://github.com/gregogiudici/python-signalsmith-stretch/actions?query=workflow%3AWheels
[actions-wheels-badge]:    https://github.com/gregogiudici/python-signalsmith-stretch/workflows/Wheels/badge.svg

A Python Wrapprer of the Signalsmith Stretch C++ library for pitch and time stretching

Installation
------------

1. Clone this repository
2. Run `cd python-signalsmith-stretch && pip install .`


CI Examples
-----------

The `.github/workflows` directory contains two continuous integration workflows
for GitHub Actions. The first one (`pip`) runs automatically after each commit
and ensures that packages can be built successfully and that tests pass.

The `wheels` workflow uses
[cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/) to automatically
produce binary wheels for a large variety of platforms. If a `pypi_password`
token is provided using GitHub Action's _secrets_ feature, this workflow can
even automatically upload packages on PyPI.


License
-------

_nanobind_ and this example repository are both provided under a BSD-style
license that can be found in the [LICENSE](./LICENSE) file. By using,
distributing, or contributing to this project, you agree to the terms and
conditions of this license.
