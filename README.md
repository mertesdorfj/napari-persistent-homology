# napari-persistent-homology

[![License BSD-3](https://img.shields.io/pypi/l/napari-persistent-homology.svg?color=green)](https://github.com/mertesdorfj/napari-persistent-homology/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-persistent-homology.svg?color=green)](https://pypi.org/project/napari-persistent-homology)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-persistent-homology.svg?color=green)](https://python.org)
[![tests](https://github.com/mertesdorfj/napari-persistent-homology/workflows/tests/badge.svg)](https://github.com/mertesdorfj/napari-persistent-homology/actions)
[![codecov](https://codecov.io/gh/mertesdorfj/napari-persistent-homology/branch/main/graph/badge.svg)](https://codecov.io/gh/mertesdorfj/napari-persistent-homology)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-persistent-homology)](https://napari-hub.org/plugins/napari-persistent-homology)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

3D shape analysis of binary segmentations using persistent homology

----------------------------------

This [napari] plugin wraps research code from [Wang et al. 2022](https://doi.org/10.1101/2022.11.08.515664) for 3D shape analysis of binary segmentation volumes using persistent homology.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-persistent-homology` via [pip]:

```bash
pip install napari-persistent-homology
```

If napari is not already installed, you can install `napari-persistent-homology` with napari and Qt via:

```bash
pip install "napari-persistent-homology[all]"
```


To install latest development version:

```bash
pip install git+https://github.com/mertesdorfj/napari-persistent-homology.git
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-persistent-homology" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/mertesdorfj/napari-persistent-homology/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
