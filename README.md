# mas2graph

[![PyPI - Version](https://img.shields.io/pypi/v/mas2graph.svg)](https://pypi.org/project/mas2graph)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mas2graph.svg)](https://pypi.org/project/mas2graph)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

<!-- ```console -->
<!-- pip install mas2graph -->
<!-- ``` -->
<!--or-->
```console
pip install -e .
```

## Prepare de Jonge data set
Download the data from zenodo
```console
pip install zenodo_get
zenodo_get -d '10.5281/zenodo.13934470' -g "negative*.mgf" -o 'data/dejonge/'
```

Then convert the data to graph data using the data preprocessing script
```console
python scripts/dejonge_preprocessor.py  negative_training_spectra.mgf negative_spectra_train.pkl --n-batches N
python scripts/dejonge_preprocessor.py  negative_validation_spectra.mgf negative_spectra_valid.pkl --n-batches N
python scripts/dejonge_preprocessor.py  negative_testing_spectra.mgf negative_spectra_test.pkl --n-batches N
```

The benchmark can the be run by using the respective scripts in the `scripts` folder.

## License

`mas2graph` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
