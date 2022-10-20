# A factorisation-aware matrix element emulator (FAME)

This is the project repository to accompany the article {}. In the article we describe in detail the construction of an emulator built using neural networks to emulate electron-positron annihilation into 5 jets for NLO QCD K-factors.

Here we provide Python code to replicate our strategy.

## Requirements
FAME-Antenna is mainly written in Python.
See setup.cfg for more details.

## Installation
1. Clone the repo
2. Install with pip:
    ```
    pip install -e .
    ```

## Usage
An example Jupyter notebook is provided in
```bash
src/fame/notebooks/quickstart_notebook.ipynb
```
which runs through the necessary steps to construct the emulator.

## License
Distributed under the [GPLv3](https://opensource.org/licenses/gpl-3.0.html) License.