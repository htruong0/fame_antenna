# A factorisation-aware matrix element emulator (FAME) for NLO QCD K-factors

This is the project repository to accompany the article {}. In the article we describe in detail the construction of an emulator built using neural networks to emulate electron-positron annihilation into 5 jets for NLO QCD K-factors. The emulator uses the factorisation properties of matrix elements, along with universal singular functions (in this case antenna functions described in the article [Antenna Subtraction at NNLO](https://arxiv.org/pdf/hep-ph/0505111.pdf)) to achieve high levels of accuracy.

Here we provide Python code to replicate our strategy.

## Requirements
FAME-Antenna is mainly written in Python.
See setup.cfg for more details on packages
required.

In the example notebook, we use LHAPDF for $\alpha_{s}$ evaluation and MadGraph for matrix element evaluations. These can of course be substituted with other packages.

## Installation
1. Clone the repo
2. Install with pip:
    ```
    pip install -e .
    ```

## Usage
An example Jupyter notebook is provided in
```bash
src/fame_antenna/notebooks/antenna_model_notebook.ipynb
```
which runs through the necessary steps to construct the emulator.

## License
Distributed under the [GPLv3](https://opensource.org/licenses/gpl-3.0.html) License.