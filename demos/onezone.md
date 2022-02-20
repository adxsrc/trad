---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# One-Zone Model for Optically Thin Accreetion Flows

This notebook contains an one-zone model used in the EHT Sgr A* Theory Paper V.

It assumes that the emissivity of Sgr A* is optically thin and comes from a sphere with radius $R$ with uniform density $n_e$, magnetic fields $B$, temperature $\Theta_\mathrm{e}$, etc.

This notebook can be modified to perfrom one-zone model estimates for other AGNs.


## Autoreload and Import Modules

To streamline the development of `trad`, we enable the autoreload `ipython` extension.
This makes any changes to `trad` code automatically available in this notebook. 

We first input the necessary `python` modules.  Just like `trad`, we use `astropy`'s unit and constant modules.
We will also use `matplotlib` for plotting.

```python
%load_ext autoreload
%autoreload 2

from astropy    import constants as c, units as u
from matplotlib import pyplot as plt

from trad.sync import *
```
