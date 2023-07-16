---
jupyter:
  jupytext:
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

# Comparing Different Approximations


## Autoreload and Import Modules

To streamline the development of `trad`, we enable the autoreload `ipython` extension.
This makes any changes to `trad` code automatically available in this notebook.

We first input the necessary `python` modules.  Just like `trad`, we use `astropy`'s unit and constant modules.
We will also use `matplotlib` for plotting.

```python
%load_ext autoreload
%autoreload 2

from math import pi
import numpy as np
from astropy import constants as c, units as u
from matplotlib import pyplot as plt

from trad.plasma import u_T_me
from trad.sync import Dexter2016 as D16
from trad.sync import Leung2011  as L11
```

## Test Implementation

```python
Te = float(1e11*u.K/u_T_me)

display(Te)
display(L11.emissivity(230e9*u.Hz, 1e7*u.cm**-3, 1e11*u.K, 10*u.G, 60*u.deg)())
display(L11.emissivity(230e9*u.Hz, 1e7*u.cm**-3, u_T_me,   10*u.G, 60*u.deg)(Te))
```
