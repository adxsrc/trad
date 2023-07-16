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

## Create Emissivity vs Frequency functions

```python
eL11 = L11.emissivity(u.Hz, 1e7*u.cm**-3, 1e11*u.K, 10*u.Gauss, 60*u.deg)
eD16 = D16.emissivity(u.Hz, 1e7*u.cm**-3, 1e11*u.K, 10*u.Gauss, 60*u.deg)
```

## Plot

```python
nu_obs = np.logspace(8,16,num=65)
eL11_obs = eL11(nu_obs)
eD16_obs = eD16(nu_obs)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.loglog(nu_obs, eL11_obs)
ax.loglog(nu_obs, eD16_obs, '--')
ax.set_xlim(1e8, 1e16)
ax.set_ylim(1e-25,1e-15)
```
