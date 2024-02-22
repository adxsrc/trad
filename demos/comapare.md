---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
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

from jax import config
config.update("jax_enable_x64", True)

from math import pi
from jax import jit
from jax import numpy as np
from astropy import constants as c, units as u
from matplotlib import pyplot as plt

from trad.plasma import u_T_me
from trad.sync import Dexter2016 as D16
from trad.sync import LeungX2011 as L11
```

## Create Emissivity vs Frequency functions

We the generate the emissivity functions `L11_org()` and `D16_org()` by providing `astro.units`.
To compare performance, we also create the jitted version `L11_jit()` and `D16_jit()`.

```python
L11_org = L11.coefficients(u.Hz, 1e7*u.cm**-3, 1e11*u.K, 10*u.Gauss, 60*u.deg)
D16_org = D16.coefficients(u.Hz, 1e7*u.cm**-3, 1e11*u.K, 10*u.Gauss, 60*u.deg, pol=True)

L11_jit = jit(L11_org)
D16_jit = jit(D16_org)
```

## Sanity Check

We can now perform a sanity check on the numerical values of the coefficients.

```python
nu_obs = np.logspace(8,24,num=9)
D16_jit(nu_obs)
```

## Plot

And plot the coefficients as functions of frequencies.

```python
nu_obs  = np.logspace(8,24,num=1025)
D16_obs = D16_jit(nu_obs)
```

```python
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.loglog(nu_obs, D16_obs[0][0])
ax.loglog(nu_obs, D16_obs[0][1])
ax.loglog(nu_obs, D16_obs[0][2])
ax.loglog(nu_obs, D16_obs[1][0])
ax.loglog(nu_obs, D16_obs[1][1])
ax.loglog(nu_obs, D16_obs[1][2])
ax.loglog(nu_obs, abs(D16_obs[1][3]))
ax.loglog(nu_obs, D16_obs[1][4])
ax.set_xlim(1e8, 1e24)
ax.set_ylim(1e-40,1e-0)
```

```python
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.semilogx(nu_obs, D16_obs[1][3])
ax.semilogx(nu_obs, D16_obs[1][4])
ax.set_xlim(1e8, 1e16)
ax.set_ylim(-1e-8, 1e-8)
```

## Test JAX and Performance

Time the original and jitted versions of the code.  The jitted version is about 6x faster.

```python
%timeit L11_obs = L11_org(nu_obs)
%timeit L11_obs = L11_jit(nu_obs)
```

```python
%timeit D16_obs = D16_org(nu_obs)
%timeit D16_obs = D16_jit(nu_obs)
```

## Autodiff

One stronge reason to use `JAX` is to enable autodiff.
So here we overplot the tangent computed from autodiff with the original curves.

```python
from jax import grad, vmap

f = lambda nu: L11_jit(nu)[0]
g = vmap(grad(f))
```

```python
h = 5e9
r = slice(100, 200)

X = nu_obs[r]
Y = f(nu_obs[r])
S = g(nu_obs[r])

plt.plot(X, Y, linewidth=3, alpha=0.2)
for x, y, s in list(zip(X, Y, S))[::10]:
    xm = x - h
    xp = x + h
    ym = y - h * s
    yp = y + h * s
    plt.plot([xm, xp], [ym, yp])
```

```python
h = 5e11
r = slice(300, 400)

X = nu_obs[r]
Y = f(nu_obs[r])
S = g(nu_obs[r])

plt.plot(X, Y, linewidth=3, alpha=0.2)
for x, y, s in list(zip(X, Y, S))[::10]:
    xm = x - h
    xp = x + h
    ym = y - h * s
    yp = y + h * s
    plt.plot([xm, xp], [ym, yp])
```
