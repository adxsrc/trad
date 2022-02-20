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
We will also use `scipy` for root finding and `matplotlib` for plotting.

```python
%load_ext autoreload
%autoreload 2

from astropy        import constants as c, units as u
from scipy.optimize import root
from matplotlib     import pyplot as plt

from trad.sync import *
```

## Standard Assumptions

We set the standard parameters.
Note that we skip setting the electron number density $n_e$, because that is the parameter we need to fit.

```python
nu     = 230    * u.GHz            # observe frequency

M      = 4.14e6 * c.M_sun          # black hole mass
R      = 5      * c.G * M / c.c**2 # radius of the solid sphere in the one-zone model 
D      = 8127   * u.pc             # distance to black hole

theta  = pi / 3 * u.rad            # angle between magnetic field and line of sight
Thetae = 10                        # dimensionless electron tempearture
Rhigh  = 3                         # R_high parameter
beta   = 1                         # plasma beta
```

## The One Zone Model

Using a uniform plasma ball with radius $R$, our one zone model leads to:

```python
def B(ne):
    "Magnetic field strength in G"
    Te = (c.m_e * c.c**2 * Thetae / c.k_B).to(u.K)
    return ((2 * c.mu0 * c.k_B * ne * Te * (1 + Rhigh) / beta)**(1/2)).to(u.G)

def Lnu(ne):
    P = jnu(nu, ne, Thetae, B(ne), theta) * (4 * pi * u.sr)
    V = (4/3) * pi * R**3
    return (P * V).to(u.erg / u.s / u.Hz)

def Fnu(ne):
    S = 4 * pi * D * D
    return (Lnu(ne) / S).to(u.Jy)

def taunu(ne):
    return (R * anu(nu, ne, Thetae, B(ne), theta)).to(u.dimensionless_unscaled)
```

## Sanity Check

We know $n_e \sim 10^6\,\mathrm{cm}^{-3}$.
Check if this give reasonable magnetic field and flux.

```python
ne = 1e6 / u.cm**3 # make a first guess...

display(B(ne))
display(Fnu(ne))
```

## Solve the One-Zone Model

Really solve for $n_e$ using `scipy.optimize.root`.

```python
Fnu_obs = 2.4 * u.Jy # target flux

r  = root(lambda ne: (Fnu(ne/u.cm**3) - Fnu_obs).value, 1e6)
x0 = r.x[0]

display(r)
```

```python

```
