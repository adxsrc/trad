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

# One-Zone Model for Optically Thin Accretion Flows

This notebook contains an one-zone model used in the EHT Sgr A* Theory Paper V.

It assumes that the emissivity of Sgr A* is optically thin and comes from a sphere with radius $R$ with uniform density $n_e$, magnetic fields $B$, temperature $\Theta_\mathrm{e}$, etc.

This notebook can be modified to perfrom one-zone model estimates for other AGNs.


## Autoreload and Import Modules

To streamline the development of `trad`, we enable the autoreload `ipython` extension.
This makes any changes to `trad` source code automatically available in this notebook.

We first input the necessary `python` modules.  Just like `trad`, we use `astropy`'s unit and constant modules.
We will also use `scipy` for root finding and `matplotlib` for plotting.

```python
%load_ext autoreload
%autoreload 2

from math import pi
import numpy as np

from astropy        import constants as c, units as u
from scipy.optimize import root
from matplotlib     import pyplot as plt

from phun        import phun
from trad.plasma import u_T_me
from trad.sync   import coefficients
```

## Standard Assumptions

We set the standard parameters.
Note that we skip setting the electron number density $n_e$, because that is the parameter we need to fit.

```python
M     = 4.14e6 * c.M_sun          # black hole mass
R     = 5      * c.G * M / c.c**2 # radius of the solid sphere in the one-zone model
D     = 8127   * u.pc             # distance to black hole

theta = pi / 3                    # angle between magnetic field and line of sight
Te    = 10                        # electron tempearture in unit of electron rest mass energy
Rhigh = 3                         # R_high parameter
beta  = 1                         # plasma beta
```

## The One Zone Model

Using a uniform plasma ball with radius $R$, our one zone model leads to:

```python
@phun
def magneticfield(u_ne, u_res=u.G, backend=None): # closure on Rhigh and beta
    s = float((2 * c.mu0 * c.k_B * u_ne * u_T_me * (1 + Rhigh) / beta)**(1/2) / u_res)
    def pure(ne):
        return s * (ne * Te)**0.5
    return pure

@phun
def luminosity(u_nu, u_ne, u_res=u.erg/u.s/u.Hz, backend=None): # closure on R
    B = magneticfield(u_ne)
    C = coefficients(u_nu, u_ne, u_T_me, B.unit, u.rad, pol=False)
    V = (4/3) * pi * R**3
    s = float((4 * pi * u.sr) * V * C.unit[0] / u_res)

    def pure(nu, ne): # closure on theta and Te
        return s * C(nu, ne, Te, B(ne), theta)[0]

    return pure

@phun
def flux(u_nu, u_ne, u_res=u.Jy, backend=None): # closure on D
    Lnu = luminosity(u_nu, u_ne)
    S   = 4 * pi * D * D
    s   = float(Lnu.unit / S / u_res)

    def pure(nu, ne):
        return s * Lnu(nu, ne)

    return pure

@phun
def opticaldepth(u_nu, u_ne, u_res=u.dimensionless_unscaled, backend=None): # closure on R
    B = magneticfield(u_ne)
    C = coefficients(u_nu, u_ne, u_T_me, B.unit, u.rad)
    s = float(R * C.unit[1] / u_res)

    def pure(nu, ne): # closure on theta and Te
        return s * C(nu, ne, Te, B(ne), theta)[1]

    return pure
```

```python
B     = magneticfield(u.cm**-3)
Lnu   = luminosity(u.Hz, u.cm**-3)
Fnu   = flux(u.Hz, u.cm**-3)
taunu = opticaldepth(u.Hz, u.cm**-3)
```

## Sanity Check

We know $n_e \sim 10^6\,\mathrm{cm}^{-3}$.
Check if this give reasonable magnetic field and flux.

```python
nu = 230e9 # observe frequency
ne =   1e6 # make a first guess...

display(B(ne))
display(Fnu(nu, ne))
```

## Solve the One-Zone Model

Really solve for $n_e$ using `scipy.optimize.root`.

```python
Fnu_obs = 2.4 # target flux in Jy

r  = root(lambda ne: Fnu(nu, ne) - Fnu_obs, 1e6)
x0 = r.x[0]

display(r)
```

## Display the Solution

```python
ne = x0 # solution

display(ne)
display(B(ne))
display(nu * Lnu(nu, ne))
display(Fnu(nu, ne))
display(taunu(nu, ne))
```

## Sgr A* SED

Plot only the synchrotron SED for Sgr A*, assuming the electron number density is the 1/2 of the solved one, the solved one, and 2x the solved one.

```python
nu_obs = np.logspace(8,16,num=65)

fig, ax = plt.subplots(1,1,figsize=(8,8))

ax.set_xlim(1e8, 1e16)
ax.set_ylim(1e28,1e36)

ax.set_xlabel(r'Frequency $\nu$ [Hz]')
ax.set_ylabel(r'$\nu L_\nu$ [erg/s]')

nuLnu_obs = nu_obs * Lnu(nu_obs, ne/2)
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * Lnu(nu_obs, ne)
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * Lnu(nu_obs, ne*2)
ax.loglog(nu_obs, nuLnu_obs)
```
