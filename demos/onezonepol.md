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

# Polarized One-Zone Model for Accretion Flows

This notebook contains an one-zone model used in the EHT Sgr A* Theory Paper VIII.

It assumes that the emissivity of Sgr A* comes from a sphere with radius $R$ with uniform density $n_e$, magnetic fields $B$, temperature $\Theta_\mathrm{e}$, etc.

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

from phun          import phun
from trad.plasma   import u_T_me
from trad.sync     import coefficients
from trad.solution import constant
```

## Standard Assumptions

We set the standard parameters.
Note that we skip setting the electron number density $n_e$, because that is the parameter we need to fit.
Other explicit parameters include the EHT observation frequency $\nu$ and electron temperature $T_e$.

```python
M     = 4.14e6 * c.M_sun # black hole mass
D     = 8127   * u.pc    # distance to black hole

R     = 5                # radius of the solid sphere in the one-zone model
theta = pi / 3           # angle between magnetic field and line of sight
Rhigh = 0                # R_high parameter
```

## The One Zone Model

Using a uniform plasma ball with radius $R$, our one zone model leads to:

```python
rg = u.def_unit('rg', c.G * M / c.c**2)

@phun
def magneticfield(u_ne, u_Te, u_res=u.G, backend=None): # closure on Rhigh
    s = float((2 * c.mu0 * c.k_B * u_ne * u_Te * (1 + Rhigh))**(1/2) / u_res)
    def pure(ne, Te, beta):
        return s * (ne * Te / beta)**0.5
    return pure

@phun
def luminosity(u_nu, u_ne, u_Te, u_B, u_res=u.erg/u.s/u.Hz, backend=None): # closure on R
    W    = 16
    N    = 256
    cc   = W * ((np.arange(N) + 0.5) / N - 0.5) # cell-center
    x, y = np.meshgrid(cc, cc)
    L    = (2 * (backend.maximum(R*R - x*x - y*y, 0))**0.5)[:,:,None,None]

    Inu = constant(u_nu, u_ne, u_Te, u_B, theta * u.rad, rg, pol=True)
    s   = float((4*pi*u.sr) * (rg*W/N)**2 * Inu.unit / u_res)

    def pure(nu, ne, Te, B):
        I, tau, tauV = Inu(nu, ne, Te, B, L)
        return s * I, tau, tauV

    return pure

@phun
def flux(u_nu, u_ne, u_Te, u_B, u_res=u.Jy, backend=None): # closure on D
    Lnu = luminosity(u_nu, u_ne, u_Te, u_B)
    S   = 4 * pi * D * D
    s   = float(Lnu.unit / S / u_res)

    def pure(nu, ne, Te, B):
        L, tau, tauV = Lnu(nu, ne, Te, B)
        return s * L, tau, tauV

    return pure
```

```python
B     = magneticfield(u.cm**-3, u_T_me)
Lnu   = luminosity(u.Hz, u.cm**-3, u_T_me)
Fnu   = flux(u.Hz, u.cm**-3, u_T_me)
taunu = depth(u.Hz, u.cm**-3, u_T_me)
```

## Sanity Check

We know $n_e \sim 10^6\,\mathrm{cm}^{-3}$.
Check if this give reasonable magnetic field and flux.

```python
nu = 230e9 # observe frequency
ne =   1e6 # make a first guess...
Te =    10 # electron tempearture in unit of electron rest mass energy

display(B(ne, Te))
display(Fnu(nu, ne, Te))
```

## Solve the One-Zone Model

Really solve for $n_e$ using `scipy.optimize.root`.

```python
Fnu_obs = 2.4 # target flux in Jy

r  = root(lambda ne: Fnu(nu, ne, Te) - Fnu_obs, 1e6)
x0 = r.x[0]

display(r)
```

## Display the Solution

```python
ne = x0 # solution

display(nu)
display(ne)
display(Te)
display(B(ne, Te))
display(nu * Lnu(nu, ne, Te))
display(Fnu(nu, ne, Te))
display(taunu(nu, ne, Te))
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

nuLnu_obs = nu_obs * Lnu(nu_obs, ne/2, Te)
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * Lnu(nu_obs, ne, Te)
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * Lnu(nu_obs, ne*2, Te)
ax.loglog(nu_obs, nuLnu_obs)
```
