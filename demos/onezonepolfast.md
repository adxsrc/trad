---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fast Polarized One-Zone Model for Accretion Flows

This notebook contains a fast one-zone model identical to the one used in the EHT Sgr A* Theory Paper VIII.

It assumes that the emissivity of Sgr A* comes from a sphere with radius $R$ with uniform density $n_e$, magnetic fields $B$, temperature $\Theta_\mathrm{e}$, etc.

This notebook can be modified to perfrom one-zone model estimates for other AGNs.


## Autoreload and Import Modules

To streamline the development of `trad`, we enable the autoreload `ipython` extension.
This makes any changes to `trad` source code automatically available in this notebook.

We first input the necessary `python` modules.  Just like `trad`, we use `astropy`'s unit and constant modules.
We will also use `scipy` for root finding and `matplotlib` for plotting.

```python
try:
    %load_ext autoreload
    %autoreload 2
except ModuleNotFoundError as e:
    print(e)
```

```python
import jax
jax.config.update("jax_enable_x64", True)
```

```python
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
```

```python
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from astropy    import constants as c, units as u

from jax       import jit, numpy as np
from jax.numpy import pi, exp

from phun          import phun
from trad.plasma   import u_T_me
from trad.solution import constant

import pandas as pd
```

## The One Zone Model

Using a uniform plasma ball with radius $R$, our one zone model leads to:

```python
@phun
def magneticfield(
    u_ne, u_Te, u_res=u.G,
    Rhigh=0,
    backend=None,
):
    s = float((2 * c.mu0 * c.k_B * u_ne * u_Te * (1 + Rhigh))**(1/2) / u_res)
    def pure(ne, Te, beta):
        return s * (ne * Te / beta)**0.5
    return pure

@phun
def luminosity(
    u_nu, u_ne, u_Te, u_B, u_res=u.erg/u.s/u.Hz,
    M=4.14e6*c.M_sun, R=5, theta=pi/3,
    backend=None,
):
    N  = 256
    r  = R * (np.arange(N) + 0.5) / N # cell-center
    L  = (2 * (backend.maximum(R*R - r*r, 0))**0.5)
    rg = u.def_unit('rg', c.G * M / c.c**2)

    Inu = constant(u_nu, u_ne, u_Te, u_B, theta * u.rad, rg, pol=True)
    s   = float((4*pi*u.sr) * (2*pi*rg*rg*R/N) * Inu.unit / u_res) * r

    def pure(nu, ne, Te, B):
        I, tau, tauV = Inu(nu, ne, Te, B, L)
        return s * I, tau, tauV

    return pure

@phun
def flux(
    u_nu, u_ne, u_Te, u_B, u_res=u.Jy,
    D=8127*u.pc,
    backend=None,
):
    Lnu = luminosity(u_nu, u_ne, u_Te, u_B)
    S   = 4 * pi * D * D
    s   = float(Lnu.unit / S / u_res)

    def pure(nu, ne, Te, B):
        L, tau, tauV = Lnu(nu, ne, Te, B)
        return s * L, tau, tauV

    return pure
```

```python
class OneZone:
    def __init__(self, M, D):
        self.B_   = magneticfield(      u.cm**-3, u_T_me)
        self.Lnu_ = luminosity   (u.Hz, u.cm**-3, u_T_me, u.G, M=M)
        self.Fnu_ = flux         (u.Hz, u.cm**-3, u_T_me, u.G, D=D)

    def Lnu(self, nu, ne, Te, beta):
        B            = self.B_(ne, Te, beta)
        L, tau, tauV = self.Lnu_(nu, ne, Te, B)
        return np.sum(L), np.max(tau), np.max(tauV)
    
    def Fnu(self, nu, ne, Te, beta):
        B            = self.B_(ne, Te, beta)
        F, tau, tauV = self.Fnu_(nu, ne, Te, B)
        return np.sum(F), np.max(tau), np.max(tauV)
```

## Solve the One-Zone Model

Really solve for $n_e$ using `scipy.optimize.root`.

```python
obs = pd.read_csv("onezone_spectra.txt")
nus = np.logspace(8,16,num=65)
```

```python
obs
```

```python
fig, ax = plt.subplots(1,1)

ax.set_xlabel(r'Frequency $\nu$ [Hz]')
ax.set_ylabel(r'$F_\nu$ [Jy]')

for i, row in enumerate(obs.iloc):
    
    oz = OneZone(
        M=row.MBH_solar_mass * c.M_sun,
        D=row.Distance_Mpc   * u.Mpc,
    )

    nu_F = {
        3e9:   1e-3 * row.Flux_3GHz_mJy,
        86e9:  1e-3 * row.Flux_86GHz_mJy,
        230e9: 1e-3 * row.Flux_230GHz_mJy,
    }
    
    def f(x):
        ne, Te, logbeta = x
        s = 0
        for nu, F in nu_F.items():
            d  = np.log(oz.Fnu(nu, ne, Te, exp(logbeta))[0]) - np.log(F)
            s += d * d
        return s

    r = minimize(f, [1e20, 1, 0])
    if r.success:
        print('SUCCEED', r.x)
    else:
        print('FAILED', r.x)
    nu, Te, logbeta = r.x
        
    Fnus = np.array([oz.Fnu(nu, nu, Te, exp(logbeta))[0] for nu in nus])
    
    ax.loglog(nus, Fnus, color=f'C{i}')
    ax.scatter(nu_F.keys(), nu_F.values(), color=f'C{i}')
```
