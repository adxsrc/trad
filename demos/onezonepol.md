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

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from math import pi
import numpy as np

from astropy        import constants as c, units as u
from scipy.optimize import root
from matplotlib     import pyplot as plt

from phun          import phun
from trad.plasma   import u_T_me
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
B   = magneticfield(u.cm**-3, u_T_me)
Inu = constant(u.Hz, u.cm**-3, u_T_me, u.G, u.deg, rg, pol=True)
Lnu = luminosity(u.Hz, u.cm**-3, u_T_me, u.G)
Fnu = flux(u.Hz, u.cm**-3, u_T_me, u.G)
```

```python
Inu(230e9, 100, 10, 10, 60, R)
```

```python
L = np.logspace(2,10,num=65)
I, tau, tauV = Inu(230e9, 100, 10, 10, 60, L)
plt.loglog(L, I)
plt.loglog(L, tau)
plt.loglog(L, tauV)
plt.xlabel(f'Path length ($M$)')
```

```python
F, tau, tauV = Fnu(230e9, 100, 10, 10)
plt.imshow(F[:,:,0,0])
```

## Sanity Check

We know $n_e \sim 10^6\,\mathrm{cm}^{-3}$.
Check if this give reasonable magnetic field and flux.

```python
nu   = 230e9 # observe frequency
ne   = 1e6   # make a first guess...
Te   = 10    # electron tempearture in unit of electron rest mass energy
beta = 1

L, tau1, tauV1 = Lnu(nu, ne, Te, B(ne, Te, beta))
F, tau2, tauV2 = Fnu(nu, ne, Te, B(ne, Te, beta))

assert (tau1  == tau2 ).all()
assert (tauV1 == tauV2).all()

plt.imshow(F[:,:,0,0])

display(B(ne, Te, beta))
display(np.sum(L))
display(np.sum(F))
display(np.max(tau1))
display(np.max(tauV1))
```

## Solve the One-Zone Model

Really solve for $n_e$ using `scipy.optimize.root`.

```python
Fnu_obs = 2.4 # target flux in Jy

r  = root(lambda ne: np.sum(Fnu(nu, ne, Te, B(ne, Te, beta))[0]) - Fnu_obs, 1e6)
x0 = r.x[0]

display(r)
```

## Display the Solution

```python
ne = x0 # solution

display(nu)
display(ne)
display(Te)
display(B(ne, Te, beta))
display(np.sum(Lnu(nu, ne, Te, B(ne, Te, beta))[0]) * nu)
display(np.sum(Fnu(nu, ne, Te, B(ne, Te, beta))[0]))
display(np.max(Fnu(nu, ne, Te, B(ne, Te, beta))[1]))
```

```python
L, tau1, tauV1 = Lnu(nu, ne, Te, B(ne, Te, beta))
F, tau2, tauV2 = Fnu(nu, ne, Te, B(ne, Te, beta))

assert (tau1  == tau2 ).all()
assert (tauV1 == tauV2).all()

plt.imshow(F[:,:,0,0])

display(B(ne, Te, beta))
display(np.sum(L))
display(np.sum(F))
display(np.max(tau1))
display(np.max(tauV1))
```

## Sgr A* SED

Plot only the synchrotron SED for Sgr A*, assuming the electron number density is the 1/2 of the solved one, the solved one, and 2x the solved one.

```python
nu_obs = np.logspace(8,16,num=65)

fig, ax = plt.subplots(1,1,figsize=(8,8))

ax.set_xlim(1e8, 1e16)
ax.set_ylim(1e28,1e40)

ax.set_xlabel(r'Frequency $\nu$ [Hz]')
ax.set_ylabel(r'$\nu L_\nu$ [erg/s]')

nuLnu_obs = nu_obs * np.array([np.sum(Lnu(nu, ne/2, Te, B(ne/2, Te, beta))[0]) for nu in nu_obs])
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * np.array([np.sum(Lnu(nu, ne,   Te, B(ne,   Te, beta))[0]) for nu in nu_obs])
ax.loglog(nu_obs, nuLnu_obs)

nuLnu_obs = nu_obs * np.array([np.sum(Lnu(nu, ne*2, Te, B(ne*2, Te, beta))[0]) for nu in nu_obs])
ax.loglog(nu_obs, nuLnu_obs)
```

## Create Emissivity Plot

```python
nu = np.linspace(1,500e9)
F, tau, tauV = Fnu(nu, 1e6, 10, 30)
plt.plot(nu, F.sum(0).sum(0)[0])
plt.axhline(2.4)
```

## Create Contour Plots

```python
ne1 = np.logspace(4,10,num=61)
Te1 = np.logspace(0,3, num=61)

ne, Te = np.meshgrid(ne1, Te1)

F1, tau1, tauV1 = Fnu(230e9, ne, Te, B(ne, Te, 0.01))
F1tot    = F1.sum(0).sum(0)
tau1max  = tau1.max(0).max(0)
tauV1max = tauV1.max(0).max(0)

F2, tau2, tauV2 = Fnu(230e9, ne, Te, B(ne, Te, 1))
F2tot    = F2.sum(0).sum(0)
tau2max  = tau2.max(0).max(0)
tauV2max = tauV2.max(0).max(0)

F3, tau3, tauV3 = Fnu(230e9, ne, Te, B(ne, Te, 100))
F3tot    = F3.sum(0).sum(0)
tau3max  = tau3.max(0).max(0)
tauV3max = tauV3.max(0).max(0)
```

```python
def mklabels(CS, labels, color, start_nth=0, every_nth=1, offset=0):

    # get limits if they're automatic
    xmin,xmax,ymin,ymax = CS.axes.axis()
    
    # work with logarithms for loglog scale middle of the figure:
    logmid = (np.log10(xmin)+np.log10(xmax))/2, (np.log10(ymin)+np.log10(ymax))/2

    label_pos = []
    for path in CS.get_paths():
        if path.vertices.size > 0:
            logvert = np.log10(path.vertices)

            logvert2 = logvert.copy()
            logvert2[:,1] = (logvert2[:,1] - logmid[1])*1.5 + logmid[1] + offset
        
            # find closest point
            logdist = np.linalg.norm(logvert2-logmid, ord=2, axis=1)

            min_ind = np.argmin(logdist)
            label_pos.append(10**logvert[min_ind,:])

    # draw labels, hope for the best
    ax.clabel(CS, inline=True, inline_spacing=3, rightside_up=True, colors=color, fontsize=9,
              fmt={v:l for v,l in zip(CS.levels[start_nth::every_nth], labels[start_nth::every_nth])},
              manual=label_pos[start_nth::every_nth])
    
def mkpanel(ax, F, tau, tauV, beta, logmin=-6, start_nth=0):
    b = B(ne, Te, beta)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e0, 1e3)

    ax.contourf(ne1, Te1, tau,  levels=[0,1],      colors='g', alpha=0.2)
    ax.contourf(ne1, Te1, F,    levels=[2,3],      colors='k', alpha=0.2)
    ax.contourf(ne1, Te1, tauV, levels=[2*pi,1e8], colors='#0066ff', alpha=0.2)
    
    logrange = np.arange(int(np.floor(np.log10(np.min(b)))),10)
    levels = 10.0**logrange
    labels = [r"$10^{"+f"{l}"+r"}$G" for l in logrange]
    cs = ax.contour(ne1, Te1, b,
                    levels=levels,
                    colors='r', linewidths=0.5, linestyles='--')
    mklabels(cs, labels, 'r', offset=0.75)

    ax.set_xlabel(r"Electron number density $n_e$ [cm$^{-3}$]")
    ax.set_title(r"$\beta = "+f"{beta}"+"$")
    
    ax.text(1e6*beta**.3, 1e2, 'Optically Thin', color='g',       rotation=62)
    ax.text(2e8,          5,   'Total Flux',     color='k'                   )
    ax.text(6e7*beta**.3, 1e2, 'Faraday Thick',  color='#0066ff', rotation=62)
```

```python
fig, axes = plt.subplots(1,3, figsize=(10,3), sharey=True)
plt.subplots_adjust(wspace=0.1)

mkpanel(axes[0], F1tot, tau1max, tauV1max, 0.01, logmin=-5, start_nth=1)
mkpanel(axes[1], F2tot, tau2max, tauV2max, 1,    logmin=-6)
mkpanel(axes[2], F3tot, tau3max, tauV3max, 100,  logmin=-7, start_nth=1)

axes[0].set_ylabel(r'Electron temperature $\Theta_e$ [$m_e c^2 k_B^{-1}$]')

fig.savefig("onezonepol.pdf", bbox_inches='tight')
fig.savefig("onezonepol.png", dpi=300)
```
