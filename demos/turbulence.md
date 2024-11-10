---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# The Effect of Turbulence in Radiative Transfer Coefficients

This notebook contains a study on how radiative transfer coefficients (i.e., emissivity and absorbability) depend on turbulence. 

Turbulence is modeled by simply providing randomly generating velocity and magnetic fields according to power spectrum indices, although it should be straightforward to incorporate actual turbulence simulations. 

```python
import numpy  as np
from matplotlib import pyplot as plt
```

```python
def divless(N=256, p=5/3, seed=None):

    rng = np.random.default_rng(seed)

    r = np.linspace(0, 1, N, endpoint=False)
    k = np.fft.fftfreq(N, r[1])

    kx,ky,kz  = np.meshgrid(k,k,k)
    ik        = (1e-32 + kx*kx + ky*ky + kz*kz)**(-1/2)
    ik[0,0,0] = 0

    Ek        = ik**(p + (ik.ndim - 1))
    Ek[0,0,0] = 0
    Uk        = Ek**(1/2)
    
    # Randomize directions
    Ux = rng.normal(size=Ek.shape)
    Uy = rng.normal(size=Ek.shape)
    Uz = rng.normal(size=Ek.shape)

    # Randomize phase
    Ux = Ux * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))
    Uy = Uy * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))
    Uz = Uz * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))

    # Spectral shape
    Ux *= Uk
    Uy *= Uk
    Uz *= Uk

    # Obtain divergence-less field in physical domain
    ux = np.fft.ifftn(Ux, norm='forward').real
    uy = np.fft.ifftn(Uy, norm='forward').real
    uz = np.fft.ifftn(Uz, norm='forward').real

    return (ux, uy, uz), r, k
```

```python
(ux,uy,uz), r, k = divless()

x, y = np.meshgrid(r,r)
```

```python
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4,figsize=(16,4))

ax0.imshow(ux[:,:,0], vmin=-10, vmax=10, cmap='coolwarm')
ax0.set_aspect('equal')

ax1.imshow(uy[:,:,0], vmin=-10, vmax=10, cmap='coolwarm')
ax1.set_aspect('equal')

ax2.imshow(uz[:,:,0], vmin=-10, vmax=10, cmap='coolwarm')
ax2.set_aspect('equal')

ax3.quiver(x[::8,::8], y[::8,::8], ux[::8,::8,0], uy[::8,::8,0])
ax3.set_aspect('equal')
```
