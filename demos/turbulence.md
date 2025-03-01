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

# The Effect of Turbulence in Radiative Transfer Coefficients

This notebook contains a study on how radiative transfer coefficients (i.e., emissivity and absorbability) depend on turbulence. 

Turbulence is modeled by simply providing randomly generating velocity and magnetic fields according to power spectrum indices, although it should be straightforward to incorporate actual turbulence simulations. 

```python
try:
    %load_ext autoreload
    %autoreload 2
except ModuleNotFoundError as e:
    print(e)
```

```python
import numpy  as np
from matplotlib import pyplot as plt
```

```python
def fieldgen(
    N=256,     # Number of grid points per dimension
    p=5/3,     # Energy spectrum is Ek ~ k^{-p}
    a=1,       # degree of divergenless; a=1 divergenless field; a=0 potential field
    seed=None, # random seed
    iso=True,  # force isotropic spectrum by removing "corners" of the spectrum
):
    rng = np.random.default_rng(seed)

    r = np.linspace(0, 1, N, endpoint=False)
    k = np.fft.fftfreq(N, r[1])

    kx,ky,kz  = np.meshgrid(k,k,k)
    kk        = kx*kx + ky*ky + kz*kz

    ik        = kk**(-1/2)
    ik[0,0,0] = 0
    if iso:
        kmax = np.max(k)
        ik[kk > kmax*kmax] = 0

    Ek        = ik**(p + (ik.ndim - 1))
    Ek[0,0,0] = 0
    
    # Randomize directions
    Ux = rng.normal(size=Ek.shape)
    Uy = rng.normal(size=Ek.shape)
    Uz = rng.normal(size=Ek.shape)

    # Divergence-less condition
    f  = (Ux*kx + Uy*ky + Uz*kz) * (ik*ik) * (1 - 2 * a)
    Ux = a * Ux + f * kx
    Uy = a * Uy + f * ky
    Uz = a * Uz + f * kz

    # Spectral shape
    Uk  = Ek**(1/2)
    Ux *= Uk
    Uy *= Uk
    Uz *= Uk

    # Normalization
    E   = np.sum(Ux * Ux + Uy * Uy + Uz * Uz) / 2
    Ux /= E
    Uy /= E
    Uz /= E

    # Randomize phase
    Ux = Ux * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))
    Uy = Uy * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))
    Uz = Uz * np.exp(1j*rng.uniform(0, 2*np.pi, size=Ek.shape))

    # Obtain divergence-less field in physical domain
    ux = np.fft.ifftn(Ux, norm='forward').real
    uy = np.fft.ifftn(Uy, norm='forward').real
    uz = np.fft.ifftn(Uz, norm='forward').real

    return (ux, uy, uz), r, k
```

```python
def mkplot(ux, uy, uz, r, vmax=np.sqrt(0.5)):
    x, y = np.meshgrid(r,r)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4,figsize=(16,4))
    
    ax0.imshow(ux[:,:,0], vmin=-vmax, vmax=vmax, cmap='coolwarm', origin='lower')
    ax0.set_aspect('equal')
    
    ax1.imshow(uy[:,:,0], vmin=-vmax, vmax=vmax, cmap='coolwarm', origin='lower')
    ax1.set_aspect('equal')
    
    ax2.imshow(uz[:,:,0], vmin=-vmax, vmax=vmax, cmap='coolwarm', origin='lower')
    ax2.set_aspect('equal')
    
    ax3.quiver(x[::8,::8], y[::8,::8], ux[::8,::8,0], uy[::8,::8,0])
    ax3.set_aspect('equal')
```

```python
(ux,uy,uz), r, k = fieldgen(a=0)
(bx,by,bz), r, k = fieldgen(a=1)
```

```python
mkplot(ux, uy, uz, r)
```

```python
mkplot(bx, by, bz, r)
```

```python
mkplot(bx+0.1, by, bz, r)
```

```python
mkplot(bx+0.2, by, bz, r)
```

```python
mkplot(bx+0.3, by, bz, r)
```
