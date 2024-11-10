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

    r = np.linspace(0, 1, N, endpoint=False)
    k = np.fft.fftfreq(N, r[1])

    kx,ky,kz  = np.meshgrid(k,k,k)
    ik        = (1e-32 + kx*kx + ky*ky + kz*kz)**(-1/2)
    ik[0,0,0] = 0

    Ek        = ik**(p + (ik.ndim - 1))
    Ek[0,0,0] = 0
    Uk        = Ek**(1/2)
```
