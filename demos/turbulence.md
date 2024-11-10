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
