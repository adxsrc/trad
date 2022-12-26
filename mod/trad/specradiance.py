# Copyright 2022 Chi-kwan Chan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""Spectral Radiances or Specific Intensities

"Intensity" is the historically name of "radiance" but in astronomy we
are stuck with it.

Spectral radiance, or specific intensity, in frequency or wavelength
are often denoted as :math:`I_\nu` or :math:`I_\lambda`.
They are sometime referred as "brightness" and has the generic symbol
:math:`B`.

The SI unit of radiance is
:math:`\mathrm{W}\,\mathrm{sr}^{-1}\mathrm{m}^{-2}`.
Hence, the SI and cgs unit of spectral radiance in frequency are
:math:`\mathrm{W}\,\mathrm{sr}^{-1}\mathrm{m}^{-2}\mathrm{Hz}^{-1}` and
:math:`\mathrm{erg}\,\mathrm{s}^{-1}\mathrm{sr}^{-1}\mathrm{m}^{-2}\mathrm{Hz}^{-1}`,
respectively.

"""


from astropy import constants as c, units as u


def blackbody(nu_unit=u.Hz, T_unit=u.K, unit=u.W/u.sr/u.m**2/u.Hz, backend=None):
    r"""Planck's law

    Spectral density of electromagnetic radiation emitted by a black
    body in thermal equilibrium at a given temperature :math:`T` at
    frequency :math:`\nu`,

    .. math::
        B_\nu(T) = \frac{A\nu^3}{e^x - 1},

    where
    :math:`A = 2h/c^2\mathrm{sr}` and
    :math:`x = h\nu/k_\mathrm{B}T`,
    with :math:`h`, :math:`c`, and :math:`k_\mathrm{B}` being the
    Planck's constant, speed of light, and Boltzmann constant,
    respectively.

    """

    if backend is None:
        import sys
        if 'jax' in sys.modules:
            import jax.numpy as backend
        else:
            import numpy as backend
    exp = backend.exp

    A_v = float((2 * c.h * nu_unit**3) / (c.c**2 * u.sr) / unit)
    x_v = float((c.h * nu_unit) / (c.k_B * T_unit))
    def B(nu, T):
        return A_v * nu**3 / (exp(x_v * nu/T) - 1)

    B.unit = unit
    return B
