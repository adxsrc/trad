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


r"""Computing the spectral radiances or specific intensities.

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
from phun    import phun

from .helper import *


@phun({
    'si' : (u.W      ) / u.sr / (u.m *u.m ) / u.Hz,
    'cgs': (u.erg/u.s) / u.sr / (u.cm*u.cm) / u.Hz,
})
def blackbody(u_nu, u_T, u_res='si', backend=None):
    r"""An implementation of Planck's law.

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
    a = float((2 * c.h * u_nu**3) / (c.c**2 * u.sr) / u_res)**(1/3)
    x = float((c.h * u_nu) / (c.k_B * u_T))

    def pure(nu, T):
        return (a*nu)**3 / backend.expm1(x*nu/T)

    return pure


@phun({
    'si' : (u.W      ) / u.sr / (u.m *u.m ) / u.Hz,
    'cgs': (u.erg/u.s) / u.sr / (u.cm*u.cm) / u.Hz,
})
def specradiance(u_res='si', backend=None):
    r"""A solution of the radiative transfer equation.

    Using :math:`j_\nu` and :math:`\alpha_\nu` to denote the emission
    and absorption coefficients, the optical depth :math:`\tau_\nu` is
    defined as

    .. math::
        \tau_\nu = \int \alpha_\nu dl.

    The solution to the radiative transfer equation is

    .. math::
        I_\nu = I_\nu(0)\exp(-\tau_\nu) + S_\nu[1 - \exp(-\tau_\nu)],

    where the source function :math:`S_\nu` is

    .. math::
        S_\nu = \frac{j_\nu}{\alpha_\nu}.

    Assuming physical depth :math:`L` in a uniform thermal radiating
    media so that :math:`S_\nu = B_\nu` is constant in the media, the
    solution becomes

    .. math::
        \tau_\nu &= \alpha_\nu L, \\
        I_\nu    &= I_\nu(0)\exp(-\tau_\nu) + B_\nu[1 - \exp(-\tau_\nu)].

    """
    ...
