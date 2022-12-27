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


r"""Synchrotron Emissivity and Absorptivity

The terminology of emissivity (and absorptivity) are confusing.

Rybicki & Lightman (1986) defines the (monochromatic) emission
coefficient :math:`j` (:math:`j_\nu`) as energy emitted per unit time
per unit solid angle and per unit volume (per frequency).
Hence, the unit of :math:`j_\nu` in cgs is
:math:`\mathrm{erg}\,\mathrm{s}^{-1}\mathrm{sr}^{-1}\mathrm{cm}^{-3}\mathrm{Hz}^{-1}`.

However, Rybicki & Lightman (1986) also defines the emissivity
:math:`\epsilon_\nu` to be "energy emitted spontaneously per unit
frequency per unit time per unit mass", i.e., with cgs unit
:math:`\mathrm{erg}\,\mathrm{s}^{-1}\mathrm{g}^{-1}\mathrm{Hz}^{-1}`.

For isotropic emitters, this leads to

.. math::

    j_\nu = \frac{1}{4\pi}\epsilon_\nu \rho = \frac{1}{4\pi} P_\nu

where :math:`\rho` and :math:`P_\nu` are density and power density,
respectively.

In ``trad``, since we never need :math:`\epsilon_\nu`, we will use
interchangable "emission coefficient" and "emissivity" to refer to
:math:`j_\nu`.
This is consistent with the usage in astrophysics literature nowadays.

"""


from astropy import constants as c, units as u
from scipy.special import kn # jax does not support kn
from phun import phun

from .plasma       import Te, gyrofrequency
from .specradiance import blackbody


@phun({
    'si': u.W / u.sr / u.m**3 / u.Hz,
})
def emissivity(u_nu, u_ne, u_Thetae, u_B, u_theta, u_res='si', backend=None):
    r"""Synchrotron emissivity

    An approximation of the synchrotron emissivity at given
    frequency               :math:`\nu`,
    electron number density :math:`n_e`,
    electron temperature    :math:`T_e`,
    magnetic field strength :math:`B`, and
    magnetic pitch angle    :math:`\theta`,
    derived by Leung et al. (2011):

    .. math::
        j_\nu = n_e
        \frac{\sqrt{2}\pi e^2 \nu_\mathrm{s}}{3 K_2(1/\Theta_e)c}
        (X^{1/2} + 2^{11/12} X^{1/6})^2 \exp(-X^{1/3}),

    where
    :math:`\Theta_e = k_\mathrm{B}T_e / m_e c^2` is the dimensionless
    electron temperature,
    :math:`\nu_\mathrm{s} = (2/9)\nu_\mathrm{c}\Theta_e^2` is a
    synchrotron characteristic frequency, and
    :math:`X = \nu/\nu_\mathrm{s}`,
    with :math:`k_\mathrm{B}`, :math:`m_e`, :math:`e`, :math:`c`,
    :math:`K_2`, and :math:`\nu_\mathrm{c} = eB/2\pi m_e` being the
    Boltzmann constant, electron mass, electron charge, speed of
    light, modified Bessel function of the second kind of order 2, and
    the electron cyclotron frequency, respectively.

    """
    pi   = backend.pi
    sqrt = backend.sqrt
    exp  = backend.exp
    sin  = backend.sin
    nuc  = gyrofrequency(u_B)

    A = float(sqrt(2) * (pi/3) * (c.cgs.e.gauss**2/c.c/u.sr) * u_ne * nuc.unit / u_res)
    x = float(1 * u_nu / nuc.unit)

    def jnu_pure(nu, ne, Thetae, B, theta):
        nus = (2/9) * nuc(B) * Thetae*Thetae * sin(theta)
        X = x * nu / nus
        Y = (X**(1/2) + 2**(11/12) * X**(1/6))**2 * exp(-X**(1/3))
        K = kn(2, 1/Thetae)
        return A * (ne*nus) * (Y/K)

    return jnu_pure


@phun({
    'si': 1/u.m,
})
def absorptivity(u_nu, u_ne, u_Thetae, u_B, u_theta, u_res='si', backend=None):
    r"""Synchrotron absorptivity"""

    Bnu = blackbody(u_nu, Te)
    jnu = emissivity(u_nu, u_ne, u_Thetae, u_B, u_theta)

    return lambda nu, ne, Thetae, B, theta: jnu(nu, ne, Thetae, B, theta) / Bnu(nu, Thetae)
