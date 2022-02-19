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


from astropy import units as u
from astropy import constants as c
from numpy   import pi, sqrt, exp, sin

from scipy.special import kn


def Bnu(nu, T):
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
    A = 2 * c.h  / (c.c*c.c) / u.sr
    x = (c.h*nu) / (c.k_B*T)
    B = A * (nu*nu*nu) / (exp(x) - 1)
    return B.to(u.W / u.sr / (u.m*u.m) / u.Hz)


def jsyncnu(nu, ne, Thetae, B, theta):
    r"""Synchrotron emissivity

    An approximation of the synchrotron emissivity at given
    frequency               :math:`\nu`,
    electron number density :math:`n_e`,
    electron temperature    :math:`\Theta_e`,
    magnetic field strength :math:`B`, and
    magnetic pitch angle    :math:`\theta`,
    derived by Leung et al. (2011):

    .. math::
        j_\nu = n_e
        \frac{\sqrt{2}\pi e^2 \nu_\mathrm{s}}{3 K_2(1/\Theta_e)c}
        (X^{1/2} + 2^{11/12} X^{1/6})^2 \exp(-X^{1/3}),

    where
    :math:`\nu_\mathrm{s} = (2/9)\nu_\mathrm{c}\Theta_e^2` is a
    synchrotron characteristic frequency and
    :math:`X = \nu/\nu_\mathrm{s}`,
    with :math:`e`, :math:`c`, :math:`K_2`, and :math:`\nu_\mathrm{c}
    = eB/2\pi m_e` being the electron charge, speed of light, modified
    Bessel function of the second kind of order 2, and the electron
    cyclotron frequency, respectively.

    """
    nuc = (c.si.e * B / (2 * pi * c.m_e)).to(u.Hz) # electron cyclotron frequency
    nus = (2/9) * nuc * Thetae*Thetae * sin(theta) # synchrotron characteristic frequency

    A = sqrt(2) * (pi/3) * (c.cgs.e.gauss**2 / c.c) / u.sr
    X = nu / nus
    Y = (X**(1/2) + 2**(11/12) * X**(1/6))**2 * exp(-X**(1/3))
    K = kn(2, 1/Thetae)

    J = A * (ne*nus) * (Y/K)
    return J.to(u.W / u.sr / u.m**3 / u.Hz)
