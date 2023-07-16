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


from astropy import constants as c, units as u
from scipy.special import kn # jax does not support kn
from phun import phun

from ..plasma import u_T_me, gyrofrequency


@phun({
    'si' : (u.W      ) / u.sr / u.m**3  / u.Hz,
    'cgs': (u.erg/u.s) / u.sr / u.cm**3 / u.Hz,
})
def emissivity(u_nu, u_ne, u_Te, u_B, u_theta, u_res='si', backend=None):
    r"""Synchrotron emissivity

    An approximation of the synchrotron emissivity at given
    frequency               :math:`\nu`,
    electron number density :math:`n_e`,
    electron temperature    :math:`T_e`,
    magnetic field strength :math:`B`, and
    magnetic pitch angle    :math:`\theta`,
    derived by Dexter (2016):

    .. math::
        j_\nu = \frac{n_e e^2 \nu}{2\sqrt{3} c \Theta_e^2} I(X)

    where
    :math:`\Theta_e = k_\mathrm{B}T_e / m_e c^2` is the dimensionless
    electron temperature,
    :math:`X \equiv \nu / \nu_\mathrm{c}` is a scaled frequency, and
    :math:`\nu_\mathrm{c} = (3/2)\gamma^2\nu_B\sin\theta` is a
    synchrotron characteristic frequency, with :math:`k_\mathrm{B}`,
    :math:`m_e`, :math:`e`, :math:`c`, and :math:`\nu_B = eB/2\pi m_e`
    being the Boltzmann constant, electron mass, electron charge,
    speed of light, and the electron cyclotron frequency,
    respectively.

    Dexter (2016) also used the approximation:

    .. math::
        I(X) \approx
        2.5651(1 + 1.92 X^{-1/3} + 0.9977 X^{-2/3}) \exp(-1.8899 X^{1/3}).

    """
    sqrt = backend.sqrt
    exp  = backend.exp
    sin  = backend.sin
    tan  = backend.tan
    nuB  = gyrofrequency(u_B)

    r = float(u_theta.to(u.rad))
    t = float(u_T_me.to(u_Te))

    s1 = float(1.5 / t**2)
    s2 = float(t   / 0.75)
    A  = float((0.5 / sqrt(3)) * (c.cgs.e.gauss**2/c.c/u.sr) * u_ne * u_nu / u_res) * t**2
    x  = float(1 * u_nu / nuB.unit)

    def pure(nu, ne, Te, B, theta):
        nuc = s1 * Te**2 * nuB(B) * sin(theta * r)
        X   = x * nu / nuc
        s3  = A * (ne * nu/Te**2) * exp(-1.8899*X**(1/3))
        II  = 2.5651 * (1 + 1.92 *X**(-1/3) + 0.9977*X**(-2/3))
        IQ  = 2.5651 * (1 + 0.932*X**(-1/3) + 0.4998*X**(-2/3))
        IV  = (1.8138*X**(-1) + 3.423*X**(-2/3) + 0.02955*X**(-1/2) + 2.0377*X**(-1/3))
        return (
            s3 * II,
            s3 * IQ,
            s3 * IV * s2 / (Te * tan(theta * r))
        )

    return pure
