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
from ..specradiance import blackbody


@phun({
    'si' : (u.W      ) / u.sr / u.m**3  / u.Hz,
    'cgs': (u.erg/u.s) / u.sr / u.cm**3 / u.Hz,
})
def emissivity(u_nu, u_ne, u_Te, u_B, u_theta, u_res='si', backend=None, pol=False):
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
    exp = backend.exp
    sin = backend.sin
    tan = backend.tan
    nuB = gyrofrequency(u_B)

    r = float(u_theta.to(u.rad))
    t = float(u_T_me.to(u_Te))

    s1 = float(1.5 / t**2)
    s2 = float(t   / 0.75)
    A  = float((0.5 / 3**0.5) * (c.cgs.e.gauss**2/c.c/u.sr) * u_ne * u_nu / u_res) * t**2
    x  = float(1 * u_nu / nuB.unit)

    def pure(nu, ne, Te, B, theta):
        nuc  = s1 * Te**2 * nuB(B) * sin(theta * r)
        iX   = nuc / (x * nu)
        iX13 = iX**(1/3)
        iX23 = iX13*iX13
        s3  = A * (ne * nu/Te**2) * exp(-1.8899/iX13)
        fI  = 2.5651 + 4.9250*iX13 + 2.5592*iX23
        return s3 * fI

    def purepol(nu, ne, Te, B, theta):
        nuc  = s1 * Te**2 * nuB(B) * sin(theta * r)
        iX   = nuc / (x * nu)
        iX16 = iX**(1/6)
        iX13 = iX16*iX16
        iX12 = iX13*iX16
        iX23 = iX13*iX13
        s3  = A * (ne * nu/Te**2) * exp(-1.8899/iX13)
        fI  = 2.5651 + 4.9250*iX13 + 2.5592*iX23
        fQ  = 2.5651 + 2.3907*iX13 + 1.2820*iX23
        fV  = 1.8138*iX + 3.4230*iX23 + 0.02955*iX12 + 2.0377*iX13
        return (
            s3 * fI,
            s3 * fQ,
            s3 * fV * s2 / (Te * tan(theta * r)),
        )

    return purepol if pol else pure


@phun({
    'si' : 1/u.m,
    'cgs': 1/u.cm,
})
def absorptivity(u_nu, u_ne, u_Te, u_B, u_theta, u_res='si', backend=None, pol=False):
    r"""Synchrotron absorptivity"""

    Bnu = blackbody(u_nu, u_Te)
    jnu = emissivity(u_nu, u_ne, u_Te, u_B, u_theta, pol=pol)

    if not pol:
        def pure(nu, ne, Te, B, theta):
            j = jnu(nu, ne, Te, B, theta)
            B = Bnu(nu, Te)
            return j / B
        return pure

    exp = backend.exp
    log = backend.log
    sin = backend.sin
    cos = backend.cos
    nuB = gyrofrequency(u_B)

    r = float(u_theta.to(u.rad))
    t = float(u_T_me.to(u_Te))

    s1 = float(1.5 / t**2)
    A  = float((u_ne * c.cgs.e.gauss**2 * nuB.unit**2) / (c.m_e * c.c * u_nu**3) / u_res)

    def purepol(nu, ne, Te, B, theta):
        j = jnu(nu, ne, Te, B, theta)
        B = Bnu(nu, Te)

        nuc = s1 * Te**2 * nuB(B) * sin(theta * r)
        X = (1.5e-3/2**(1/2) * nu/nuc)**(-1/2)
        f = 2.011 * exp(-X**1.035/4.7) - cos(X/2) * exp(-X**1.2/2.73) - 0.011 * exp(-X/47.2)
        g = 1 - 0.11 * log(1 + 0.035 * X)

        T  = Te * u_Te / u_T_me
        iT = 1 / T

        K0 = kn(0, t/Te)
        K1 = kn(1, t/Te)
        K2 = kn(2, t/Te)

        rhoQ =     A * (ne / nu**3) * sin(theta * r)**2 * f * (K0 / K2 + 6 * T)
        rhoV = 2 * A * (ne / nu**2) * cos(theta * r)    * g * (K1 / K2)

        return tuple(jS / B for jS in j) + (rhoQ, rhoV)

    return purepol
