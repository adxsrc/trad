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


"""Dexter (2016) Formulation"""


from astropy import constants as c, units as u
from scipy.special import kn # jax does not support kn
from phun import phun

from ..plasma import u_T_me, gyrofrequency
from ..specradiance import blackbody


@phun({
    'si' : ((u.W      ) / u.sr / u.m**3  / u.Hz, u.m **-1),
    'cgs': ((u.erg/u.s) / u.sr / u.cm**3 / u.Hz, u.cm**-1),
})
def coefficients(u_nu, u_ne, u_Te, u_B, u_theta, u_res='si', backend=None, pol=False):
    r"""Synchrotron emissivity, absorptivity, and Faraday coefficients

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
    log = backend.log
    sin = backend.sin
    cos = backend.cos
    grf = gyrofrequency(u_B)
    bb  = blackbody(u_nu, u_Te)

    r = float(u_theta.to(u.rad))
    t = float(u_T_me.to(u_Te))

    z0 = float(2 * u_nu / grf.unit)
    A1 = float((c.cgs.e.gauss**2/c.c/u.sr) * u_ne * u_nu / u_res[0]) * t**2 * (0.5 / 3**0.5)
    A2 = float((u_ne * c.cgs.e.gauss**2 * grf.unit**2) / (c.m_e * c.c * u_nu**3) / u_res[1])
    f1 = 3 / (t**2 * z0)
    f2 = 1.5e-3 * 2**-0.5
    f3 = t / 0.75

    def pure(nu, ne, Te, B, theta):

        sint = sin(theta * r)
        cost = cos(theta * r)
        nuB  = grf(B)

        ix   = f1 * Te**2 * nuB * sint / nu
        ix16 = ix**(1/6)
        ix13 = ix16*ix16
        ix12 = ix13*ix16
        ix23 = ix13*ix13

        f5  = A1 * (ne * nu / Te**2) * exp(-1.8899/ix13)
        fI  = 2.5651 + 4.9250*ix13 + 2.5592*ix23
        fQ  = 2.5651 + 2.3907*ix13 + 1.2820*ix23
        fV  = 1.8138*ix + 3.4230*ix23 + 0.02955*ix12 + 2.0377*ix13

        jI = f5 * fI
        jQ = f5 * fQ
        jV = f5 * fV * f3 * cost / (Te * sint)

        iB = 1 / bb(nu, Te)

        # rhoQ and rhoV are really provided by Shcherbakov (2008)

        X = (f2 / ix)**-0.5
        f = 2.011 * exp(-X**1.035/4.7) - cos(X/2) * exp(-X**1.2/2.73) - 0.011 * exp(-X/47.2)
        g = 1 - 0.11 * log(1 + 0.035 * X)

        iTheta = t / Te

        K0 = kn(0, iTheta)
        K1 = kn(1, iTheta)
        K2 = kn(2, iTheta)

        rhoQ = A2 * (ne/nu) * (sint * nuB/nu)**2 * f * (K1/K2 + 6/iTheta)
        rhoV = A2 * (ne/nu) * (cost * nuB/nu)    * g * (K0/K2) * z0

        if pol:
            return (
                (jI,    jQ,    jV),
                (jI*iB, jQ*iB, jV*iB, rhoQ, rhoV),
            )
        else:
            return (jI, jI*iB)

    return pure
