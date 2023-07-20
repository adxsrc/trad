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


"""Solution to Radiative Transfer Problems"""


from astropy import constants as c, units as u
from phun    import phun

from .helper import *
from .sync   import coefficients


@phun({
    'si' : (u.W      ) / u.sr / (u.m *u.m ) / u.Hz,
    'cgs': (u.erg/u.s) / u.sr / (u.cm*u.cm) / u.Hz,
})
def constant(u_nu, u_ne, u_Te, u_B, u_theta, u_L, u_I, u_res='si', backend=None, pol=False):
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

    Cnu = coefficients(u_nu, u_ne, u_Te, u_B, u_theta, u_res=u_res, pol=pol)
    s1  = float(Cnu.unit[0] / Cnu.unit[1] / u_res)
    s2  = float(u_I / u_res)
    s3  = float(Cnu.unit[1] * u_L)

    def pure(nu, ne, Te, B, theta, L, I=0): # closure on `pol`
        C    = Cnu(nu, ne, Te, B, theta)
        j, a = (C[0][0], C[1][0]) if pol else (C[0], C[1])
        S    = s1 * j / a
        I0   = s2 * I
        tau  = s3 * a * L
        tauV = s3 * C[1][4] * L
        return S + (I0 - S) * backend.exp(-tau), tau, tauV

    return pure
