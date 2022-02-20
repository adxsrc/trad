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


"""Absorptivities"""


from astropy import constants as c, units as u

from .specradiance import Bnu
from .emissivity   import jsyncnu


def asyncnu(nu, ne, Thetae, B, theta):
    """Synchrotron Absorptivities"""
    T = (Thetae * c.m_e * c.c**2 / c.k_B).to(u.K)
    a = jsyncnu(nu, ne, Thetae, B, theta) / Bnu(nu, T)
    return a.to(1/u.m)
