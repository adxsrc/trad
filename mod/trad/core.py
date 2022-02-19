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
from numpy   import exp


def Bnu(nu, T):
    """Planck's law"""
    A = 2 * c.h  / (c.c*c.c) / u.sr
    x = (c.h*nu) / (c.k_B*T)
    B = A * (nu*nu*nu) / (exp(x) - 1)
    return B.to(u.W / u.sr / (u.m*u.m) / u.Hz)
