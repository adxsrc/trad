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


"""Unit tests for `trad.plasma`"""


import pytest

from math import isclose
from astropy import units as u

from trad.plasma import gyrofrequency


@pytest.mark.parametrize(
    'mag,res', [
    (1   * u.T, 27992489872.33304 * u.Hz),
    (10  * u.T, 279924898723.3304 * u.Hz),
    (100 * u.T, 2799248987233.304 * u.Hz),
])
@pytest.mark.parametrize('u_mag', [u.T,  u.Gauss])
@pytest.mark.parametrize('u_res', [u.Hz, u.cycle/u.s, u.rad/u.s])
def test_gyrofrequency(u_mag, u_res, mag, res):

    mag = mag.to(u_mag)
    res = res.to(u_res, equivalencies=[(u.cy/u.s, u.Hz)])

    fe  = gyrofrequency(u_mag, u_res=u_res)

    ans = fe(mag.value)
    ref =    res.value

    print(f'{mag:8.2g} -> {res:8.2g}')

    assert isclose(ans, ref)
