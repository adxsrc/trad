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


"""Unit tests for `trad.blackbody`"""


import pytest

from math import isclose
from astropy import units as u

from trad.specradiance import blackbody


@pytest.mark.parametrize(
    'freq,T,res', [
    (230e9 * u.Hz, 1e10 * u.K, 1.6252775792033076e-07 * u.W / (u.Hz * u.sr * u.m**2)),
])
@pytest.mark.parametrize('u_freq', [u.Hz, u.MHz, u.GHz])
@pytest.mark.parametrize('u_T',    [u.K])
@pytest.mark.parametrize('u_res',  [u.W / (u.Hz * u.sr * u.m**2), u.erg / (u.s * u.Hz * u.sr * u.m**2)])
def test_blackbody(u_freq, u_T, u_res, freq, T, res):

    freq = freq.to(u_freq)
    T    = T   .to(u_T, equivalencies=u.temperature_energy())
    res  = res.to(u_res)

    B    = blackbody(u_freq, u_T, u_res=u_res)

    ans  = B(freq.value, T.value)
    ref  =   res .value

    print(f'{freq:8.2e}, {T:8.2e} -> {res:8.2e}')

    assert isclose(ans, ref)
