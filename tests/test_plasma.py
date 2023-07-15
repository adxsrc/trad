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


from math import isclose
from astropy import units as u

from trad.plasma import gyrofrequency


def test_gyrofrequency():

    fe = gyrofrequency(u.T)
    assert isclose(fe(1.0), 27992489872.33304)

    fe = gyrofrequency(u.T, u_res=u.rad/u.s)
    assert isclose(fe(1.0), 175882001077.2163)
