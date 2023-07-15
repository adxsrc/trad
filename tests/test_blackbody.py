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


from trad.specradiance import blackbody

from astropy import units as u


def test_gyrofrequency():

    B = blackbody(u.Hz, u.K)
    assert B(230e9, 1e10) == 1.6252775614257738e-07

    B = blackbody(u.GHz, u.K)
    assert B(230, 1e10) == 1.6252775614257677e-07
