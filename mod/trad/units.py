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


"""Units Helper Functions"""


from astropy import units as u


def arg_unit(name, default, kwargs):
    """Deduce Argument Units from `kwarg`"""

    unit = name+'_unit' # unit name

    wn = name in kwargs.keys() # with name
    wu = unit in kwargs.keys() # with unit

    if wn and wu:
        raise NameError(f'{name} and {unit} cannot be specified simultaneously')
    elif wn:
        a = kwargs[name]
        if isinstance(a, u.UnitBase):
            return a
        else:
            return a.unit
    elif wu:
        return kwargs[unit]
    else:
        return default
