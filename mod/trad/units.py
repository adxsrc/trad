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


"""Helper functions related to unit handling."""


from astropy import units as u


def merge(keys, args, kwargs):
    """Merge arguments with keyworded arguments."""

    if len(args) > len(keys):
        raise ValueError(
            f'Factory expects {len(keys)} argument(s) but {len(args)} were given.')

    d = {k:a for k, a in zip(keys, args)}
    return {**kwargs, **d}


def arg_unit(name, default, kwargs):
    """Deduce argument units from ``kwarg``."""

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


def ret_unit(ud, default, kwargs):
    """Deduce return units from ``kwarg``."""

    r = kwargs.get('units', default)
    if isinstance(r, u.UnitBase):
        return r
    elif r in ud:
        return ud[r]
    else:
        raise u.UnitConversionError(f'{r} is not supported by `trad`')
