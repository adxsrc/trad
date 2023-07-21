# Copyright 2022,2023 Chi-kwan Chan
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


r"""Computing the synchrotron emission, bsorption, and Faraday coefficients.

The terminology of emissivity (and absorptivity) are confusing.
Rybicki & Lightman (1986) defines the (monochromatic) emission
coefficient :math:`j` (:math:`j_\nu`) as energy emitted per unit time
per unit solid angle and per unit volume (per frequency).
Hence, the unit of :math:`j_\nu` in cgs is
:math:`\mathrm{erg}\,\mathrm{s}^{-1}\mathrm{sr}^{-1}\mathrm{cm}^{-3}\mathrm{Hz}^{-1}`.
However, Rybicki & Lightman (1986) also defines the emissivity
:math:`\epsilon_\nu` to be "energy emitted spontaneously per unit
frequency per unit time per unit mass", i.e., with cgs unit
:math:`\mathrm{erg}\,\mathrm{s}^{-1}\mathrm{g}^{-1}\mathrm{Hz}^{-1}`.

For isotropic emitters, this leads to

.. math::

    j_\nu = \frac{1}{4\pi}\epsilon_\nu \rho = \frac{1}{4\pi} P_\nu

where :math:`\rho` and :math:`P_\nu` are density and power density,
respectively.

In ``trad``, since we never need :math:`\epsilon_\nu`, we will use
interchangable "emission coefficient" and "emissivity" to refer to
:math:`j_\nu`.
This is consistent with the usage in astrophysics literature nowadays.

"""


from .Dexter2016 import coefficients # TODO: dynamically choose which implementation
