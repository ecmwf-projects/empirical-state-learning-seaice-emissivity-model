# empirical-state-learning-seaice-emissivity-model

Python code for training a hybrid empirical-physical network to represent passive microwave observations over ocean and sea ice areas. The unknowns, to be found by training, are the sea ice concentration, the physical properties of the sea ice and any overlying snow, and the microwave surface emissivity of the sea ice surface. These unknown state and modelling components are embedded in a network of known physical models.

The learning approach is a hybrid of data assimilation and machine learning that simultaneously trains an empirical model component and an empirical geophysical input state. The  empirical model component is a simple neural network and its input creates a latent space that defines the empirical geophysical state. It is proposed to call this an "empirical state" method.

The empirical sea ice emissivity model trained using this code can then be plugged into a weather forecasting system to add a sea ice concentration analysis to the atmospheric data assimilation system, and to enable assimilation of microwave data with strong surface sensitivities over sea ice for the first time

The code used in the draft manuscript is archived at https://doi.org/10.5281/zenodo.10013542

The data used in the draft manuscript is archived at https://doi.org/10.5281/zenodo.10009498

Citation:

Simultaneous inference of sea ice state and surface emissivity model using machine learning and data assimilation, A.J. Geer, 2023 (in preparation)

Further context:

Joint estimation of sea ice and atmospheric state from microwave imagers in operational weather forecasting, A.J. Geer, 2023 (in preparation)

## Dependencies  

tensorflow
xarray
numpy

## License

Copyright 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor
does it submit to any jurisdiction.


