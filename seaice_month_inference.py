#
# (C) Copyright 2023 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" 
This is an example of how to re-build the trained model and its parameters for the month of August 2020 from 
one of the sensitivity tests. This illustration runs it in inference mode to demonstrate how the loading 
functions work. However, it is training and not inference (in Keras terms) that is the aim of the model, and the
loading functions were never used in the documented work, though they could have been useful to allow an existing 
model to be further trained (for example, getting around time resource limits on a supercomputer).

Note that the seaice_model "save" method currently discards any lagged sea ice fields outside the 
training period, so it is not currently possible to recreate exactly the ouputs of the trained model, at least
for the first N days of the training period, where N is the number of lagged sea ice fields. The "load" method, 
if required, re-creates these lagged fields by filling them with the sea ice concentration from 
the nearest day. A future upgrade should be to include the lagged part of the sea ice field in the "save" method, 
allowing exact replication of the trained outputs.
"""

import xarray as xr
import numpy as np
import tensorflow as tf
import seaice_model as sm

# base path for data
ice_path = '/sea_ice_data/'

# training data path and month idenitifier - training data from the sensitivity directory (tar file)
input_path=ice_path+'sensitivity/'
yearmon='202008'

# output data paths and test identifier
output_path=input_path+'nprop/'
filename_append='202008_nprop_02'

# Load training data
training_file = input_path+'amsr2_v2_'+yearmon+'.nc'
ngrid, nstep, nobs, x0, y0 = sm.monthly_training_data(training_file)

# Initialise the model with the constant skin temperature daily map,
# but also the trained, and not the initial, sea ice map
sm.seaice_layers.ifs_tsfc_file = input_path+'ifs_tsfc_dailyx_'+yearmon+'.nc'
sm.seaice_layers.ifs_seaice_file = output_path+'seaice_'+filename_append+'.nc'
seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, nprop = 3,
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])

# Initialise other model weights with their trained values
seaice_model.load(filename_append, output_path)

# Sey up a Keras model for inference        
model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)

# Calculate simulated AMSR2 TB
inferred_tb = model.predict(x0,batch_size = 1000000)

# Compute fit (RMSE) to observed AMSR2 TB
for channel in range(0,10):
    print(channel+1, np.sqrt(np.mean((inferred_tb[:,channel]-y0[:,channel])**2)))





