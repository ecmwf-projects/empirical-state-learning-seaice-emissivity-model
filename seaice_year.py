#
# (C) Copyright 2023 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Top level control of the sea ice network parameter estimation using Keras and tensorflow, using year-long training """

import xarray as xr
import numpy as np
import tensorflow as tf
import seaice_model as sm

# This is the top level directory location for all input and output data
ice_path='/sea_ice_data/'

# ECMWF HPC specific config. 
import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

batchsize = 1024
tag = '_final'

training_filebase=ice_path+'field_v2_'
ngrid, nstep, nobs, x0, y0 = sm.yearly_training_data(training_filebase)

# initials_year is IFS monthly mean sea-ice expanded onto 366 days (inc. one "lag")
sm.seaice_layers.ifs_seaice_file = ice_path+'ifs_seaice_initials_year.nc'
# year_dailyx is the daily surface temperature on the grid
sm.seaice_layers.ifs_tsfc_file = ice_path+'ifs_tsfc_year_dailyx.nc'
seaice_model = sm.SeaiceModel(ngrid=ngrid,nstep=nstep,nobs=nobs,seaice_use_pdf_loss=True, 
  seaice_use_loss=False, seaice_use_tsfc_loss=True,
  bg_error_seaice=0.02, adjust_cloud_fraction=False, nlag=1, alpha=[0.6,0.4])

# Attempt to get reproducible results
tf.random.set_seed(409782)

model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)
model.summary()    
model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)

yearmon="year" 
filename_append = yearmon+tag

sm.save_initial_tb(model, x0, filename_append, ice_path)

# Running fit multiple times consumes memory (4xfit = memkill) but being able to inspect the outputs during the fit is handy.
nepochs = [1,2,5]
for i in range(0,len(nepochs)):
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs[i],verbose=2)
    if i == 0:
        hh = history
    else:
        for item in hh.history:
            hh.history[item] = hh.history[item] + history.history[item]
    seaice_model.save(hh, filename_append, ice_path)

sm.save_outputs(model, seaice_model, x0, filename_append, ice_path)


