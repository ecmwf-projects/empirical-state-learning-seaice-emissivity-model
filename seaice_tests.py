#
# (C) Copyright 2023 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Top level control of the sea ice network parameter estimation: sensitivity tests using a month of data """

import os
import xarray as xr
import numpy as np
import tensorflow as tf
import seaice_model as sm
import time

# This is the top level directory location for all input and output data
ice_path='/sea_ice_data/'

# ECMWF HPC specific config. 
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

output_path = ice_path + 'sensitivity/'

batchsize = 1024
nepochs = 20

yearmon='202008'
training_file = ice_path+'amsr2_v2_'+yearmon+'.nc'
ngrid, nstep, nobs, x0, y0 = sm.monthly_training_data(training_file)

sm.seaice_layers.ifs_seaice_file = ice_path+'ifs_seaice_'+yearmon+'.nc'
sm.seaice_layers.ifs_tsfc_file = ice_path+'ifs_tsfc_dailyx_'+yearmon+'.nc'

# Get reproducible results
tf.random.set_seed(8757390)

# nproperties
nepochs = 20

def new_test(name):
    test_path = output_path+name+'/'
    os.makedirs(test_path, exist_ok="True")
    return test_path

test_path = new_test('nprop')
nprop = [1,2,3,4,5,7,10]
for i in range(0,len(nprop)):
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, nprop = nprop[i],
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs)
    filename_append = yearmon+'_nprop_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)
 
nepochs = 50

# Nonlinear and deep networks to model emissivity
test_path = new_test('deep')
nlayers = [1,2,5,10,0]
for i in range(0,len(nlayers)): 
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, use_nn=(nlayers[i] > 0), nlayers_nn=nlayers[i],
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs)
    filename_append = yearmon+'_deep_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)

nepochs = 20
   
# Emis background error (1e-5 is used)
test_path = new_test('bemis')
bemis = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
for i in range(0,len(bemis)):
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, bg_error_emis=bemis[i],
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs)
    filename_append = yearmon+'_bemis_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)
    

# Seaice background error (0.02 is used)
test_path = new_test('bseaice')
bseaice = [20.0,2.0,0.2,0.02,0.002,0.0002]
for i in range(0,len(bseaice)): 
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, bg_error_seaice=bseaice[i],
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs)
    filename_append = yearmon+'_bseaice_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)


# TB bias background error (0.001 is used)
test_path = new_test('bbias')
bbias = [10.0,1.0,0.1,0.01,0.001,0.0001,0.00001]
for i in range(0,len(bbias)): 
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs, bg_error_bias=bbias[i],
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs)
    filename_append = yearmon+'_bbias_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)


# Number of epochs - continuing training the same model
test_path = new_test('epoch')
nepochs_set = [1,1,1,2,5,10,10,10,20,40,200]
seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs,
    seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
    bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs) 
model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
for i in range(0,len(nepochs_set)):
    history = model.fit(x0, y0, batch_size=batchsize, epochs = nepochs_set[i])
    if i == 0:
        hh = history
    else:
        for item in hh.history:
            hh.history[item] = hh.history[item] + history.history[item]
    filename_append = yearmon+'_epoch_'+str(i).zfill(2)
    seaice_model.save(hh, filename_append, test_path)


# Batch size
test_path = new_test('batchsize')
batchsize_set = [16384,4096,1024,256,32]
for i in range(0,len(batchsize_set)): 
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs,
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    tstart = time.process_time()
    history = model.fit(x0, y0, batch_size=batchsize_set[i], epochs = nepochs)
    print('TTEST',time.process_time() - tstart)
    filename_append = yearmon+'_batchsize_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)

# Batch size - more epochs
nepochs_vary = [200,60]
test_path = new_test('bbatchsize')
batchsize_set = [16384,4096]
for i in range(0,len(batchsize_set)): 
    seaice_model = sm.SeaiceModel( ngrid=ngrid, nstep=nstep, nobs=nobs,
        seaice_use_pdf_loss=True, seaice_use_loss=False, seaice_use_tsfc_loss=True,
        bg_error_seaice=0.02, nlag=1, alpha=[0.6,0.4])
    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)  
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted)
    tstart = time.process_time()
    history = model.fit(x0, y0, batch_size=batchsize_set[i], epochs = nepochs_vary[i])
    print('TTEST',time.process_time() - tstart)
    filename_append = yearmon+'_bbatchsize_'+str(i).zfill(2)
    seaice_model.save(history, filename_append, test_path)
