#
# (C) Copyright 2024 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" 
This creates a special purpose model to reconstruct the sea ice emissivities at grid locations, based on the
trained model, year-long maps of skin temperature and the empirical properties on the grid. Note that the sea ice
emissivity is usually generated at the observation locations and has not been stored because of memory limitations.
Hence emissivity is not otherwise available. This could have been done equally with the full model infrastructure 
(using methods from seaice_model.py) but it is done manually here for illustrative purposes and to make it easy to
create a more packaged dataset of the surface emissivity model.
"""

import xarray as xr
import numpy as np
import tensorflow as tf
import seaice_layers

# base path for data
ice_path = '/sea_ice_data/'

# identifier in filenames
yearmon = 'year'

# The trained sea ice empirical properties will define the grid (time points and day steps)
properties = xr.open_dataset(ice_path+'properties_'+yearmon+'.nc') 

# Skin temperature from the IFS at matching grid locations and times
tsfc = xr.open_dataset(ice_path+'ifs_tsfc_'+yearmon+'_dailyx.nc')

# Trained model coefficients
models = xr.open_dataset(ice_path+'models_'+yearmon+'.nc') 

# Sea ice concentration, as a diagnostic to be stored with emissivity
seaice = xr.open_dataset(ice_path+'seaice_'+yearmon+'.nc') 

nchannels = 10
nprop = properties.prop.size
ngrid = properties.grid.size
nstep = properties.step.size

# Create "observations" at each grid and time point
igrid = np.outer(np.ones(properties.step.size),properties.grid)
istep = np.outer(properties.step,np.ones(properties.grid.size))

nobs = igrid.size

# Create a model for the purposes of generating the sea ice emissivity only
nfields = 3
inputs = tf.keras.Input(shape=(nfields,))

# Split out the input fields for readability
tsfc_norm = inputs[:,0]
geolocation = tf.cast(inputs[:,1:3],tf.int32)

# Layers
ice_prop_layer = seaice_layers.IceProperty(nprop, ngrid, nstep)
seaice_emis_layer = seaice_layers.SeaiceEmis(nchannels)

# Link up the model
ice_prop = ice_prop_layer(geolocation)
outputs = seaice_emis_layer(tsfc_norm, ice_prop)
model = tf.keras.Model(inputs, outputs)

# Initialise with trained weights 
ice_prop_layer.set_weights([properties.properties])
seaice_emis_layer.set_weights([models.ice_coefs,models.ice_emis])

# Inputs from the grid, flattened as if a stream of observations
x0 = np.column_stack([np.maximum(273.0 - tsfc.TSFC.data.transpose().flatten(),0.0)/30.0, 
  igrid.flatten(), istep.flatten()])

# Inference
inferred_emis = model.predict(x0,batch_size = 1000000)

# Store output and combine with other useful data to make it more self-describing
# Note also the transpose to put the step/time index as the slowest dimension (different convention in other netCDFs)
channel=np.arange(nchannels)
dataset = xr.Dataset(coords={'step':properties.step, 'grid':properties.grid, 
  'channel':channel, 'prop':properties.prop, 'inputs':np.arange(4)})
dataset['lon'] = (['grid'],tsfc.LON.data)
dataset['lat'] = (['grid'],tsfc.LAT.data)
dataset['time_start'] = (['step'],
  np.arange(np.datetime64('2020-05-31T21:00'), np.datetime64('2021-05-31T21:00'), np.timedelta64(24, 'h')))
dataset['time_end']   = (['step'],
  np.arange(np.datetime64('2020-06-01T21:00'), np.datetime64('2021-06-01T21:00'), np.timedelta64(24, 'h')))
dataset['channel_name'] = ('channel',['10v','10h','19v','19h','24v','24h','37v','37h','89v','89h'])
dataset['sea_ice_concentration'] = seaice.seaice.transpose('step','grid')
dataset['tsfc_norm'] = (['step','grid'],x0[:,0].reshape([nstep,ngrid]).astype(np.float32))
dataset['properties'] = properties.properties.transpose('step','grid','prop')
dataset['sea_ice_emissivity'] = (['step','grid','channel'], inferred_emis.reshape([nstep,ngrid,nchannels]))
dataset['ice_emis'] = ('channel',models.ice_emis.data)
dataset['ice_coefs'] = (['inputs','channel'],models.ice_coefs.data)
dataset.to_netcdf(ice_path+'emissivity_grid_'+yearmon+'.nc')




