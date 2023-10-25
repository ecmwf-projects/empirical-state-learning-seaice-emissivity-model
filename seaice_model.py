#
# (C) Copyright 2023 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Defines the sea ice Bayesian network as a Keras model, plus routines to read training data and save results"""

import tensorflow as tf
import numpy as np
import xarray as xr
import seaice_layers

class SeaiceModel:
    """
    Define a Keras model for the sea ice Bayesian network
    """
    def __init__(self, nchannels=10, nprop=3, ngrid=1, nstep=1, nobs=1, nlag=0, alpha=[1.0],
                 bg_error_seaice=0.2, bg_error_emis=1e-5, bg_error_bias=0.001, seaice_use_loss=True,
                 seaice_use_pdf_loss=False, seaice_use_tsfc_loss=False, adjust_cloud_fraction=False,
                 use_nn=False, nlayers_nn=2):
        """
        Initialize the network structure and internal and external dimensions
        """
        self.setup=dict()
        nfields = 7
        self.setup['nfields'] = nfields       
        self.setup['nchannels'] = nchannels
        self.setup['nprop'] = nprop
        self.setup['ngrid'] = ngrid
        self.setup['nstep'] = nstep
        self.setup['nlag']  = nlag
        self.setup['nobs']  = nobs 
        self.setup['alpha'] = alpha
        self.setup['bg_error_seaice'] = bg_error_seaice
        self.setup['bg_error_emis']   = bg_error_emis
        self.setup['bg_error_bias']   = bg_error_bias
        self.setup['adjust_cloud_fraction'] = adjust_cloud_fraction
        self.setup['seaice_use_tsfc_loss'] = seaice_use_tsfc_loss
        self.setup['use_nn'] = use_nn
        self.setup['nlayers_nn'] = nlayers_nn
                
        # TF model training is much faster if the inputs are a single tensor
        self.inputs = tf.keras.Input(shape=(nfields+nchannels*7,))

        # Split the input tensor into named variables
        tsfc_norm = self.inputs[:,0]
        geolocation = tf.cast(self.inputs[:,1:3],tf.int32)
        tsfc = self.inputs[:,3]
        windspeed = self.inputs[:,4]
        cloud_fraction = self.inputs[:,5]
        iobs = tf.cast(self.inputs[:,6],tf.int32)
        emis_ocean = self.inputs[:,nfields:nfields+nchannels]
        tausfc_clear = self.inputs[:,nfields+nchannels:nfields+2*nchannels]
        tdown_clear = self.inputs[:,nfields+2*nchannels:nfields+3*nchannels]
        tup_clear = self.inputs[:,nfields+3*nchannels:nfields+4*nchannels]
        tausfc_cloud = self.inputs[:,nfields+4*nchannels:nfields+5*nchannels]
        tdown_cloud = self.inputs[:,nfields+5*nchannels:nfields+6*nchannels]
        tup_cloud = self.inputs[:,nfields+6*nchannels:nfields+7*nchannels]

        # Define layer objects
        self.ice_prop_layer = seaice_layers.IceProperty(nprop, ngrid, nstep)
        if use_nn:
            self.seaice_emis_layer = seaice_layers.SeaiceEmisNN(nchannels,nlayers=nlayers_nn,width=10,activation='sigmoid')
        else:
            self.seaice_emis_layer = seaice_layers.SeaiceEmis(nchannels, nobs=nobs, bg_error=bg_error_emis)
        self.ocean_emis_layer = seaice_layers.OceanEmis(nchannels)
        self.seaice_layer = seaice_layers.SeaiceFraction(nchannels, ngrid, nstep,
                              nlag, nobs, alpha=alpha, bg_error=bg_error_seaice, 
                              use_loss=seaice_use_loss, use_pdf_loss=seaice_use_pdf_loss,
                              use_tsfc_loss=seaice_use_tsfc_loss)
        self.specular_clear_layer = seaice_layers.Specular(nchannels) 
        self.specular_cloud_layer = seaice_layers.Specular(nchannels)
        self.bias_layer = seaice_layers.BiasCorrection(nchannels, nobs=nobs, bg_error=bg_error_bias)
        self.cloud_fraction_delta = seaice_layers.CloudFractionDelta(nobs)

        if not use_nn:                
            print('Emis bg error',self.seaice_emis_layer.bg_error)
            print('Emis background',self.seaice_emis_layer.background)
        print('Seaice fraction bg error',self.seaice_layer.bg_error)
        print('Seaice fraction alpha',self.seaice_layer.alpha)
        print('Seaice fraction nobs',self.seaice_layer.nobs)
        print('Use seaice background loss',self.seaice_layer.use_loss)
        print('Use seaice PDF loss',self.seaice_layer.use_pdf_loss)
        print('Use seaice Tsfc loss',self.seaice_layer.use_tsfc_loss)
        print('Bias correction bg error',self.bias_layer.bg_error)
        print('Adjust cloud fraction',adjust_cloud_fraction)
                        
        # Plug together the model
        ice_prop = self.ice_prop_layer(geolocation)
        self.emis_seaice = self.seaice_emis_layer(tsfc_norm, ice_prop)
        emis_ocean_bc = self.ocean_emis_layer(windspeed, emis_ocean)
        emis_mixed, seaice_fraction = self.seaice_layer(geolocation, emis_ocean_bc, self.emis_seaice, tsfc)
        clear_tb = self.specular_clear_layer(tsfc, emis_mixed, tausfc_clear, tdown_clear, tup_clear)
        cloudy_tb = self.specular_cloud_layer(tsfc, emis_mixed, tausfc_cloud, tdown_cloud, tup_cloud)
        if adjust_cloud_fraction:
            self.adjusted_cloud_fraction = cloud_fraction + self.cloud_fraction_delta(iobs)
        else:
            self.adjusted_cloud_fraction = cloud_fraction
        allsky_tb = (     tf.tensordot(self.adjusted_cloud_fraction, tf.ones((nchannels)),0)  * cloudy_tb + 
                     (1 - tf.tensordot(self.adjusted_cloud_fraction, tf.ones((nchannels)),0)) * clear_tb)
        self.outputs = self.bias_layer(allsky_tb, seaice_fraction) 

    def save(self, history, fappend, outpath):
        """
        Save the sea ice model details to disk
        """
        grid=np.arange(self.setup['ngrid'])
        step=np.arange(self.setup['nstep'])  
        channel=np.arange(self.setup['nchannels'])
        biases=np.arange(2)
        prop=np.arange(self.setup['nprop'])
        epoch=np.arange(len(history.history['loss']))

        filename_append = fappend+'.nc'

        da1 = xr.DataArray(data=self.seaice_layer.seaice[:,self.setup['nlag']:],dims=("grid","step"),coords={"grid":grid,"step":step},name=('seaice'))
        da1.to_netcdf(outpath+'seaice_'+filename_append)

        da2 = dict()
        for key, value in self.setup.items():
            if key == 'alpha':
                da2[key] = xr.DataArray(data=value,dims=("alphas"),coords={"alphas":np.arange(len(value))},name=(key))
            else:
                da2[key] = xr.DataArray(data=[value],dims=("one"),coords={"one":[1]},name=(key))
        if not self.setup['use_nn']: 
            da2['ice_coefs'] = xr.DataArray(data=(self.seaice_emis_layer.weights[0])[:,:],dims=("in","channel"),
                            coords={"in":np.arange(self.setup['nprop']+1),"channel":channel},name=('ice_coefs'))
            da2['ice_emis'] = xr.DataArray(data=(self.seaice_emis_layer.weights[1])[:],dims=("channel"),
                            coords={"channel":channel},name=('ice_emis'))
        da2['ocean_emis_bias'] = xr.DataArray(data=(self.ocean_emis_layer.weights[0])[:],dims=("channel"),
                            coords={"channel":channel},name=('ocean_emis_bias'))
        da2['tb_bias'] = xr.DataArray(data=(tf.concat(tf.transpose(self.bias_layer.weights),0)),dims=("channel","biases"),
                            coords={"channel":channel,"biases":biases},name=('tb_bias'))
        for key, value in history.history.items():
            da2[key] = xr.DataArray(data=value,dims=("epoch"),coords={"epoch":epoch},name=(key))
        
        da2 = xr.Dataset(da2)
        da2.to_netcdf(outpath+'models_'+filename_append)

        da3 = xr.DataArray(data=self.ice_prop_layer.properties[:,:,:],dims=("grid","step","prop"),
            coords={"grid":grid,"step":step,"prop":prop},name=('properties'))
        da3.to_netcdf(outpath+'properties_'+filename_append)
  
    def load(self, filename_append, outpath):
        """
        Initialise the network trainable weights - to allow Keras training to be broken into 
        sequential calls. Note that seaice fields are loaded through the layer initialiser, not 
        through this routine
        """
        properties = xr.open_dataset(outpath+'properties_'+filename_append+'.nc')  
        self.ice_prop_layer.set_weights([properties.properties])

        models = xr.open_dataset(outpath+'models_'+filename_append+'.nc')  
        self.seaice_emis_layer.set_weights([models.ice_coefs,models.ice_emis])
        self.ocean_emis_layer.set_weights([models.ocean_emis_bias])
        self.bias_layer.set_weights([models.tb_bias[:,0],models.tb_bias[:,1]])

def monthly_training_data(filename, nsteps_per_day=1):
    """
    Load data for a single-month training of the sea ice network
    """
    obs = xr.open_dataset(filename)

    ngrid = obs.IGRID.data.max() + 1
    istep = np.floor(nsteps_per_day * (obs.JULIAN_DAY.data - np.floor(obs.JULIAN_DAY.data.min()) - 0.375))
    istep = istep.astype(int)
    nstep = istep.max() + 1
    nobs = obs.JULIAN_DAY.data.size  

    x0 = np.column_stack([np.maximum(273.0 - obs.TSFC,0.0)/30.0, obs.IGRID, istep, obs.TSFC, obs.WINDSPEED10M, obs.CLOUD_FRACTION, np.arange(nobs),
        obs.EMIS_WATER[:,4:], obs.TAUSFC[:,4:], obs.TDOWN[:,4:], obs.TUP[:,4:], obs.TAUSFC_CLD[:,4:], obs.TDOWN_CLD[:,4:], obs.TUP_CLD[:,4:]])

    y0 = np.column_stack([obs.OBSVALUE[:,4:]])
    
    return ngrid, nstep, nobs, x0, y0

def yearly_training_data(filebase, nsteps_per_day=1):
    """
    Load data for full year of training data for the sea ice network
    """
    fields = [
       'JULIAN_DAY','IGRID','TSFC','WINDSPEED10M','CLOUD_FRACTION','OBSVALUE',
       'EMIS_WATER','TAUSFC','TAUSFC_CLD','TUP','TUP_CLD','TDOWN','TDOWN_CLD'
       ]   
    obs = {}
    for i in range(0,13): 
       obs[fields[i]] = xr.open_dataset(filebase+fields[i]+'.nc')

    ngrid = obs["IGRID"].IGRID.data.max() + 1
    istep = np.floor(nsteps_per_day*(obs["JULIAN_DAY"].JULIAN_DAY.data - np.floor(obs["JULIAN_DAY"].JULIAN_DAY.data.min()) - 0.375))
    istep = istep.astype(int)
    nstep = istep.max() + 1
    nobs = obs["JULIAN_DAY"].JULIAN_DAY.data.size  

    x0 = np.column_stack([
        np.maximum(273.0 - obs["TSFC"].TSFC,0.0)/30.0, obs["IGRID"].IGRID, istep, obs["TSFC"].TSFC, 
        obs["WINDSPEED10M"].WINDSPEED10M, obs["CLOUD_FRACTION"].CLOUD_FRACTION, np.arange(nobs), obs["EMIS_WATER"].EMIS_WATER, 
        obs["TAUSFC"].TAUSFC, obs["TDOWN"].TDOWN, obs["TUP"].TUP, obs["TAUSFC_CLD"].TAUSFC_CLD, 
        obs["TDOWN_CLD"].TDOWN_CLD, obs["TUP_CLD"].TUP_CLD
        ])
    y0 = np.column_stack([obs["OBSVALUE"].OBSVALUE])
    
    return ngrid, nstep, nobs, x0, y0

def save_initial_tb(model, x0, fappend, outpath):
    """
    Compute and save initial (i.e. before training) TB outputs of the network as a diagnostic
    """
    model_tb = model.predict(x0,batch_size = 100000)
    
    iobs=np.arange((model_tb.shape)[0])
    channel=np.arange((model_tb.shape)[1])

    da4 = xr.DataArray(data=model_tb,dims=("iobs","channel"),coords={"iobs":iobs,"channel":channel},name=('tb'))
    da4.to_netcdf(outpath+'tbsim_initial_'+fappend+'.nc')

def save_outputs(model, seaice_model, x0, fappend, outpath):
    """
    Save model output (TB) and internal outputs (cloud fraction - not used normally; sea ice emissivity)
    """
    model_tb = model.predict(x0,batch_size = 100000)
    
    iobs=np.arange((model_tb.shape)[0])
    channel=np.arange((model_tb.shape)[1])

    da4 = xr.DataArray(data=model_tb,dims=("iobs","channel"),coords={"iobs":iobs,"channel":channel},name=('tb'))
    da4.to_netcdf(outpath+'tbsim_'+fappend+'.nc')

    model3 = tf.keras.Model(seaice_model.inputs, seaice_model.adjusted_cloud_fraction)    
    model3.compile()
    model_cloud_fraction = model3.predict(x0,batch_size = 100000)
    
    da6 = xr.DataArray(data=model_cloud_fraction,dims=("iobs"),coords={"iobs":iobs},name=('cloud_fraction'))
    da6.to_netcdf(outpath+'cloud_fraction_'+fappend+'.nc')
    
    model2 = tf.keras.Model(seaice_model.inputs, seaice_model.emis_seaice)    
    model2.compile()
    model_emis = model2.predict(x0,batch_size = 100000)
    
    da5 = xr.DataArray(data=model_emis,dims=("iobs","channel"),coords={"iobs":iobs,"channel":channel},name=('ice_emis'))
    da5.to_netcdf(outpath+'ice_emis_'+fappend+'.nc')
    

