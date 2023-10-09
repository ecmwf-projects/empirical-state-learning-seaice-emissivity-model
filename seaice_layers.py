#
# (C) Copyright 2023 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Contains the physical layers and observation loss function for the sea ice Bayesian network """

import xarray as xr
import tensorflow as tf

ifs_seaice_file='to_be_set'
ifs_tsfc_file='to_be_set'

obs_error = [2.5, 4.0, 2.5, 4.5, 2.5, 5.0, 4.0, 7.0, 4.5, 10.0]  

class IceProperty(tf.keras.layers.Layer):
    """
    Empirical properties of sea ice, representing ice and snow microstructure and other physical influences on 
    the surface emissivity. The layer weights contain maps of these properties as a function of grid point and
    timestep. Calling this layer takes the geolocation as input and returns the empirical properties at that 
    location in time and space. 
       Geolocation (0) is the igrid, (1) the istep
    """
    def __init__(self, nprop=1, ngrid=1, nstep=31):
        super(IceProperty, self).__init__()
        self.properties = self.add_weight(shape=(ngrid,nstep,nprop), initializer="zeros",  trainable=True)
    def call(self, geolocation):
        return tf.gather_nd(self.properties,geolocation)


class SeaiceEmis(tf.keras.layers.Layer):
    """
    Linear dense layer representing the sea ice emissivity empirical model.
    
    The sea ice loss applies to just the first mean emissivity (e.g. channel 10v); it's a single number as required.  
    """
    def __init__(self, channels=10, bg_error=0.1, nobs=1, background=0.93):
        super(SeaiceEmis, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(channels,activation='linear',bias_initializer=tf.keras.initializers.Constant(background))
        self.bg_error = bg_error
        self.background = background
        self.nobs = nobs
    def call(self, tsfc, ice_properties):
        inputs = tf.concat([tf.reshape(tsfc,(-1,1)),ice_properties],1)
        ice_emis = self.dense_1(inputs)
        emis_loss = tf.math.squared_difference((self.weights[1])[0],self.background)/tf.square(self.bg_error)/self.nobs
        self.add_loss(emis_loss) 
        self.add_metric(emis_loss,name='emis_loss',aggregation='mean')
        return ice_emis


class SeaiceEmisNN(tf.keras.layers.Layer):
    """
    Sea ice emissivity empirical model - dense multi-layer neural network version
    """
    def __init__(self, channels=10, nlayers=3, width=20, activation='relu'):
        super(SeaiceEmisNN, self).__init__()
        self.layers = list()
        for i in range(0,nlayers-1):
            self.layers.append(tf.keras.layers.Dense(width,activation=activation))
        self.layers.append(tf.keras.layers.Dense(channels,activation=activation)) 
    def call(self, tsfc, ice_properties):
        inputs = tf.concat([tf.reshape(tsfc,(-1,1)),ice_properties],1)
        out = self.layers[0](inputs)
        for i in range(1,len(self.layers)):    
            out = self.layers[i](out)
        return out


def seaice_initializer(shape, dtype=None):
    """
    Initializer for the sea ice fraction maps
    """
    ifs_seaice = xr.open_dataset(ifs_seaice_file)
    if hasattr(ifs_seaice,'SEAICE'):
        seaice_map = tf.convert_to_tensor(ifs_seaice.SEAICE, dtype=dtype)
    if hasattr(ifs_seaice,'seaice'):
        seaice_map = tf.convert_to_tensor(ifs_seaice.seaice, dtype=dtype)
    ifs_seaice.close()
    if tf.rank(seaice_map) == 2:
        return seaice_map
    else: 
        return tf.repeat(tf.reshape(seaice_map,(shape[0],1)),shape[1],1)


def tsfc_initializer(shape, dtype=None):
    """
    Initializer for the ocean surface temperature used in a sea ice fraction loss term
    """
    ifs_tsfc = xr.open_dataset(ifs_tsfc_file)
    tsfc_map = tf.convert_to_tensor(ifs_tsfc.TSFC, dtype=dtype)
    return tsfc_map


class OceanEmis(tf.keras.layers.Layer):
    """
    Ocean emissivity layer including a windspeed bias correction. Input vector is same as inputs (e.g. windspeed = 4)
    """
    def __init__(self, channels=10):
        super(OceanEmis, self).__init__()
        self.windspeed_bias = self.add_weight(shape=(channels), initializer="zeros", trainable=True)
        self.nchans=channels
    def call(self, windspeed, ocean_emis):
        return tf.tensordot(windspeed,self.windspeed_bias,0) + ocean_emis


class SeaiceFraction(tf.keras.layers.Layer):
    """
    Layer encapsulating the sea ice fraction maps, and returning whole-surface emissivity.
    
        Inputs to the call method are geolocation (igrid, istep), ocean emissivity (over channels) and 
        seaice emissivity (over channels). Returned is the whole-surface emissivity weighted according
        to the sea ice fraction.
        
        nlag is number of lagged sea-ice timesteps weighted by alpha, with 0=current time and N being prior times
        nlag = 0 means that sea-ice is not smoothed
    
    """
    def __init__(self, channels=10, ngrid=1, nstep=31, nlag=0, nobs=1, alpha=[1.0], bg_error=0.5, 
                 train=True, use_loss=True, use_pdf_loss=False, use_tsfc_loss=False):
        super(SeaiceFraction, self).__init__()
        self.seaice = self.add_weight(shape=(ngrid,nstep+nlag), initializer=seaice_initializer, trainable=train)
        if use_loss:    
            self.seaice_background = self.add_weight(shape=(ngrid,nstep+nlag), initializer=seaice_initializer, trainable=False) 
        if use_tsfc_loss:  
            self.tsfc = self.add_weight(shape=(ngrid,nstep), initializer=tsfc_initializer, trainable=False)   
        self.nchans = channels
        self.nlag = nlag
        self.nobs = nobs
        self.alpha = alpha
        self.bg_error = bg_error
        self.use_loss = use_loss
        self.use_pdf_loss = use_pdf_loss
        self.use_tsfc_loss = use_tsfc_loss
    def call(self, geolocation, emis_ocean, emis_seaice, tsfc):
        seaice_at_obs = self.alpha[0]*tf.gather_nd(self.seaice,tf.add(geolocation,[[0,self.nlag]]))
        for i in range(1,self.nlag+1):
            seaice_at_obs = seaice_at_obs + self.alpha[i]*tf.gather_nd(self.seaice,tf.add(geolocation,[[0,self.nlag-i]]))
        if self.use_loss:
            seaice_loss = tf.reduce_sum(tf.math.squared_difference(self.seaice,self.seaice_background)/(tf.square(self.bg_error)))/self.nobs
            self.add_loss(seaice_loss) 
            self.add_metric(seaice_loss,name='seaice_loss',aggregation='mean')
        if self.use_pdf_loss:
            seaice_loss = (tf.reduce_sum((tf.math.square(tf.math.maximum(self.seaice,1.0)-1.0)+
              tf.math.square(tf.math.maximum(-1*self.seaice,0.0)))/(tf.square(self.bg_error))))/self.nobs             
              #tf.reduce_mean(tf.where(seaice_at_obs > 0.01,1.0,0.0)*tf.math.maximum(tsfc-273.2,0.0)*4.0))
            self.add_loss(seaice_loss) 
            self.add_metric(seaice_loss,name='seaice_loss',aggregation='mean')
        if self.use_tsfc_loss:
            tsfc_loss = (tf.reduce_mean(tf.where(self.seaice[:,self.nlag:] > 0.01,1.0,0.0)*tf.math.maximum(self.tsfc-273.2,0.0)*4.0))/self.nobs
            self.add_loss(tsfc_loss) 
            self.add_metric(tsfc_loss,name='tsfc_loss',aggregation='mean')
        return (tf.tensordot(seaice_at_obs,tf.ones((self.nchans)),0) * emis_seaice +
           tf.tensordot(1 - seaice_at_obs,tf.ones((self.nchans)),0) * emis_ocean) , seaice_at_obs 

 
class Specular(tf.keras.layers.Layer):
    """
    Simple radiative transfer model (specular). 
    """
    def __init__(self, channels=10):
        super(Specular, self).__init__()
        self.nchans=channels
    def call(self,tsfc,emis,tausfc,tdown,tup):
        tsfc_inflated = tf.tensordot(tsfc,tf.ones((self.nchans)),0)
        # Emitted, reflected and atmospheric emission term
        return tsfc_inflated*emis*tausfc + (1-emis)*tausfc*tdown + tup


class CloudFractionDelta(tf.keras.layers.Layer):
    """
    Optional cloud fraction modification as a delta in observation space.  
    """
    def __init__(self, nobs):
        super(CloudFractionDelta, self).__init__()
        self.cloud_fraction_delta = self.add_weight(shape=(nobs), initializer=tf.zeros, trainable=True)
    def call(self,iobs):
        return tf.gather(self.cloud_fraction_delta,(iobs))


class BiasCorrection(tf.keras.layers.Layer):
    """
    Bias correction for TB output - with seaice fraction as the predictor 
    """
    def __init__(self, channels=10, background=[2.5,5.0], nobs=1, bg_error=1.0):
        super(BiasCorrection, self).__init__()
        self.instrument_bias_ocean  = self.add_weight(shape=(channels), initializer=tf.keras.initializers.Constant(background[0]), trainable=True)
        self.instrument_bias_seaice = self.add_weight(shape=(channels), initializer=tf.keras.initializers.Constant(background[1]), trainable=True)
        self.background = background
        self.nchans = channels
        self.nobs = nobs
        self.bg_error = bg_error
    def call(self, tb, seaice_fraction):
        bias = (tf.tensordot(seaice_fraction,self.instrument_bias_seaice,0) 
             + tf.tensordot(1 - seaice_fraction,self.instrument_bias_ocean,0))
        bias_loss = tf.reduce_sum(tf.math.squared_difference(self.instrument_bias_ocean,self.background[0])
                                  +tf.math.squared_difference(self.instrument_bias_seaice,self.background[1]))/tf.square(self.bg_error)/self.nobs
        self.add_loss(bias_loss) 
        self.add_metric(bias_loss,name='bias_loss',aggregation='mean')
        return tf.math.add(tb,bias)


@tf.function
def loss_channel_weighted(y_true, y_pred):
    """
    Loss equivalent to the data assimilation observation term, weighted as a function of observation error.
    Returned loss is a vector over the batch. The TF printed loss is a running average. The total TF 
    loss is sum(all obs&batches)/Nobs   
    """
    normdep = tf.math.divide(y_true - y_pred,obs_error)
    obs_loss = tf.reduce_sum(tf.square(normdep), axis=-1)
    return obs_loss 

