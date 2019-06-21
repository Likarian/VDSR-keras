#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers.convolutional import Conv2D
from keras.layers.merge import Add
from keras.layers import Input, Activation
from keras.models import Model

from keras import backend as K
K.set_image_data_format('channels_last')

def VDSR_origin( input_shape = (600, 480, 3) ):
    low_resolution_image = Input( shape= input_shape )

    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(low_resolution_image)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu')(processing)
    processing = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(processing)
    Residual = processing

    high_resolution_image = Add() ([low_resolution_image, Residual])
    
    model = Model(low_resolution_image, high_resolution_image)
    return model