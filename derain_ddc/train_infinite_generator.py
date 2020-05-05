# coding: utf-8
'''
    - train "ZF_UNET_224" CNN with random images
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
from zf_unet_224_model import *
from keras import losses
image_list = []

# def addNoise(self, image, var):
#     return skimage.util.random_noise(image, var=0.005)

import data_generator as dg
import tensorflow as tf
import loss_msssim

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1))))

def SSIM(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def train_unet():
    out_model_path = 'temp.h5'
    epochs = 10
    patience = 20
    batch_size = 12
    optim_type = 'Adam'
    learning_rate = 0.001
    model = ZF_UNET_224()
    #if os.path.isfile(out_model_path):
        #model.load_weights(out_model_path)
	#print "load previous model"


    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=loss_msssim.MS_SSIM_l1_loss, metrics=[PSNR, SSIM] )

    callbacks = [

        ModelCheckpoint('temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=dg.batch_generator(batch_size),
        epochs=epochs,
        steps_per_epoch=100,
        validation_data=dg.batch_generator(batch_size),
        validation_steps=100,
        verbose=2,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('train.csv', index=False)
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')


if __name__ == '__main__':

    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())
    train_unet()
