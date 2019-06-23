#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import load_model
from keras import metrics
import tensorflow as tf
import numpy as np
from utils import load_test, PSNR
import scipy.misc
import argparse

parser = argparse.ArgumentParser(description='Test function')
parser.add_argument('--test', metavar='test', type=str, 
                    help='test directory')
parser.add_argument('--network', metavar='network', type=str, 
                    help='network weight')
args = parser.parse_args()

val_in, val_out = load_test(directory = args.test)
model = load_model(args.network)

prediction = model.predict(val_in, batch_size = 1, verbose = 1)

Result = PSNR(val_out, prediction)
sess = tf.Session()
RR = sess.run(Result)

print(RR)

for img_count in range(prediction.shape[0]):
    img_in = val_in[img_count,:,:,:]
    img_out = prediction[img_count,:,:,:]
    img_gt = val_out[img_count,:,:,:]
    scipy.misc.imsave('./Result/LR'+'{0:03d}'.format(img_count)+'.png', img_in)
    scipy.misc.imsave('./Result/HR'+'{0:03d}'.format(img_count)+'.png', img_out)
    scipy.misc.imsave('./Result/GT'+'{0:03d}'.format(img_count)+'.png', img_gt)
    
