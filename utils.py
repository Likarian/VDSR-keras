#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.callbacks import CSVLogger
import tensorflow as tf
from keras import backend as K

import glob
import scipy.misc

import csv
import numpy as np

from collections import OrderedDict
from collections import Iterable
from datetime import datetime

class CSVLoggerTimestamp(CSVLogger):
	def __init__(self, filename, separator=',', append=False ):
		super(CSVLoggerTimestamp, self).__init__(filename, separator, append)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		def handle_value(k):
			is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
			if isinstance(k, Iterable) and not is_zero_dim_ndarray:
				return '"[%s]"' % (', '.join(map(str, k)))
			else:
				return k

		if not self.writer:
			self.keys = sorted(logs.keys())

			class CustomDialect(csv.excel):
				delimiter = self.sep

			self.writer = csv.DictWriter(self.csv_file, fieldnames=['timestamp'] + ['epoch'] + self.keys , dialect=CustomDialect)

			if self.append_header:
				self.writer.writeheader()

		timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
		row_dict = OrderedDict({'timestamp': timestamp})
		row_dict.update({'epoch': epoch})
		row_dict.update((key, handle_value(logs[key])) for key in self.keys)
		self.writer.writerow(row_dict)
		self.csv_file.flush()


def load_test(directory):
	ImageList = sorted(glob.glob(directory+'/*'))
	LR = []
	HR = []
	for Image in ImageList:
		BaseImage = scipy.misc.imread(Image)
		Height, Width = BaseImage.shape[0], BaseImage.shape[1]
		HighResolutionImage = BaseImage.copy()
		ResizedBaseImage = scipy.misc.imresize(BaseImage, 1/3, interp='bicubic')
		LowResolutionImage = scipy.misc.imresize(ResizedBaseImage, (Height, Width), interp='bicubic')

		LR.append(LowResolutionImage)
		HR.append(HighResolutionImage)

	val_in = np.asarray(LR, dtype=np.float32)
	val_out = np.asarray(HR, dtype=np.float32)

	return val_in, val_out

def tf_log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator

def PSNR(y_true, y_pred):
	max_pixel = 255
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
