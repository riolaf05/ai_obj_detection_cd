import tensorflow as tf
assert tf.__version__.startswith('2')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

#save to Saved Model
saved_model_dir = '/save'

#convert the saved model to a tf lite compatible format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('/save/model.tflite', 'wb') as f:
  f.write(tflite_model)