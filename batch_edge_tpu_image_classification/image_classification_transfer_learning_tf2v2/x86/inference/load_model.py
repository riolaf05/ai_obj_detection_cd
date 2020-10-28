import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

BASE_DIR=r"C:\Users\lafacero\Documents\GitHub\ai_obj_detection_cd\real_time_object_detection_edge_tpu\object_detection_transfer_learning_tensorflow2\x86"

new_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'saved_model', '1602150252'))

# Check its architecture
new_model.summary()