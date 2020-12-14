import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

feature_extractor_url = r"https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

# Convert the model.
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('1602079553.h5')
tflite_model = converter.convert()

# Save the model.
with open('1602079553.tflite', 'wb') as f:
  f.write(tflite_model)