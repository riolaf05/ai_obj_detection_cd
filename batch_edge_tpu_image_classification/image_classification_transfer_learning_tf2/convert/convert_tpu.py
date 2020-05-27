#Vedi: https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only
import os
import numpy as np
import argparse
import tensorflow as tf
assert tf.__version__.startswith('2')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, help='dimensione di input del modello', default=224, required=True)
    args = parser.parse_args()

    #get Saved Model dir
    saved_model_dir = '/save/fine_tuning'

    #convert the saved model to a tf lite compatible format
    num_calibration_steps=60 #TODO: cos'è?
    input_shape=np.array([  1, args.image_size, args.image_size,   3])

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
      for _ in range(num_calibration_steps):
        # Get sample input data as a numpy array in a method of your choosing:
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        yield [input_data]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()

    with open('/save/model.tflite', 'wb') as f:
      f.write(tflite_quant_model)

