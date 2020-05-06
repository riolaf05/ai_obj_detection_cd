import os
import argparse
from google.cloud import storage
import tensorflow as tf


def read_from_gcs(bucket, filename, path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.download_to_filename(os.path.join(path, filename))
    print("Data downloaded to ", os.path.join(path, filename))

def write_to_gcs(bucket, filename, path):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.upload_from_filename(os.path.join(path, filename))
    print(path+filename, " Model written to ", bucket, " bucket.")

def main():
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--input-model', type=str, help='Input model.') 
    parser.add_argument('--output-bucket', type=str, help='output bucket') 
    args = parser.parse_args()

    #read_from_gcs(args.input_bucket, "activity_classification.h5", "/")

    converter = tf.lite.TFLiteConverter.from_keras_model_file(args.input_model)
    tfmodel = converter.convert()
    file = open('/activity_classification.tflite', 'wb' ) 
    file.write(tfmodel)
    write_to_gcs(args.output_bucket, 'activity_classification.tflite', '/')

if __name__ == "__main__":
    main()
        
