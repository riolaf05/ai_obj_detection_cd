import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import argparse

BASE_DIR=r"C:\Users\lafacero\Documents\GitHub\ai_obj_detection_cd\real_time_object_detection_edge_tpu\object_detection_transfer_learning_tensorflow2\x86"

def representative_data_gen():
  for input_value, _ in test_batches.take(100):
    yield [input_value]

def main():
    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('--exported-model', type=str, help='Exported model folder name', default='saved_model')
    parser.add_argument('--model', type=str, help='Exported model name')
    args = parser.parse_args() 
    
    

if __name__ == "__main__":
    main()