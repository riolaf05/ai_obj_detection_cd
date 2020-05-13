#!/usr/local/bin/python3	
'''
This scripts is used to retrieve first and last layer of a Saved Model Tensorflow Graph
'''
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(description='Getting Frozen Graph .pb file')
parser.add_argument('--frozen-graph', help='frozen graph file')
args = parser.parse_args()

Graph = tf.GraphDef()   
File = open(args.frozen_graph,"rb")
Graph.ParseFromString(File.read())

os.environ['FIRST_LAYER'] = Graph.node[0].name #first layer
os.environ['FIRST_LAYER'] = Graph.node[-1].name #last layer??