from os
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
import sys
sys.path.append('model/')
from model import model
import json
import mlflow

def train(trainX, trainY, model):
    print('Loading params from json..')
    with open('training_conf.json') as f:
    data = json.oad(f)

    print('Current data params are: ')
    print(data)
    
    model.fit(trainX, trainY, batch_size=data['batch_size'], epochs=data['epochs'], verbose=data['verbose'], validation_split=0.3, shuffle=data['shuffle'], callbacks=[history])
    return model
