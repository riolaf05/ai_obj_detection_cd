from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense

#putting data to arrays for RNN
data_array=contents
y_train=responses

#turning str responses to int 
y_train = [ int(x) for x in y_train ]
print(len(y_train))
print(type(y_train[0]))