from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation 
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


def model(opt):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=trainX.shape[1]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu')) 
    model.add(Dense(10, activation='softmax'))
    model.summary()

    #Compile
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
