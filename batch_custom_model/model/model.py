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
    model.add(Conv2D(filters=8, input_shape=(28,28,1), kernel_size=(2,2), padding='valid'))
    model.add(Activation('relu'))
    #Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))

    #2nd Convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid'))
    model.add(Activation('relu'))

    #Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.summary()

    #Compile
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metric=['accuracy'])

    return model