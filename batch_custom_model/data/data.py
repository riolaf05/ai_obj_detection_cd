from keras.datasets import fashion_mnist
from keras.utils import np_utils
def get_data():
    #Put custom data retrieve function here!
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    #reshape
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    #scale data to range [0,1]
    trainX=trainX.astype("float32") / 255.0
    testX=testX.astype("float32") / 255.0
    #one-shot encode the training and testing label
    trainX = np_utils.to_categorical(trainY, 10)
    testX = np_utils.to_categorical(testY, 10)
    return trainX, trainY, testX, testY

