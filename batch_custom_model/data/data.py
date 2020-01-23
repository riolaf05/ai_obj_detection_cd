from keras.datasets import fashion_mnist
from keras.utils import to_categorical
def get_data():
    #Put custom data retrieve function here!
    labels = ["T-shirt/top","Pantalone","Pullover","Vestito","Cappotto","Sandalo","Maglietta","Sneaker","Borsa","Stivaletto"]
    num_classes=10
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    #reshape
    trainX = trainX.reshape(trainX.shape[0],28*28) 
    testX = testX.reshape(testX.shape[0],28*28)
    #scale data to range [0,1]
    trainX=trainX/255
    testX=testX/255
    #one-shot encode the training and testing label
    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)
    return trainX, trainY, testX, testY

