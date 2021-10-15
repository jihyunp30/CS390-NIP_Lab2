import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#LGORITHM = "tf_conv"

DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    EPOCH = 10
    CNN_EPOCH = 10
    DROPRATE = .2
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
    NUM_NEURON = 256
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    EPOCH = 10
    CNN_EPOCH = 10
    DROPRATE = .2
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
    NUM_NEURON = 256
elif DATASET == "cifar_10":
    # TODO: Add this case.
    NUM_CLASSES = 20
    EPOCH = 10
    CNN_EPOCH = 10
    DROPRATE = .3
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    NUM_NEURON = 256
elif DATASET == "cifar_100_f":
    # TODO: Add this case.
    NUM_CLASSES = 100
    EPOCH = 30
    CNN_EPOCH = 10
    DROPRATE = .3
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    NUM_NEURON = 256
elif DATASET == "cifar_100_c":
    # TODO: Add this case.
    NUM_CLASSES = 20
    EPOCH = 30
    CNN_EPOCH = 20
    DROPRATE = .3
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    NUM_NEURON = 256


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = EPOCH):
    # TODO: Implement a standard ANN here.
    model = tf.keras.Sequential()
    if DATASET == "mnist_d" or DATASET == "mnist_f":
        model.add(tf.keras.layers.Dense(NUM_NEURON, activation='relu'))
        model.add(tf.keras.layers.Dense(NUM_NEURON, activation='relu'))
        model.add(tf.keras.layers.Dense(NUM_NEURON, activation='relu'))
    else:
        model.add(tf.keras.layers.Dense(1536, activation='relu'))
        model.add(tf.keras.layers.Dense(768, activation='relu'))
        model.add(tf.keras.layers.Dense(384, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy)
    model.fit(x, y, epochs=eps)
    return model


def buildTFConvNet(x, y, eps = CNN_EPOCH, dropout = True, dropRate = DROPRATE):
    #TODO: Implement a CNN here. dropout option is required.
    model = tf.keras.Sequential()
    if DATASET == "mnist_d" or DATASET == "mnist_f":
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation ='relu',
                                     input_shape=[IH, IW, IZ]))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    else:
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation ='relu',
                                     input_shape=[IH, IW, IZ]))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    model.add(tf.keras.layers.Flatten())
    if dropout:
        model.add(tf.keras.layers.Dropout(dropRate))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy)
    model.fit(x, y, epochs=eps)

    return model

# =========================<Pipeline Functions>==================================


def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # TODO: Add this case.
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        # TODO: Add this case.
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
