from models_tf import PointNet_TF, train, test
from preprocessing import preprocess_tf

import numpy as np
import random

import tensorflow as tf
import math
# from tf import nn, optim

def testNN_tf():
    test_size = 0.2
    num_epoch = 1
    
    shapes, labels = preprocess_tf('sphereConeCubeData.csv', 3)
    trainX = shapes[:math.floor(len(shapes)*(1-test_size))]
    trainY = labels[:math.floor(len(labels)*(1-test_size))]
    print(trainX.shape, trainY.shape)

    testX = shapes[math.floor(len(shapes)*(1-test_size)):]
    testY = labels[math.floor(len(labels)*(1-test_size)):]

    model = PointNet_TF(num_classes=3)

    train(model, trainX, trainY, num_epoch)
    loss_train = test(model, trainX, trainY)
    loss_test = test(model, testX, testY)

    print('Average Training Loss:', loss_train)
    print('Average Testing Loss:', loss_test)
    

def main():
    testNN_tf()

if __name__ == "__main__":
    main()