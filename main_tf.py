from models_tf import PointNet_TF, train, test
from preprocessing import preprocess_tf

import numpy as np
import random

import tensorflow as tf
# import tensorflow.keras
import math
# from tf import nn, optim

def getSphereConeCubeData(test_size = 0.2):
    
    shapes, labels = preprocess_tf('sphereConeCubeData.csv', 3)
    trainX = shapes[:math.floor(len(shapes)*(1-test_size))]
    trainY = labels[:math.floor(len(labels)*(1-test_size))]
    print(trainX.shape, trainY.shape)

    testX = shapes[math.floor(len(shapes)*(1-test_size)):]
    testY = labels[math.floor(len(labels)*(1-test_size)):]

    return trainX, trainY, testX, testY

def testNN_tf_2():
    test_size = 0.2
    num_epoch = 1
    
    trainX, trainY, testX, testY = getSphereConeCubeData()

    model = PointNet_TF(num_classes=3)

    train(model, trainX, trainY, num_epoch)
    loss_train = test(model, trainX, trainY)
    loss_test = test(model, testX, testY)

    print('Average Training Loss:', loss_train)
    print('Average Testing Loss:', loss_test)
    

def testNN_tf():
    test_size = 0.2
    num_epoch = 1
    BATCH_SIZE = 128

    trainX, trainY, testX, testY = getSphereConeCubeData(test_size)

    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))

    train_dataset = train_dataset.shuffle(len(trainX)).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(testX)).batch(BATCH_SIZE)

    # print(train_dataset)

    model = PointNet_TF(num_classes=3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    
    model.fit(train_dataset, epochs=num_epoch, validation_data=test_dataset)
    model.summary()
    preds = model.predict(test_dataset)

    print(model.accuracy(preds, testY))

def main():
    testNN_tf()

if __name__ == "__main__":
    main()