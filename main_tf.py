from models_tf import PointNet_TF
from preprocessing import preprocess_tf

import numpy as np
import random

import tensorflow as tf
# import tensorflow.keras
import math
# from tf import nn, optim

def getTrainTestValData(fileLoc, test_size = 0.15, val_size = 0.15):
    shapes, labels = preprocess_tf(fileLoc, 3)
    trainX = shapes[:math.floor(len(shapes)*(1-test_size-val_size))]
    trainY = labels[:math.floor(len(labels)*(1-test_size-val_size))]

    valX = shapes[math.floor(len(shapes)*(1-test_size-val_size)) : math.floor(len(shapes*(1-test_size)))]
    valY = labels[math.floor(len(labels)*(1-test_size-val_size)) : math.floor(len(labels*(1-test_size)))]

    # print(trainX.shape, trainY.shape)

    testX = shapes[math.floor(len(shapes)*(1-test_size)):]
    testY = labels[math.floor(len(labels)*(1-test_size)):]

    return trainX, trainY, valX, valY, testX, testY
    
def testNN_tf():
    # test_size = 0.2
    num_epoch = 6
    BATCH_SIZE = 32

    primitiveFile = 'sphereConeCubeData.csv'
    snowFile = 'snowDataset.csv'

    trainX, trainY, valX, valY, testX, testY = getTrainTestValData(snowFile)

    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    val_dataset = tf.data.Dataset.from_tensor_slices((valX, valY))
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))

    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # print(train_dataset)

    model = PointNet_TF(num_classes=3)
    # model.build((BATCH_SIZE, 500,3))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    # model.summary()
    
    model.fit(train_dataset, epochs=num_epoch, validation_data=val_dataset)

    preds = model.predict(test_dataset.batch(2))

    print(model.accuracy(preds, testY))

def main():
    testNN_tf()

if __name__ == "__main__":
    main()