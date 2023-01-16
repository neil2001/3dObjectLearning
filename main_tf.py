from models_tf import PointNet_TF
from preprocessing import preprocess_tf

import numpy as np
import random

import tensorflow as tf
# import tensorflow.keras
import math
# from tf import nn, optim

from sklearn.metrics import confusion_matrix
import seaborn as sns

from shapeGeneration import printShape

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
    
def testNN_tf(snow=True):
    # test_size = 0.2
    num_epoch = 8
    BATCH_SIZE = 32

    primitiveFile = 'data/sphereConeCubeData.csv'
    snowFile = 'data/snowDataset.csv'

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

    snowModel = 'snowModel/snowModel'
    shapeModel = 'shapeModel/shapeModel'
    modelName = snowModel if snow else shapeModel

    model.save_weights(modelName)

    preds = model.predict(test_dataset.batch(2))

    print(model.accuracy(preds, testY))

def visualizeMistakes(pred, real, data):
    assert(len(pred) == len(real))
    n = len(pred)
    for i in range(n):
        predClass = str(tf.keras.backend.get_value(pred[i]))
        realClass = str(real[i])
        if predClass != realClass:
            printShape(data[i], "images/confusion/pred_" + predClass + "_real_" + realClass + "_" + str(i) + ".png", realClass + " predicted as " + predClass)
    return

def visualizeModel(snow=True):
    modelName = 'snowModel/snowModel' if snow else 'shapeModel/shapeModel'
    # model = tf.keras.model.load_model(modelName)
    model = PointNet_TF(num_classes=3)
    load_status = model.load_weights(modelName)
    load_status.expect_partial()
    fileName = 'data/snowDataset.csv' if snow else 'data/sphereConeCubeData.csv'
    _, _, _, _, testX, testY = getTrainTestValData(fileName, test_size = 0.5)
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))

    logits = model.predict(test_dataset.batch(2))
    print("accuracy", model.accuracy(logits, testY).numpy())
    preds = tf.argmax(logits, 1)
    # print(preds)

    visualizeMistakes(preds, testY, testX)

    confMat = confusion_matrix(testY, preds)
    confMatPlot = sns.heatmap(confMat, annot=True)
    figure = confMatPlot.get_figure()    
    figure.savefig('images/conf_mat_' + modelName.split('/')[-1] +'.png', dpi=400)

def main():
    testNN_tf()
    visualizeModel()

if __name__ == "__main__":
    main()