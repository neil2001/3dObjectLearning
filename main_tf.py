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
import sys

def getTrainTestValData(fileLoc, test_size = 0.15, val_size = 0.15):
    '''
    outputs the train, val, and test data
    fileLoc: the location of the data file
    test_size: the proportion of data used for testing
    val_size: the proportion of data used for validation
    '''
    shapes, labels = preprocess_tf(fileLoc)
    trainX = shapes[:math.floor(len(shapes)*(1-test_size-val_size))]
    trainY = labels[:math.floor(len(labels)*(1-test_size-val_size))]

    valX = shapes[math.floor(len(shapes)*(1-test_size-val_size)) : math.floor(len(shapes*(1-test_size)))]
    valY = labels[math.floor(len(labels)*(1-test_size-val_size)) : math.floor(len(labels*(1-test_size)))]

    # print(trainX.shape, trainY.shape)

    testX = shapes[math.floor(len(shapes)*(1-test_size)):]
    testY = labels[math.floor(len(labels)*(1-test_size)):]

    return trainX, trainY, valX, valY, testX, testY
    
def testNN_tf(snow=True):
    '''
    trains the model and saves it
    '''
    classes = 3 if snow else 4
    # test_size = 0.2
    num_epoch = 6
    BATCH_SIZE = 128

    primitiveFile = 'data/fourShapeData.csv'
    snowFile = 'data/snowDataset.csv'

    fileName = snowFile if snow else primitiveFile

    trainX, trainY, valX, valY, testX, testY = getTrainTestValData(fileName)

    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
    val_dataset = tf.data.Dataset.from_tensor_slices((valX, valY))
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))

    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    model = PointNet_TF(num_classes=classes, multiplier=1, tnet_mult=1)
    # model.build((BATCH_SIZE, 500,3))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    
    model.fit(train_dataset, epochs=num_epoch, validation_data=val_dataset)

    snowModel = 'snowModel/snowModel'
    shapeModel = 'shapeModel/shapeModel'
    modelName = snowModel if snow else shapeModel

    model.save_weights(modelName)

    preds = model.predict(test_dataset.batch(2))

    print(model.accuracy(preds, testY))

def visualizeMistakes(pred, real, data, snow=True):
    '''
    visualizing the mistakes made by the model
    pred: the predicted classes
    real: the real class labels
    data: the points of all the samples
    '''
    assert(len(pred) == len(real))

    modelName = "snow" if snow else "shapes"

    n = len(pred)
    for i in range(n):
        predClass = str(tf.keras.backend.get_value(pred[i]))
        realClass = str(real[i])
        if predClass != realClass:
            printShape(data[i], "images/confusion_" + modelName + "/pred_" + predClass + "_real_" + realClass + "_" + str(i) + ".png", realClass + " predicted as " + predClass)
    return

def visualizeModel(snow=True):
    '''
    creates a confusion matrix and outputs images of the shapes that were misclassified
    '''
    classes = 3 if snow else 4
    modelName = 'snowModel/snowModel' if snow else 'shapeModel/shapeModel'
    # predicting with the model
    model = PointNet_TF(num_classes=classes, multiplier=1, tnet_mult=1)
    load_status = model.load_weights(modelName)
    load_status.expect_partial()
    fileName = 'data/snowDataset.csv' if snow else 'data/fourShapeData.csv'
    _, _, _, _, testX, testY = getTrainTestValData(fileName, test_size = 0.5)
    test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
    logits = model.predict(test_dataset.batch(2))

    print("accuracy", model.accuracy(logits, testY).numpy())
    preds = tf.argmax(logits, 1)

    visualizeMistakes(preds, testY, testX, snow)
    # generate the confusio matrix
    confMat = confusion_matrix(testY, preds)
    confMatPlot = sns.heatmap(confMat, annot=True)
    figure = confMatPlot.get_figure()    
    figure.savefig('images/conf_mat_' + modelName.split('/')[-1] +'.png', dpi=400)

def main():
    useSnow = True
    if len(sys.argv) > 1 and sys.argv[1] == "shapes":
        print("Using Shapes Dataset")
        useSnow = False
    
    if len(sys.argv) > 2 and sys.argv[2] == "train":
        testNN_tf(useSnow)

    visualizeModel(useSnow)

if __name__ == "__main__":
    main()