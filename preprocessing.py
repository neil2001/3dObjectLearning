import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from shapeGeneration import printShape

import torch
from torch.utils.data import DataLoader, Dataset

# import tensorflow as tf

class ShapeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], torch.Tensor([self.Y[index]])


# Applying random rotation and translation
def normalizeScale(shapeMat):
    # return shapeMat
    minCoord = shapeMat.min(axis=0)
    maxCoord = shapeMat.max(axis=0)
    # print(minCoord, maxCoord)
    # axSizes = 1/(maxCoord - minCoord)
    scaleVal = max(maxCoord-minCoord)
    scaled = shapeMat / scaleVal
    # print(axSizes)
    # scaleMat = np.array([
    #     [axSizes[0],0,0],
    #     [0,axSizes[1],0,],
    #     [0,0,axSizes[2]]
    #     ])
    # scaled = np.matmul(shapeMat, scaleMat)
    return scaled

    # minCoord = shapeMat.min(axis=0)
    # shapeMat = (shapeMat - minCoord)/(shapeMat.max(axis=0) - minCoord)
    # return shapeMat

def randomRotation(shapeMat):
    rotation_angle = np.random.uniform(0,1) * 2 * math.pi
    rotation_axis = np.random.uniform(-1, 1, 3)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cosA = math.cos(rotation_angle)
    cosInv = (1-cosA)
    sinA = math.sin(rotation_angle)
    sinInv = (1-sinA)
    x = rotation_axis[0]
    y = rotation_axis[1]
    z = rotation_axis[2]
    rodrigues_mat = np.array([[cosA + (x**2)*(cosInv), x*y*cosInv - z*sinA,  x*z*cosInv + y*sinA],
                    [x*y*cosInv + z*sinA,    cosA + (y**2)*cosInv, y*z*cosInv - x*sinA],
                    [x*z*cosInv - y*sinA,    y*z*cosInv + x*sinA,  cosA + (z**2)*cosInv]
                    ])
    rod_tp = np.transpose(rodrigues_mat)
    rotated_shape = np.matmul(shapeMat, rod_tp)
    return rotated_shape

def randomTranslate(shapeMat):
    # r,c = shapeMat.shape
    noise = np.random.uniform(-1,1, shapeMat.shape) * 0.025
    shapeMat += noise
    return shapeMat

def transform(shapeMat):
    normalized = normalizeScale(shapeMat)
    rotated = randomRotation(normalized)
    translated = randomTranslate(rotated)
    # normalized = normalizeScale(shapeMat)
    # print(normalized)
    return translated

    # return shapeMat

def preprocess_tf(file_loc, num_points=500):
    shapeDataset = pd.read_csv(file_loc, index_col=0)
    shapeDataset = shapeDataset.sample(frac=1).reset_index(drop=True)
    numSamples = len(shapeDataset)
    print("number of samples", numSamples)
    # print(shapeDataset.head())
    labels = np.array(shapeDataset['0']).flatten() # labels
    shapeCoords = shapeDataset.drop(columns=['0'])
    coordsAsNumpy = shapeCoords.to_numpy().reshape((numSamples,num_points,3))
    transformed = np.array(list(map(transform, list(coordsAsNumpy))))
    
    # one_hot_labels = tf.one_hot(labels, num_classes)
    return transformed, labels 

def preprocess(file_loc, batch_size, test_size):
    shapeDataset = pd.read_csv(file_loc, index_col=0)
    numSamples = len(shapeDataset)
    print("number of samples", numSamples)
    labels = np.array(shapeDataset['0']).flatten() # labels
    shapeCoords = shapeDataset.drop(columns=['0'])
    coordsAsNumpy = shapeCoords.to_numpy().reshape((numSamples,500,3))
    transformed = np.array(list(map(transform, list(coordsAsNumpy))))
    # transformed = coordsAsNumpy
    X_train, X_test, Y_train, Y_test = train_test_split(transformed, labels, test_size=test_size, random_state=42)
    # print(X_train, X_test, Y_train, Y_test)
    printShape(transformed[501])

    dataset_train = ShapeDataset(X_train, Y_train)
    dataset_test = ShapeDataset(X_test, Y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return dataloader_train, dataloader_test

def main():
    # snowData, snowLabels = preprocess_tf('/Users/neilxu/Documents/3dObjectLearning/data/snowDataset.csv', 3)
    # printShape(snowData[0])

    shapeData, shapeLabels = preprocess_tf('data/fourShapeData.csv')
    printShape(shapeData[0], "images/testShape1.png", "test primitive " + str(shapeLabels[0]))
    printShape(shapeData[10], "images/testShape2.png", "test primitive " + str(shapeLabels[10]))
    return

if __name__ == "__main__":
    main()