import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

def generateSphere(radius, x, y, z, n=200, toPrint=False):
    points = np.zeros((n, 3))

    translation_matrix = [
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ]

    for i in range(n):
        u = np.random.uniform(-1,1)
        v = np.random.uniform(-1,1)
        w = np.random.uniform(-1,1)
        vec = np.array([u,v,w,0])
        normVec = vec / np.linalg.norm(vec)
        pt = radius * normVec
        pt[3] = 1
        pt = np.matmul(translation_matrix, pt)
        points[i] = pt[:3]

    if toPrint:
        xLocs = [pt[0] for pt in points]
        yLocs = [pt[1] for pt in points]
        zLocs = [pt[2] for pt in points]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xLocs, yLocs, zLocs)
        ax.set_title('Example of a uniformly sampled sphere', fontdict={'fontsize':20})
    np.random.shuffle(points)
    return points

def generateCone(radius, height, x, y, z, baseN = 100, coneN = 200, toPrint = False):
    points = np.zeros((baseN + coneN, 3))

    for i in range(baseN):
        theta = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0, radius)
        points[i] = np.array([r*math.cos(theta), r*math.sin(theta), 0])

    for i in range(coneN):
        h = np.random.uniform(0, height)
        hRad = radius * ((height - h)/height)
        theta = np.random.uniform(0, 2*math.pi)
        points[baseN + i] = np.array([hRad * math.cos(theta), hRad * math.sin(theta),h])

    if toPrint:
        xLocs = [pt[0] for pt in points]
        yLocs = [pt[1] for pt in points]
        zLocs = [pt[2] for pt in points]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xLocs, yLocs, zLocs)
        ax.set_title('Example of a uniformly sampled sphere', fontdict={'fontsize':20})
    
    np.random.shuffle(points)
    return points
  
def generateShapeDataset(samples=1000, points=500):
    data = []
    for i in range(samples//2):
        newSphere = generateSphere(np.random.uniform(1, 50), 0,0,0, n=500)
        sphereAsRow = [0] + list(newSphere.flatten())
        data.append(sphereAsRow)
    for i in range(samples//2):
        newCone = generateCone(np.random.uniform(1, 50), np.random.uniform(5, 80), 0,0,0, baseN=150, coneN = 350)
        sphereAsRow = [1] + list(newCone.flatten())
        data.append(sphereAsRow)
    df = pd.DataFrame(data)
    return df

def printShape(points):
    xLocs = [pt[0] for pt in points]
    yLocs = [pt[1] for pt in points]
    zLocs = [pt[2] for pt in points]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xLocs, yLocs, zLocs)
    ax.set_title('Example of a uniformly sampled sphere', fontdict={'fontsize':20})
    fig.savefig('./full_figure.png')

def main():
    shapes = generateShapeDataset()
    shapes.to_csv('./sphereConeData.csv')

if __name__ == "__main__":
    main()