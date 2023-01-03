import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

class Ball():
    def __init__(self, rad, loc, pts=[]):
        self.radius = rad
        self.x = loc[0]
        self.y = loc[1]
        self.z = loc[2]
        self.pos = loc
        self.points = pts

class Box():
    def __init__(self, l, w, h, pos, pts=[]):
        self.length = l
        self.width = w
        self.height = h
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.pos = pos
        self.points = pts

def generateCube(n=500, toPrint=False):
#     Choose an axis (x,y,z), a, represented as (0,1,2) at random
# Choose side, s, either (-0.5 or 0.5) at random
# Choose three values, coord, between (-0.5 and 0.5)
# Set coord[a] to s
# Repeat n times to sample the unit cube centered at 0,0

    points = np.zeros((n,3))
    for i in range(n):
        axis = random.randint(0, 2)
        side = random.choice([-0.5, 0.5])
        coord = np.random.uniform(-0.5, 0.5, 3)
        coord[axis] = side
        points[i] = coord
    # print(points)
    np.random.shuffle(points)
    return points

def generateCircle2D(radius, zloc, n=200):
    points = np.zeros((n,3))
    for i in range(n):
        theta = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0, radius)
        points[i] = np.array([r*math.cos(theta), r*math.sin(theta), zloc])
    return points

def generateCylinder(radius, height, n=500, baseN = 150):
    # generate points on circle for top and bottom,
    # choose a random height, choose a point around the radius
    EPSILON = 0.1
    top = generateCircle2D(radius, height, baseN)
    bottom = generateCircle2D(radius, 0, baseN)
    bodyN = n - (2 * baseN)
    body = np.zeros((bodyN,3))
    for i in range(bodyN):
        z = np.random.uniform(0 + EPSILON, height - EPSILON)
        theta = np.random.uniform(0, 2*math.pi)
        body[i] = np.array([radius*math.cos(theta), radius*math.sin(theta), z])

    points = np.concatenate((top, bottom, body))
    np.random.shuffle(points)
    return points


def generateSphere(radius, x=0, y=0, z=0, n=200, toPrint=False):
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

def generateCone(radius, height, x=0, y=0, z=0, baseN = 100, coneN = 200, toPrint = False):
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
  
def generateShapeDataset(samples=1000, points=500, num_classes=3):
    data = []
    for i in range(samples//num_classes):
        newSphere = generateSphere(np.random.uniform(1, 50), 0,0,0, n=500)
        sphereAsRow = [0] + list(newSphere.flatten())
        data.append(sphereAsRow)
    for i in range(samples//num_classes):
        newCone = generateCone(np.random.uniform(1, 50), np.random.uniform(5, 80), 0,0,0, baseN=150, coneN = 350)
        coneAsRow = [1] + list(newCone.flatten())
        data.append(coneAsRow)
    for i in range(samples//num_classes):
        newCube = generateCube()
        cubeAsRow = [2] + list(newCube.flatten())
        data.append(cubeAsRow)
    df = pd.DataFrame(data)
    return df

def printShape(points):
    xLocs = [pt[0] for pt in points]
    yLocs = [pt[1] for pt in points]
    zLocs = [pt[2] for pt in points]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xLocs, yLocs, zLocs)
    ax.set_title('Example of a uniformly sampled object', fontdict={'fontsize':20})
    fig.savefig('./full_figure.png')
    fig.show()

def main():
    # cylinder = generateCylinder(1,2, 1000, 250)

    shapes = generateShapeDataset(1500,500,3)
    shapes.to_csv('./sphereConeCubeData.csv')

if __name__ == "__main__":
    main()