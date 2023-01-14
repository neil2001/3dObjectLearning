import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

from shapeGeneration import generateCone, generateCylinder, generateSphere, printShape, Ball

def mergeSpheres(s1, s2):
    s2.points = np.array(list(filter(lambda x: euclideanDist(s1.pos, x) > s1.radius, s2.points)))
    s1.points = np.array(list(filter(lambda x: euclideanDist(s2.pos, x) > s2.radius, s1.points)))

def euclideanDist(p1, p2):
    dist = np.linalg.norm(p2-p1)
    return dist

def translateObj(points, x, y, z):
    translation_matrix = [
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ]

    translated = list(map(lambda x: np.matmul(translation_matrix, list(x) + [1]), points))
    toNumpy = np.array([x[:3] for x in translated])
    return toNumpy

def generateIceCream():
    coneRad = np.random.uniform(1,2)
    cone = generateCone(
        coneRad,
        -1 * np.random.uniform(6,10),
        400,
        400
    )
    # generate scoops, a should exclude b, b should exclude a
    numScoops = random.randint(1,4)
    scoopObjs = []
    for i in range(numScoops):
        if i == 0:
            scoopRad = np.random.uniform(coneRad, coneRad * 1.25)
            scoopZLoc = scoopRad - coneRad # place scoop where it would be tangent to the cone
        else:
            prevScoop = scoopObjs[i-1]
            scoopRad = np.random.uniform(prevScoop.radius * 0.7, prevScoop.radius * 0.9)
            scoopZLoc = prevScoop.z + (np.random.uniform(0.6, 0.8) * prevScoop.radius) + scoopRad

        newScoop = Ball(scoopRad, [0,0,scoopZLoc])
        newScoop.points = np.array(list(generateSphere(scoopRad, 0,0, scoopZLoc, 600)))
        scoopObjs.append(newScoop)

    allPoints = [cone]

    for i in range(1, len(scoopObjs)):
        mergeSpheres(scoopObjs[i], scoopObjs[i-1])
    
    for s in scoopObjs:
        allPoints.append(s.points)

    coords = np.concatenate(allPoints)
    np.random.shuffle(coords)
    return coords

def generateHat():
    brimHeight = np.random.uniform(0.2,0.4)
    hatHeight = np.random.uniform(1.5,2.25)

    brim = generateCylinder(np.random.uniform(1.5,1.75), brimHeight, n=300, baseN=100)
    hat = generateCylinder(np.random.uniform(0.75,1.15), hatHeight)

    return np.concatenate((brim, hat))

def generateSnowMan():
    ballCount = 3 #random.randint(2,3)
    ballObjs = []
    for i in range(ballCount):
        if i == 0:
            ballRad = np.random.uniform(3,5)
            ballZLoc = 0 # place scoop where it would be tangent to the cone
        else:
            prevBall = ballObjs[i-1]
            ballRad = np.random.uniform(prevBall.radius * 0.7, prevBall.radius * 0.9)
            ballZLoc = prevBall.z + (np.random.uniform(0.6, 0.8) * prevBall.radius) + ballRad

        newBall = Ball(ballRad, [0,0,ballZLoc])
        newBall.points = np.array(list(generateSphere(ballRad, 0,0, ballZLoc, 600)))
        ballObjs.append(newBall)

    topHat = generateHat()
    # print(ballObjs[-1].z, ballObjs[-1].z + ballObjs[-1].radius)
    topHat = translateObj(topHat, 0, 0, ballObjs[-1].z + ballObjs[-1].radius)
    allObjs = [topHat]
    for b in ballObjs:
        allObjs.append(b.points)
    
    points = np.concatenate(allObjs)
    np.random.shuffle(points)
    return points


def generateCloud():
    centerRad = np.random.uniform(3,5)
    center = Ball(centerRad, np.array([0,0,0]), generateSphere(centerRad, n=800))

    lRad = centerRad * np.random.uniform(0.5, 0.9)
    lX = -1 * np.random.uniform(0.8, 0.9) * centerRad
    lSphere = Ball(lRad, np.array([lX,0,0]), generateSphere(lRad, x=lX, n=800))

    rRad = centerRad * np.random.uniform(0.5, 0.9)
    rX = np.random.uniform(0.8, 0.9) * centerRad
    rSphere = Ball(rRad, np.array([rX,0,0]), generateSphere(rRad, x = rX, n=800))

    mergeSpheres(lSphere, center)
    mergeSpheres(rSphere, center)

    points = np.concatenate((center.points,lSphere.points,rSphere.points))
    np.random.shuffle(points)
    return points

def generateSnowDataset(samples=1500, num_points=500, num_classes=3):
    funcs = [generateCloud, generateIceCream, generateSnowMan]
    data = []
    for i in range(samples):
        objClass = random.randint(0,2)
        points = funcs[objClass]()
        pointsFlat = [objClass] + list(points[:num_points].flatten())
        data.append(pointsFlat)
        if i % 100 == 0:
            print(str(i) + " shapes generated")

    df = pd.DataFrame(data)
    return df

def main():
    snowShapes = generateSnowDataset()
    snowShapes.to_csv("./snowDataset.csv")
    # print(snowShapes.head())

    # ice_cream = generateIceCream()
    # printShape(ice_cream)

    # snowMan = generateSnowMan()
    # printShape(snowMan)

    # cloud = generateCloud()
    # printShape(cloud)


if __name__ == "__main__":
    main()