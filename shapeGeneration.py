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

def generateCube(n=500, toPrint=False):
    # Choose an axis (x,y,z), a, represented as (0,1,2) at random
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
    np.random.shuffle(points)
    return points

def generateCircle2D(radius, zloc, n=200):
    '''
    for n points, choose a random angle and random radius, compute the 
    rectangular coordinates
    '''
    points = np.zeros((n,3))
    for i in range(n):
        theta = np.random.uniform(0, 2*math.pi)
        r = math.sqrt(np.random.uniform(0, radius**2))
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


def generateSphere(radius, x=0, y=0, z=0, n=500, toPrint=False):
    '''
    choose a random point in the [-1,1] cube. Normalize this point such that it 
    has length 1. Multiply by radius to yield a point on the circle
    '''
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

def generateCone(radius, height, x=0, y=0, z=0, baseN = 150, coneN = 350, toPrint = False):
    '''
    sample a 2D circle forming the base. Choose a random height between [0,H].
    Choose a random angle, compute the radius at that height, conver to 
    rectangular coordinates
    '''
    points = np.zeros((baseN + coneN, 3))

    for i in range(baseN):
        theta = np.random.uniform(0, 2*math.pi)
        r = math.sqrt(np.random.uniform(0, radius**2))
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
    '''
    generates some number of sample shapes (equal samples for each)
    '''
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

def generateShapesRandom(samples=2000, num_points=500):
    '''
    generates some number of samples of the four shapes from random
    '''
    data = []
    funcs = [generateSphere, generateCone, generateCube, generateCylinder]
    for i in range(samples):
        objClass = random.randint(0,3)
        func = funcs[objClass]
        if objClass == 2:
            points = func()
        else:
            rad = np.random.uniform(5, 15)
            if objClass == 0:
                points = func(rad)
            else:
                height = np.random.uniform(15,50)
                points = func(rad, height)
        pointsFlat = [objClass] + list(points[:num_points].flatten())
        data.append(pointsFlat)

        if i % 100 == 0:
            print(str(i) + " shapes generated")

    return pd.DataFrame(data)

def set_axes_equal(ax):
    '''
    Taken from stack overflow 
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def printShape(points, dest, title):
    '''
    visualizes the shape and saves it to a png (dest)
    '''
    xLocs = [pt[0] for pt in points]
    yLocs = [pt[1] for pt in points]
    zLocs = [pt[2] for pt in points]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xLocs, yLocs, zLocs)
    set_axes_equal(ax)
    ax.set_title(title, fontdict={'fontsize':20})
    fig.savefig(dest)
    fig.show()
    plt.close()

def main():
    # cylinder = generateCylinder(1,2, 1000, 250)
    # printShape(cylinder, "images/cylinder.png", "Example of a cylinder")

    # sphere = generateSphere(5, n=500)
    # printShape(sphere, "images/sphere.png", "Example of a sphere")

    # cone = generateCone(2, 6, baseN = 200, coneN = 300)
    # printShape(cone, "images/cone.png", "Example of a cone")

    # cube = generateCube()
    # printShape(cube, "images/cube.png", "Example of a cube")

    shapes = generateShapesRandom()
    shapes.to_csv('data/fourShapeData.csv')

if __name__ == "__main__":
    main()