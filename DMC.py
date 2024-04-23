
import numpy as np
from numba import njit

@njit
def calculateEuclideanDist(sample,attributes):
    diff =  np.array(attributes) - np.array(sample)
    elevate = np.sum((diff) ** 2)
    result = np.sqrt(elevate)
    return result
def DMC(xtrain, ytrain, xtest):
    predicts = []
    centroids = {}
    uniqueLabels = np.unique(ytrain)
    for label in uniqueLabels:
        xtrain = np.array(xtrain)
        classSamples = xtrain[ytrain == label]
        centroids[label] = np.mean(classSamples, axis=0)
    for testSample in xtest:
        selectedLabel = None
        closestDistance = None
        for label, centroid in centroids.items():
            centroid = centroid.tolist()
            dist = calculateEuclideanDist(testSample, centroid)
            if closestDistance is None or dist < closestDistance:
                closestDistance = dist
                selectedLabel = label
        predicts.append(selectedLabel)

    return predicts