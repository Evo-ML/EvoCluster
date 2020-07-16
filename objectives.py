# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:12:29 2019

@author: Raneem
"""

from sklearn import cluster, metrics
from scipy.spatial.distance import pdist, cdist
import numpy
import sys

def getLabelsPred(startpts, points, k):
    labelsPred = [-1] * len(points)
    
    for i in range(len(points)):
        distances = numpy.linalg.norm(points[i]-startpts, axis = 1)
        labelsPred[i] = numpy.argmin(distances)
    
    return labelsPred
    

def SSE(startpts, points, k, metric):
    labelsPred = getLabelsPred(startpts, points, k)
    fitness = 0
        
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        centroidsForPoints = startpts[labelsPred]
        fitness = 0
        for i in range(k):
            indexes = [n for n,x in enumerate(labelsPred) if x==i]
            fit = cdist(points[indexes], centroidsForPoints[indexes], metric)**2
            fit = sum(fit)[0]
            fitness += fit
    return fitness, labelsPred


def TWCV(startpts, points, k):
    labelsPred = getLabelsPred(startpts, points, k)
    
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        sumAllFeatures = sum(sum(numpy.power(points,2)))
        sumAllPairPointsCluster = 0
        for clusterId in range(k):
            indices = numpy.where(numpy.array(labelsPred) == clusterId)[0]
            pointsInCluster = points[numpy.array(indices)]
            sumPairPointsCluster = sum(pointsInCluster)
            sumPairPointsCluster = numpy.power(sumPairPointsCluster,2)
            sumPairPointsCluster = sum(sumPairPointsCluster)
            sumPairPointsCluster = sumPairPointsCluster/len(pointsInCluster)
            
            sumAllPairPointsCluster += sumPairPointsCluster
        fitness = (sumAllFeatures - sumAllPairPointsCluster)
    return fitness, labelsPred


def SC(startpts, points, k, metric):    
    labelsPred = getLabelsPred(startpts, points, k)
    
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        silhouette = metrics.silhouette_score(points, labelsPred, metric=metric)
        #silhouette = (silhouette - (-1)) / (1 - (-1))
        silhouette = (silhouette + 1) / 2
        fitness = 1 - silhouette
    return fitness, labelsPred


def DB(startpts, points, k):
    labelsPred = getLabelsPred(startpts, points, k)
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        fitness = metrics.davies_bouldin_score(points, labelsPred)
    return fitness, labelsPred

def CH(startpts, points, k):
    labelsPred = getLabelsPred(startpts, points, k)
    
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        ch = metrics.calinski_harabaz_score(points, labelsPred)
        fitness = 1 / ch
    return fitness, labelsPred


def delta_fast(ck, cl, distances):
    values = distances[numpy.where(ck)][:, numpy.where(cl)]
    values = values[numpy.nonzero(values)]

    return numpy.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[numpy.where(ci)][:, numpy.where(ci)]
    #values = values[numpy.nonzero(values)]
            
    return numpy.max(values)

def dunn_fast(points, labels, metric):
    v = pdist(points, metric)
    size_X  = len(points)
    X = numpy.zeros((size_X,size_X))
    X[numpy.triu_indices(X.shape[0], k = 1)] = v
    distances = X + X.T
    ks = numpy.sort(numpy.unique(labels))
    
    deltas = numpy.ones([len(ks), len(ks)])*1000000
    big_deltas = numpy.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = numpy.min(deltas)/numpy.max(big_deltas)
    return di
    

def DI(startpts, points, k, metric):
    labelsPred = getLabelsPred(startpts, points, k)
    
    if numpy.unique(labelsPred).size < k:
        fitness = sys.float_info.max
    else:
        dunn = dunn_fast(points, labelsPred, metric)
        if(dunn < 0):
            dunn = 0
        fitness = 1 - dunn
    return fitness, labelsPred


def getFunctionDetails(a):    
    # [name, lb, ub]
    param = {  0: ["SSE",0,1],
            1: ["TWCV",0,1],
            2: ["SC",0,1],
            3: ["DB",0,1],
            #4: ["CH",0,1],
            4: ["DI",0,1]
            }
    return param.get(a, "nothing")
