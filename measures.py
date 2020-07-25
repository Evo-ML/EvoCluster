# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:22:53 2019

@author: Raneem
"""


from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
import statistics
import math  
import numpy
import sys
import math

def HS(labelsTrue, labelsPred):    
    return float("%0.2f"%metrics.homogeneity_score(labelsTrue,labelsPred))

def CS(labelsTrue, labelsPred):    
    return float("%0.2f"%metrics.completeness_score(labelsTrue,labelsPred))
    
def VM(labelsTrue, labelsPred):
    return float("%0.2f"%metrics.v_measure_score(labelsTrue,labelsPred))
    
def AMI(labelsTrue, labelsPred):
    return float("%0.2f"%metrics.adjusted_mutual_info_score(labelsTrue,labelsPred))
    
def ARI(labelsTrue, labelsPred):
    return float("%0.2f"%metrics.adjusted_rand_score(labelsTrue,labelsPred))
    
def Fmeasure(labelsTrue, labelsPred):
    return float("%0.2f"%metrics.f1_score(labelsTrue, labelsPred, average='macro'))

def SC(points, labelsPred):#Silhouette Coefficient

    if numpy.unique(labelsPred).size == 1:
        fitness = sys.float_info.max
    else:    
        silhouette=  float("%0.2f"%metrics.silhouette_score(points, labelsPred, metric='euclidean'))
        silhouette = (silhouette + 1) / 2
        fitness = 1 - silhouette
    return fitness

def accuracy(labelsTrue, labelsPred):#Silhouette Coefficient
    #silhouette = metrics.accuracy_score(labelsTrue, labelsPred, normalize=False)
    return ARI(labelsTrue, labelsPred)


def delta_fast(ck, cl, distances):
    values = distances[numpy.where(ck)][:, numpy.where(cl)]
    values = values[numpy.nonzero(values)]

    return numpy.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[numpy.where(ci)][:, numpy.where(ci)]
    #values = values[numpy.nonzero(values)]
            
    return numpy.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : numpy.array
        numpy.array([N, p]) of all points
    labels: numpy.array
        numpy.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
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
    

def DI(points, labelsPred):#dunn index
    dunn = float("%0.2f"%dunn_fast(points, labelsPred))
    if(dunn < 0):
        dunn = 0
    fitness = 1 - dunn
    return fitness

    
def DB(points, labelsPred):
    return float("%0.2f"%metrics.davies_bouldin_score(points, labelsPred))

def stdev(individual, labelsPred, k, points):
    std = 0    
    distances = []
    f = (int)(len(individual) / k)
    startpts = numpy.reshape(individual, (k,f))   

    for i in range(k):
        index_list = numpy.where(labelsPred == i)
        distances = numpy.append(distances, numpy.linalg.norm(points[index_list]-startpts[i], axis = 1))
    
    std =  numpy.std(distances)
        
    #stdev = math.sqrt(std)/ k 
    #print("stdev:",stdev)
    return std
    
    
'''
def SSE(individual, k, points):
    
    f = (int)(len(individual) / k)
    startpts = numpy.reshape(individual, (k,f))    
    labelsPred = [-1] * len(points)
    sse = 0
    
    for i in range(len(points)):
        distances = numpy.linalg.norm(points[i]-startpts, axis = 1)
        sse = sse + numpy.min(distances)
        clust = numpy.argmin(distances)
        labelsPred[i] = clust
        
    if numpy.unique(labelsPred).size < k:
        sse = sys.float_info.max
               
    print("SSE:",sse)
    return sse
'''

def SSE(individual, labelsPred, k, points):
    
    f = (int)(len(individual) / k)
    startpts = numpy.reshape(individual, (k,f)) 
    fitness = 0
       
    
    centroidsForPoints = startpts[labelsPred]
    fitnessValues = numpy.linalg.norm(points-centroidsForPoints, axis = 1)**2
    fitness = sum(fitnessValues)
    return fitness


def TWCV(individual, labelsPred, k, points):    
    sumAllFeatures = sum(sum(numpy.power(points,2)))
    sumAllPairPointsCluster = 0
    for clusterId in range(k):
        indices = numpy.where(numpy.array(labelsPred) == clusterId)[0]
        pointsInCluster = points[numpy.array(indices)]
        sumPairPointsCluster = sum(pointsInCluster)
        sumPairPointsCluster = numpy.power(sumPairPointsCluster,2)
        if len(pointsInCluster) != 0:
            sumPairPointsCluster = sum(sumPairPointsCluster)
            sumPairPointsCluster = sumPairPointsCluster/len(pointsInCluster)
        
        sumAllPairPointsCluster += sumPairPointsCluster
    fitness = (sumAllFeatures - sumAllPairPointsCluster)
    return fitness


def purity(labelsTrue,labelsPred):            
    # get the set of unique cluster ids
    labelsTrue=numpy.asarray(labelsTrue).astype(int)
    labelsPred=numpy.asarray(labelsPred).astype(int)
    
    k=(max(labelsTrue)+1).astype(int)
    
    totalSum = 0;
    
    for i in range(0,k):
        max_freq=0
    
        t1=numpy.where(labelsPred == i)

        for j in range(0,k):
            t2=numpy.where(labelsTrue == j)
            z=numpy.intersect1d(t1,t2);
           
            e=numpy.shape(z)[0]
               
            if (e >= max_freq):
                max_freq=e
             
        totalSum=totalSum + max_freq

    purity=totalSum/numpy.shape(labelsTrue)[0]
     
    #print("purity:",purity)
    
    return purity

def entropy(labelsTrue,labelsPred):            
        # get the set of unique cluster ids
    labelsTrue=numpy.asarray(labelsTrue).astype(int)
    labelsPred=numpy.asarray(labelsPred).astype(int)
    
    k=(max(labelsTrue)+1).astype(int)
    
    entropy=0
       
    for i in range(0,k):
        
        t1=numpy.where(labelsPred == i)

        entropyI=0

        for j in range(0,k):
            t2=numpy.where(labelsTrue == j)
            z=numpy.intersect1d(t1,t2);
           
            e=numpy.shape(z)[0]
            if (e!=0):
                entropyI=entropyI+(e/numpy.shape(t1)[1])*math.log(e/numpy.shape(t1)[1])
               
        a=numpy.shape(t1)[1]
        b=numpy.shape(labelsTrue)[0]
        entropy=entropy+(( a / b )*((-1 / math.log(k))*entropyI)) 
    
    #print("entropy:",entropy)
     
    return entropy
