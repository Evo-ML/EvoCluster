# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:04:15 2019

@author: Raneem
"""

import random
import numpy
import math
from solution import solution
import time

def PSO(objf,lb,ub,dim,PopSize,iters, k, points, metric):
    
    # PSO parameters
    
#    dim=30
#    iters=200
    
#    Vmax=6
    Vmax=6#raneem
#    PopSize=50     #population size
    wMax=0.9
    wMin=0.2
    c1=2
    c2=2
#    lb=-10
#    ub=10
#    
    s=solution()
    
    
    ######################## Initializations
    
    vel=numpy.zeros((PopSize,dim))
    
    pBestScore=numpy.zeros(PopSize) 
    pBestScore.fill(float("inf"))    
    pBest=numpy.zeros((PopSize,dim))
    pBestLabelsPred=numpy.full((PopSize,len(points)), numpy.inf)
    
    
    gBest=numpy.zeros(dim)
    gBestScore=float("inf")
    gBestLabelsPred=numpy.full(len(points), numpy.inf)
    
    pos=numpy.random.uniform(0,1,(PopSize,dim)) *(ub-lb)+lb
    
    convergence_curve=numpy.zeros(iters)
    
    ############################################
    print("PSO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters):
        for i in range(0,PopSize):
            #pos[i,:]=checkBounds(pos[i,:],lb,ub)
            pos[i,:]=numpy.clip(pos[i,:], lb, ub)
            #Calculate objective function for each particle
            
            startpts = numpy.reshape(pos[i,:], (k,(int)(dim/k)))
            if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
                fitness, labelsPred=objf(startpts, points, k, metric) 
            else:
                fitness, labelsPred=objf(startpts, points, k) 
    
            if(pBestScore[i]>fitness):
                pBestScore[i]=fitness
                pBest[i,:]=pos[i,:].copy()
                pBestLabelsPred[i,:]=numpy.copy(labelsPred)
                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=pos[i,:].copy()
                gBestLabelsPred=numpy.copy(labelsPred)
        
        #Update the W of PSO
        w=wMax-l*((wMax-wMin)/iters);#check this
        
        for i in range(0,PopSize):
            for j in range (0,dim):
                r1=random.random()
                r2=random.random()
                vel[i,j]=w*vel[i,j]+c1*r1*(pBest[i,j]-pos[i,j])+c2*r2*(gBest[j]-pos[i,j])
                
                if(vel[i,j]>Vmax):
                    vel[i,j]=Vmax
                
                if(vel[i,j]<-Vmax):
                    vel[i,j]=-Vmax
                            
                pos[i,j]=pos[i,j]+vel[i,j]
        
        convergence_curve[l]=gBestScore
        
        if (l%1==0):
               print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)]);
        
        
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="PSO"
    s.objfname=objf.__name__
    s.labelsPred = numpy.array(gBestLabelsPred, dtype=numpy.int64)
    s.bestIndividual = gBest

    return s
         
    
