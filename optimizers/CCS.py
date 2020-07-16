# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""
import math
import numpy
import random
import time
from solution import solution


    
def get_cuckoos(nest,best,lb,ub,n,dim):
    
    # perform Levy flights
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.array(nest)
    beta=3/2;
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);

    s=numpy.zeros(dim)
    for j in range (0,n):
        s=nest[j,:]
        u=numpy.random.randn(len(s))*sigma
        v=numpy.random.randn(len(s))
        step=u/abs(v)**(1/beta)
 
        stepsize=0.01*(step*(s-best))

        s=s+stepsize*numpy.random.randn(len(s))
    
        tempnest[j,:]=numpy.clip(s, lb, ub)

    return tempnest

def get_best_nest(nest,labelsPred, newnest,fitness,n,dim,objf, k, points, metric):
# Evaluating all new solutions
    tempnest=numpy.zeros((n,dim))
    tempnest=numpy.copy(nest)
    templabels=numpy.copy(labelsPred)

    for j in range(0,n):
    #for j=1:size(nest,1),
        startpts = numpy.reshape(newnest[j,:], (k,(int)(dim/k)))
        if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
            fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
        else:
            fitnessValue, labelsPredValues=objf(startpts, points, k) 
        fnew= fitnessValue
        newLabels = labelsPredValues
        
        if fnew<=fitness[j]:
           fitness[j]=fnew
           tempnest[j,:]=newnest[j,:]
           templabels[j,:]=newLabels
        
    # Find the current best

    fmin = min(fitness)
    I=numpy.argmin(fitness)
    bestlocal=tempnest[I,:]
    bestlabels=templabels[I,:]

    return fmin,bestlocal,bestlabels,tempnest,fitness, templabels

# Replace some nests by constructing new solutions/nests
def empty_nests(nest,pa,n,dim):

    # Discovered or not 
    tempnest=numpy.zeros((n,dim))

    K=numpy.random.uniform(0,1,(n,dim))>pa
    
    
    stepsize=random.random()*(nest[numpy.random.permutation(n),:]-nest[numpy.random.permutation(n),:])

    
    tempnest=nest+stepsize*K
 
    return tempnest
##########################################################################


def CS(objf,lb,ub,dim,n,N_IterTotal,k,points, metric):

    #lb=-1
    #ub=1
    #n=50
    #N_IterTotal=1000
    #dim=30
    
    # Discovery rate of alien eggs/solutions
    pa=0.25
    
    
#    Lb=[lb]*nd
#    Ub=[ub]*nd
    convergence=[]

    # RInitialize nests randomely
    nest=numpy.random.rand(n,dim)*(ub-lb)+lb
    labelsPred = numpy.zeros((n,len(points)))
    
    new_nest=numpy.zeros((n,dim))
    new_nest=numpy.copy(nest)
    
    bestnest=[0]*dim;
    bestLabelsPred=[0]*len(points);
     
    fitness=numpy.zeros(n) 
    fitness.fill(float("inf"))
    

    s=solution()

     
    print("CS is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    fmin,bestnest,bestlabels,nest,fitness,labelsPred =get_best_nest(nest,labelsPred,new_nest,fitness,n,dim,objf,k,points, metric)
    convergence = [];
    # Main loop counter
    for iter in range (0,N_IterTotal):
        # Generate new solutions (but keep the current best)
     
         new_nest=get_cuckoos(nest,bestnest,lb,ub,n,dim)
         
         
         # Evaluate new solutions and find best
         fnew,best,bestlabels,nest,fitness,labelsPred=get_best_nest(nest,labelsPred,new_nest,fitness,n,dim,objf,k,points, metric)
         
        
         new_nest=empty_nests(new_nest,pa,n,dim) ;
         
        
        # Evaluate new solutions and find best
         fnew,best,bestlabels,nest,fitness,labelsPred=get_best_nest(nest,labelsPred,new_nest,fitness,n,dim,objf,k,points, metric)
    
         if fnew<fmin:
            fmin=fnew
            bestnest=best
            bestLabelsPred=bestlabels
    
         if (iter%1==0):
            print(['At iteration '+ str(iter)+ ' the best fitness is '+ str(fmin)]);
         convergence.append(fmin)

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="CS"
    s.objfname=objf.__name__
    s.bestIndividual = bestnest
    s.labelsPred = numpy.array(bestLabelsPred, dtype=numpy.int64)
    
     
    
    return s
    




 



