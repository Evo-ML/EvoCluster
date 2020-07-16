# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:49:35 2016

@author: hossam
"""

#% ======================================================== % 
#% Files of the Matlab programs included in the book:       %
#% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
#% Second Edition, Luniver Press, (2010).   www.luniver.com %
#% ======================================================== %    
#
#% -------------------------------------------------------- %
#% Firefly Algorithm for constrained optimization using     %
#% for the design of a spring (benchmark)                   % 
#% by Xin-She Yang (Cambridge University) Copyright @2009   %
#% -------------------------------------------------------- %

import numpy
import math
import time
from solution import solution





def alpha_new(alpha,NGen):
    #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    #% alpha_0=0.9
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha



def FFA(objf,lb,ub,dim,n,MaxGeneration, k, points, metric):

    #General parameters

    #n=50 #number of fireflies
    #dim=30 #dim  
    #lb=-50
    #ub=50
    #MaxGeneration=500
 
    #FFA parameters
    alpha=0.5  # Randomness 0--1 (highly random)
    betamin=0.20  # minimum value of beta
    gamma=1   # Absorption coefficient
    
    
    zn=numpy.ones(n)
    zn.fill(float("inf")) 
    
    
    #ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
    ns = numpy.zeros((n, dim))
    ns=numpy.random.uniform(0,1,(n,dim)) *(ub-lb)+lb
    
    Lightn=numpy.ones(n)
    Lightn.fill(float("inf")) 
    labelsPred=numpy.zeros((n,len(points)))
    
    #[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
    
    convergence=[]
    s=solution()

     
    print("CS is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for Iteration in range (0,MaxGeneration):     # start iterations
    
        #% This line of reducing alpha is optional
        alpha=alpha_new(alpha,MaxGeneration);
        
        #% Evaluate new solutions (for all n fireflies)
        for i in range(0,n):            
            startpts = numpy.reshape(ns[i,:], (k,(int)(dim/k)))
            if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
                fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
            else:
                fitnessValue, labelsPredValues=objf(startpts, points, k) 
            zn[i] = fitnessValue
            Lightn[i]=zn[i]
            labelsPred[i,:] = labelsPredValues
        
        
                
        
        # Ranking fireflies by their light intensity/objectives
    
        
        Lightn=numpy.sort(zn)
        Index=numpy.argsort(zn)
        ns=ns[Index,:]
        
        
        #Find the current best
        nso=ns
        Lighto=Lightn
        nbest=ns[0,:] 
        Lightbest=Lightn[0]
        labelsPredBest=labelsPred[0]
        
        #% For output only
        fbest=Lightbest
        
        #% Move all fireflies to the better locations
    #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
    #          Lightbest,alpha,betamin,gamma,Lb,Ub);
        
        scale=numpy.ones(dim)*abs(ub-lb)

        for i in range (0,n):
            # The attractiveness parameter beta=exp(-gamma*r)
            for j in range(0,n):
                r=numpy.sqrt(numpy.sum((ns[i,:]-ns[j,:])**2));
                #r=1
                # Update moves
                if Lightn[i]>Lighto[j]: # Brighter and more attractive
                   beta0=1
                   beta=(beta0-betamin)*math.exp(-gamma*r**2)+betamin
                   tmpf=alpha*(numpy.random.rand(dim)-0.5)*scale
                   ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf
        
        
        #ns=numpy.clip(ns, lb, ub)
        
        convergence.append(fbest)
        	
        if (Iteration%1==0):
               print(['At iteration '+ str(Iteration)+ ' the best fitness is '+ str(fbest)])
    #    
       ####################### End main loop
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="FFA"
    s.objfname=objf.__name__
    s.labelsPred = numpy.array(labelsPredBest, dtype=numpy.int64)
    s.bestIndividual = nbest
    
    return s
    
    
    
    
    
