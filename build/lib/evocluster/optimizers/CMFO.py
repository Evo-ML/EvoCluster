# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:42:18 2016

@author: hossam
"""

import random
import numpy
import math
from ..solution import solution
import time



  
def MFO(objf,lb,ub,dim,N,Max_iteration,k,points, metric):


    #Max_iteration=1000
    #lb=-100
    #ub=100
    #dim=30
    N=50 # Number of search agents
    
    
    
    #Initialize the positions of moths
    Moth_pos=numpy.random.uniform(0,1,(N,dim)) *(ub-lb)+lb
    Moth_fitness=numpy.full(N,float("inf"))
    Moth_labels=numpy.zeros((N,len(points)))
    #Moth_fitness=numpy.fell(float("inf"))
    
    Convergence_curve=numpy.zeros(Max_iteration)
    
    
    sorted_population=numpy.copy(Moth_pos)
    sorted_labels=numpy.copy(Moth_labels)
    fitness_sorted=numpy.zeros(N)
    #####################
    best_flames=numpy.copy(Moth_pos)
    best_labels=numpy.copy(Moth_labels)
    best_flame_fitness=numpy.zeros(N)
    ####################
    double_population=numpy.zeros((2*N,dim))
    double_labels=numpy.zeros((2*N,len(points)))
    double_fitness=numpy.zeros(2*N)
    
    double_sorted_population=numpy.zeros((2*N,dim))
    double_sorted_labels=numpy.zeros((2*N,len(points)))
    double_fitness_sorted=numpy.zeros(2*N)
    #########################
    previous_population=numpy.zeros((N,dim))
    previous_labels=numpy.zeros((N,len(points)))
    previous_fitness=numpy.zeros(N)


    s=solution()

    print("MFO is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    Iteration=1;    
    
    # Main loop
    while (Iteration<Max_iteration):
        
        # Number of flames Eq. (3.14) in the paper
        Flame_no=round(N-Iteration*((N-1)/Max_iteration));
        
        for i in range(0,N):
            
            # Check if moths go out of the search spaceand bring it back
            Moth_pos[i,:]=numpy.clip(Moth_pos[i,:], lb, ub)

            # evaluate moths
            startpts = numpy.reshape(Moth_pos[i,:], (k,(int)(dim/k)))
            if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
                fitnessValue, labelsPredValues=objf(startpts, points, k, metric) 
            else:
                fitnessValue, labelsPredValues=objf(startpts, points, k) 
            Moth_fitness[i] = fitnessValue
            Moth_labels[i,:] = labelsPredValues
           
            
        
           
        if Iteration==1:
            # Sort the first population of moths
            fitness_sorted=numpy.sort(Moth_fitness)
            I=numpy.argsort(Moth_fitness)
            
            sorted_population=Moth_pos[I,:]
            sorted_labels=Moth_labels[I,:]
               
            
            #Update the flames
            best_flames=sorted_population
            best_flame_fitness=fitness_sorted
            best_labels=sorted_labels
        else:
    #        
    #        # Sort the moths
            double_population=numpy.concatenate((previous_population,best_flames),axis=0)
            double_labels=numpy.concatenate((previous_labels,best_labels),axis=0)
            double_fitness=numpy.concatenate((previous_fitness, best_flame_fitness),axis=0);
    #        
            double_fitness_sorted =numpy.sort(double_fitness);
            I2 =numpy.argsort(double_fitness);
    #        
    #        
            for newindex in range(0,2*N):
                double_sorted_population[newindex,:]=numpy.array(double_population[I2[newindex],:])           
                double_sorted_labels[newindex,:]=numpy.array(double_labels[I2[newindex],:])           
            
            fitness_sorted=double_fitness_sorted[0:N]
            sorted_population=double_sorted_population[0:N,:]
            sorted_labels=double_sorted_labels[0:N,:]
    #        
    #        # Update the flames
            best_flames=sorted_population;
            best_labels=sorted_labels;
            best_flame_fitness=fitness_sorted;
    
    #    
    #   # Update the position best flame obtained so far
        Best_flame_score=fitness_sorted[0]
        Best_flame_pos=sorted_population[0,:]
        Best_labelsPred=sorted_labels[0,:]
    #      
        previous_population=Moth_pos;
        previous_labels=Moth_labels;
        previous_fitness=Moth_fitness;
    #    
        # a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a=-1+Iteration*((-1)/Max_iteration);
        

        
        # Loop counter
        for i in range(0,N):
    #        
            for j in range(0,dim):
                if (i<=Flame_no): #Update the position of the moth with respect to its corresponsing flame
    #                
                    # D in Eq. (3.13)
                    distance_to_flame=abs(sorted_population[i,j]-Moth_pos[i,j])
                    b=1
                    t=(a-1)*random.random()+1;
    #                
    #                % Eq. (3.12)
                    Moth_pos[i,j]=distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi)+sorted_population[i,j]
    #            end
    #            
                if i>Flame_no: # Upaate the position of the moth with respct to one flame
    #                
    #                % Eq. (3.13)
                    distance_to_flame=abs(sorted_population[i,j]-Moth_pos[i,j]);
                    b=1;
                    t=(a-1)*random.random()+1;
    #                
    #                % Eq. (3.12)
                    Moth_pos[i,j]=distance_to_flame*math.exp(b*t)*math.cos(t*2*math.pi)+sorted_population[Flame_no,j]
        
        Convergence_curve[Iteration]=Best_flame_score
        #Display best fitness along the iteration
        if (Iteration%1==0):
            print(['At iteration '+ str(Iteration)+ ' the best fitness is '+ str(Best_flame_score)]);
    
    
    
    
        Iteration=Iteration+1; 
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="MFO"   
    s.objfname=objf.__name__
    s.bestIndividual = Best_flame_pos
    s.labelsPred = numpy.array(Best_labelsPred, dtype=numpy.int64)
    
    
    
    return s
    



