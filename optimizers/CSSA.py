import random
import numpy
import math
from solution import solution

from time import sleep
import time
import sys

  
def SSA(objf,lb,ub,dim,N,Max_iteration, k, points, metric):

    #Max_iteration=1000
    #lb=-100
    #ub=100
    #dim=30
    #N=50 # Number of search agents
    
    Convergence_curve=numpy.zeros(Max_iteration)

        
    #Initialize the positions of salps
    SalpPositions=numpy.random.uniform(0,1,(N,dim)) *(ub-lb)+lb
    SalpFitness=numpy.full(N,float("inf"))
    SalpLabelsPred=numpy.full((N,len(points)), numpy.inf)
    
    FoodPosition=numpy.zeros(dim)
    FoodFitness=float("inf")
    FoodLabelsPred=numpy.full(len(points), numpy.inf)
    #Moth_fitness=numpy.fell(float("inf"))
    
    s=solution()

    print("SSA is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for i in range(0,N):
       # evaluate moths
        startpts = numpy.reshape(SalpPositions[i,:], (k,(int)(dim/k)))

        if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
            fitness, labelsPred=objf(startpts, points, k, metric) 
        else:
            fitness, labelsPred=objf(startpts, points, k) 
        SalpFitness[i] = fitness
        SalpLabelsPred[i,:] = labelsPred
        
    sorted_salps_fitness=numpy.sort(SalpFitness)
    I=numpy.argsort(SalpFitness)
    
    Sorted_salps=numpy.copy(SalpPositions[I,:])
    Sorted_LabelsPred=numpy.copy(SalpLabelsPred[I,:])
       
    
    FoodPosition=numpy.copy(Sorted_salps[0,:])
    FoodFitness=sorted_salps_fitness[0]
    FoodLabelsPred=Sorted_LabelsPred[0]
    '''
    Convergence_curve[0]=FoodFitness
    print(['At iteration 0 the best fitness is '+ str(FoodFitness)])
    '''
    Iteration=1;    
    
    # Main loop
    while (Iteration<Max_iteration):
        
        #sleep(1)
    
        # Number of flames Eq. (3.14) in the paper
        #Flame_no=round(N-Iteration*((N-1)/Max_iteration));
        
        c1 = 2*math.exp(-(4*Iteration/Max_iteration)**2); # Eq. (3.2) in the paper

        for i in range(0,N):
            
            SalpPositions= numpy.transpose(SalpPositions);

            if i<N/2:
                for j in range(0,dim):
                    c2=random.random()
                    c3=random.random()
                    #Eq. (3.1) in the paper 
                    if c3<0.5:
                        SalpPositions[j,i]=FoodPosition[j]+0.1*c1*((ub-lb)*c2+lb);
                    else:
                        SalpPositions[j,i]=FoodPosition[j]-0.1*c1*((ub-lb)*c2+lb);
                    
                    ####################
            
            
            elif i>=N/2 and i<N:
                point1=numpy.copy(SalpPositions[:,i-1]);
                point2=numpy.copy(SalpPositions[:,i]);
                
                SalpPositions[:,i]=(point2+point1)/2; # Eq. (3.4) in the paper
        
        
            SalpPositions= numpy.transpose(SalpPositions);
        
           
        for i in range(0,N):
            
        
           # Check if salps go out of the search spaceand bring it back
            SalpPositions[i,:]=numpy.clip(SalpPositions[i,:], lb, ub)            
            
            startpts = numpy.reshape(SalpPositions[i,:], (k,(int)(dim/k)))
        
            if objf.__name__ == 'SSE' or objf.__name__ == 'SC' or objf.__name__ == 'DI':
                fitness, labelsPred=objf(startpts, points, k, metric) 
            else:
                fitness, labelsPred=objf(startpts, points, k) 
                
            SalpFitness[i] = fitness
            SalpLabelsPred[i,:] = numpy.copy(labelsPred)
        
             
            if SalpFitness[i]<FoodFitness:
                FoodPosition=numpy.copy(SalpPositions[i,:])
                FoodFitness=SalpFitness[i]
                FoodLabelsPred = numpy.copy(SalpLabelsPred[i,:])
                
                
        Convergence_curve[Iteration]=FoodFitness
        #Display best fitness along the iteration
        
        
        if (Iteration%1==0):
            print(['At iteration '+ str(Iteration)+ ' the best fitness is '+ str(FoodFitness)]);
        
    
    
    
        Iteration=Iteration+1; 
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=Convergence_curve
    s.optimizer="SSA"   
    s.objfname=objf.__name__
    s.labelsPred = numpy.array(FoodLabelsPred, dtype=numpy.int64)
    s.bestIndividual = FoodPosition
    
    return s
    




