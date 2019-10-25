# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""

from sklearn import metrics, preprocessing

import CSSA as cssa
import CPSO as cpso
import CGA as cga
import CBAT as cbat
import CFFA as cffa
import CGWO as cgwo
import CWOA as cwoa
import CMVO as cmvo
import CMFO as cmfo
import CCS as ccs
import objectives
import measures
import os
import numpy
import statistics as stat
import warnings
import time
import csv

warnings.simplefilter(action='ignore')


def readDataset(directory, filename):
    """
    Reads the dataset file and store a list of Point ............

    Parameters
    ----------
    directory : str
        The file location of the dataset
    filename : str
        The dataset file name

    Returns
    -------
    N/A
    """
    myPath = os.path.abspath(os.path.dirname(__file__))
    rawData = open(os.path.join(myPath, directory + filename), 'rt')
    data = numpy.loadtxt(rawData, delimiter=",")
    return data



def selector(algo,func_details, k, f, popSize,Iter, points):
    function_name=func_details[0]
    lb=0
    ub=1
    
    if(algo==0):
        x=cssa.SSA(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==1):
        x=cpso.PSO(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==2):
        x=cga.GA(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==3):
        x=cbat.BAT(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==4):
        x=cffa.FFA(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==5):
        x=cgwo.GWO(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==6):
        x=cwoa.WOA(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==7):
        x=cmvo.MVO(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==8):
        x=cmfo.MFO(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    if(algo==9):
        x=ccs.CS(getattr(objectives, function_name),lb,ub,k * f,popSize,Iter, k, points)        
    return x
    

# Select optimizers
CSSA= False
CPSO= False
CGA= False
CBAT= False
CFFA=False
CGWO=False
CWOA=False
CMVO=False
CMFO=False
CCS=True

# Select objective function
SSE=True
TWCV=False
SC=False
DB=False
#CH=True
DI=False


optimizer=[CSSA, CPSO, CGA, CBAT, CFFA, CGWO, CWOA, CMVO, CMFO, CCS]
objectivefunc=[SSE, TWCV, SC, DB, DI] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=10

# Select general parameters for all optimizers (population size, number of iterations) ....
PopulationSize = 50
Iterations= 100
directory = "datasets/" # the directory where the dataset is stored
plot = False

#Export results ?
Export=True
Export_details=True

#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment "+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 
ExportToFileDetails="experiment "+time.strftime("%Y-%m-%d-%H-%M-%S")+"_details.csv" 
ExportToFileDetailsLabels="experiment "+time.strftime("%Y-%m-%d-%H-%M-%S")+"_details_Labels.csv" 

# Check if it works at least once
Flag=False
Flag_details=False
Flag_details_Labels=False

# CSV Header for for the cinvergence 
CnvgHeader=[]


dataset_List = ["aggregation.csv","aniso.csv","appendicitisNorm.csv", "balance.csv",
                "banknote.csv", "blobs.csv","Blood.csv","circles.csv","diagnosis_II.csv",
                "ecoli.csv","flame.csv","glass.csv","heart.csv","ionosphere.csv",
                "iris.csv","iris2D.csv","jain.csv","liver.csv","moons.csv",
                "mouse.csv","pathbased.csv","seeds.csv","smiley.csv","sonar.csv",
                "varied.csv","vary-density.csv","vertebral2.csv","vertebral3.csv",
                "wdbc.csv","wine.csv"]
                

dataset_len = len(dataset_List)

k = [-1] * dataset_len
f = [-1] * dataset_len
points= [0] * dataset_len
labelsTrue = [0] * dataset_len

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))
  
#read all datasets
for h in range(dataset_len):
        
    # Read the dataset file and generate the points list and true values
    data = readDataset(directory, dataset_List[h])
    
    nPoints, nValues = data.shape #Number of points and Number of values for each point
    f[h] = nValues - 1 #Dimension value
    k[h] = len(numpy.unique(data[:,-1]))#k: Number of clusters
    points[h] = data[:,:-1].tolist() #list of points
    labelsTrue[h] = data[:,-1].tolist() #List of actual cluster of each points (last field)
    
    #points =(preprocessing.normalize(points, norm='max', axis=0) * 200) - 100
    points[h] =preprocessing.normalize(points[h], norm='max', axis=0)

for i in range (0, len(optimizer)):
    for j in range (0, len(objectivefunc)):
        if((optimizer[i]==True) and (objectivefunc[j]==True)): # start experiment if an optimizer and an objective function is selected
           for h in range(len(dataset_List)):     
                HS = [0]*NumOfRuns     
                CS = [0]*NumOfRuns
                VM = [0]*NumOfRuns 
                AMI = [0]*NumOfRuns 
                ARI = [0]*NumOfRuns    
                Fmeasure = [0]*NumOfRuns   
                SC = [0]*NumOfRuns   
                accuracy = [0]*NumOfRuns   
                DI = [0]*NumOfRuns   
                DB = [0]*NumOfRuns   
                stdev = [0]*NumOfRuns   
                exSSE = [0]*NumOfRuns 
                exTWCV = [0]*NumOfRuns
                purity = [0]*NumOfRuns
                entropy = [0]*NumOfRuns
                convergence = [0]*NumOfRuns
                executionTime = [0]*NumOfRuns
                #Agg = [0]*NumOfRuns
                
                for z in range (0,NumOfRuns):
                    print("Run no.: " + str(z)) 
                    
                    func_details=objectives.getFunctionDetails(j)
                    x=selector(i,func_details, k[h], f[h], PopulationSize,Iterations, points[h])
                    
                    
                    HS[z] = measures.HS(labelsTrue[h],x.labelsPred)
                    CS[z] = measures.CS(labelsTrue[h],x.labelsPred)
                    VM[z] = measures.VM(labelsTrue[h],x.labelsPred)
                    AMI[z] = measures.AMI(labelsTrue[h],x.labelsPred)
                    ARI[z] = measures.ARI(labelsTrue[h],x.labelsPred)
                    Fmeasure[z] = measures.Fmeasure(labelsTrue[h],x.labelsPred)
                    SC[z] = measures.SC(points[h],x.labelsPred)
                    accuracy[z] = measures.accuracy(labelsTrue[h],x.labelsPred)
                    DI[z] = measures.DI(points[h], x.labelsPred)
                    DB[z] = measures.DB(points[h], x.labelsPred)
                    stdev[z] = measures.stdev(x.bestIndividual,x.labelsPred, k[h], points[h])
                    exSSE[z] = measures.SSE(x.bestIndividual, x.labelsPred, k[h], points[h])
                    exTWCV[z] = measures.TWCV(x.bestIndividual, x.labelsPred, k[h], points[h])
                    purity[z] = measures.purity(labelsTrue[h],x.labelsPred)
                    entropy[z] = measures.entropy(labelsTrue[h],x.labelsPred)
                    #Agg[z] = float("%0.2f"%(float("%0.2f"%(HS[z] + CS[z] + VM[z] + AMI[z] + ARI[z])) / 5))
                    
                    executionTime[z] = x.executionTime
                    convergence[z] = x.convergence
                    optimizerName = x.optimizer
                    objfname = x.objfname
                    
                    
                    with open(ExportToFileDetailsLabels, 'a',newline='\n') as out_details_labels:
                        writer_details = csv.writer(out_details_labels,delimiter=',')
                        if (Flag_details_Labels==False): # just one time to write the header of the CSV file
                            header_details= numpy.concatenate([["Dataset", "Optimizer","objfname"]])
                            writer_details.writerow(header_details)
                            Flag_details_Labels = True
                        a=numpy.concatenate([[dataset_List[h], optimizerName, objfname],x.labelsPred])  
                        writer_details.writerow(a)
                    out_details_labels.close()
                    
                    with open(ExportToFileDetails, 'a',newline='\n') as out_details:
                        writer_details = csv.writer(out_details,delimiter=',')
                        if (Flag_details==False): # just one time to write the header of the CSV file
                            header_details= numpy.concatenate([["Dataset", "Optimizer","objfname","ExecutionTime","SSE","Purity","Entropy","HS","CS","VM","AMI","ARI","Fmeasure","TWCV","SC","Accuracy","DI","DB","STDev"],CnvgHeader])
                            writer_details.writerow(header_details)
                            Flag_details = True
                        a=numpy.concatenate([[dataset_List[h], optimizerName, objfname, float("%0.2f"%(executionTime[z])), 
                              float("%0.2f"%(exSSE[z])), float("%0.2f"%(purity[z])), float("%0.2f"%(entropy[z])), float("%0.2f"%(HS[z])), 
                              float("%0.2f"%(CS[z])),  float("%0.2f"%(VM[z])),  float("%0.2f"%(AMI[z])),  float("%0.2f"%(ARI[z])), 
                              float("%0.2f"%(Fmeasure[z])),  float("%0.2f"%(exTWCV[z])),  float("%0.2f"%(SC[z])),  float("%0.2f"%(accuracy[z])),  float("%0.2f"%(DI[z])), 
                              float("%0.2f"%(DB[z])), float("%0.2f"%(stdev[z]))],numpy.around(convergence[z],decimals=2)])  
                        writer_details.writerow(a)
                    out_details.close()
                    '''    
                    print(dataset_List[h] + "," + optimizerName + "," + objfname + "," + str(executionTime[z]) + "," + 
                          str(exSSE[z]) + "," + str(purity[z]) + "," + str(entropy[z]) + "," + str(HS[z]) + "," + 
                          str(CS[z]) + "," +  str(VM[z]) + "," +  str(AMI[z]) + "," +  str(ARI[z]) + "," + 
                          str(Fmeasure[z]) + "," +  str(exTWCV[z]) + "," +  str(SC[z]) + "," +  str(accuracy[z]) + "," +  str(DI[z]) + "," + 
                          str(DB[z]) + "," +  str(stdev[z]) + "," + ",".join(str(x) for x in convergence[z]))
                    '''
            
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Dataset", "Optimizer","objfname","ExecutionTime","SSE","Purity","Entropy","HS","CS","VM","AMI","ARI","Fmeasure","TWCV","SC","Accuracy","DI","DB","STDev"],CnvgHeader])
                            writer.writerow(header)
                                        
                        avgSSE = str(float("%0.2f"%(sum(exSSE) / NumOfRuns)))
                        avgTWCV = str(float("%0.2f"%(sum(exTWCV) / NumOfRuns)))
                        avgPurity = str(float("%0.2f"%(sum(purity) / NumOfRuns)))
                        avgEntropy = str(float("%0.2f"%(sum(entropy) / NumOfRuns)))
                        avgHomo = str(float("%0.2f"%(sum(HS) / NumOfRuns)))
                        avgComp = str(float("%0.2f"%(sum(CS) / NumOfRuns)))
                        avgVmeas = str(float("%0.2f"%(sum(VM) / NumOfRuns)))
                        avgAMI = str(float("%0.2f"%(sum(AMI) / NumOfRuns)))
                        avgARI = str(float("%0.2f"%(sum(ARI) / NumOfRuns)))
                        avgFmeasure = str(float("%0.2f"%(sum(Fmeasure) / NumOfRuns)))
                        avgSC = str(float("%0.2f"%(sum(SC) / NumOfRuns)))
                        avgAccuracy = str(float("%0.2f"%(sum(accuracy) / NumOfRuns)))
                        avgDI = str(float("%0.2f"%(sum(DI) / NumOfRuns)))
                        avgDB = str(float("%0.2f"%(sum(DB) / NumOfRuns)) )    
                        avgStdev = str(float("%0.2f"%(sum(stdev) / NumOfRuns)))                
                        #avgAgg = str(float("%0.2f"%(sum(Agg) / NumOfRuns)))
                        
                        avgExecutionTime = float("%0.2f"%(sum(executionTime) / NumOfRuns))
                        avgConvergence = numpy.around(numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2).tolist()
                        a=numpy.concatenate([[dataset_List[h], optimizerName,objfname,avgExecutionTime,avgSSE,avgPurity,avgEntropy,avgHomo, avgComp, avgVmeas, avgAMI, avgARI, avgFmeasure, avgTWCV, avgSC, avgAccuracy, avgDI, avgDB, avgStdev],avgConvergence])
                        writer.writerow(a)
                    out.close()
                Flag=True # at least one experiment
                    
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
