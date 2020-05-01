# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""

from sklearn import preprocessing
from pathlib import Path

import optimizers.CSSA as cssa
import optimizers.CPSO as cpso
import optimizers.CGA as cga
import optimizers.CBAT as cbat
import optimizers.CFFA as cffa
import optimizers.CGWO as cgwo
import optimizers.CWOA as cwoa
import optimizers.CMVO as cmvo
import optimizers.CMFO as cmfo
import optimizers.CCS as ccs
import objectives
import measures
import os
import numpy
import warnings
import time
import csv
import plot_convergence as conv_plot
import plot_boxplot as box_plot

warnings.simplefilter(action='ignore')

def selector(algo,objective_name, k, f, popSize,Iter, points):
	"""
	This is used to call the algorithm which is selected

	Parameters
	----------
	algo : int
	     The index of the selected algorithm
	objective_name : str
	     The name of the selected function
	k : int
	     Number of clusters
	f : int
	     Number of features
	popSize : int
	     Size of population (the number of individuals at each iteration)
	Iter : int
	     The number of iterations
	points : numpy.ndaarray
	     The attribute values of all the points

	Returns
	-----------
	obj
	     x: solution object returned by the selected algorithm
	"""
	lb=0
	ub=1

	if(algo=="SSA"):
		x=cssa.SSA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="PSO"):
		x=cpso.PSO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="GA"):
		x=cga.GA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="BAT"):
		x=cbat.BAT(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="FFA"):
		x=cffa.FFA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="GWO"):
		x=cgwo.GWO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="WOA"):
		x=cwoa.WOA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="MVO"):
		x=cmvo.MVO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="MFO"):
		x=cmfo.MFO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	if(algo=="CS"):
		x=ccs.CS(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points)        
	return x
		
def run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags):
	
	"""
	It serves as the main interface of the framework for running the experiments.

	Parameters
	----------    
	optimizer : list
	    The list of optimizers names
	objectivefunc : list
	    The list of boolean preference of objective functions
	dataset_List : list
	    The list of the names of the data sets files
	NumOfRuns : int
	    The number of independent runs 
	params  : set
	    The set of parameters which are: 
	    1. Size of population (PopulationSize)
	    2. The number of iterations (Iterations)
	export_flags : set
	    The set of Boolean flags which are: 
	    1. Export (Exporting the results in a file)
	    2. Export_details (Exporting the detailed results in files)
	    3. Export_details_labels (Exporting the labels detailed results in files)
	    4. Export_convergence (Exporting the covergence plots)
	    5. Export_boxplot (Exporting the box plots)

	Returns
	-----------
	N/A
	"""
	
	# Select general parameters for all optimizers (population size, number of iterations) ....
	PopulationSize = params['PopulationSize']
	Iterations= params['Iterations']

	#Export results ?
	Export=export_flags['Export_avg']
	Export_details=export_flags['Export_details']
	Export_details_labels = export_flags['Export_details_labels']
	Export_convergence = export_flags['Export_convergence']
	Export_boxplot = export_flags['Export_boxplot']

	#Automaticly generated name by date and time

	# Check if it works at least once
	Flag=False
	Flag_details=False
	Flag_details_Labels=False

	# CSV Header for for the cinvergence 
	CnvgHeader=[]


	datasets_directory = "datasets/" # the directory where the dataset is stored
	results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + '/'
	Path(results_directory).mkdir(parents=True, exist_ok=True)

	dataset_len = len(dataset_List)

	k = [-1] * dataset_len
	f = [-1] * dataset_len
	points= [0] * dataset_len
	labelsTrue = [0] * dataset_len

	for l in range(0,Iterations):
		CnvgHeader.append("Iter"+str(l+1))
	  
	#read all datasets
	for h in range(dataset_len):
			
		dataset_filename = dataset_List[h] + '.csv'		
		# Read the dataset file and generate the points list and true values 
		rawData = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), datasets_directory + dataset_filename), 'rt')
		data = numpy.loadtxt(rawData, delimiter=",")
		
		
		nPoints, nValues = data.shape #Number of points and Number of values for each point
		f[h] = nValues - 1 #Dimension value
		k[h] = len(numpy.unique(data[:,-1]))#k: Number of clusters
		points[h] = data[:,:-1].tolist() #list of points
		labelsTrue[h] = data[:,-1].tolist() #List of actual cluster of each points (last field)
		
		points[h] =preprocessing.normalize(points[h], norm='max', axis=0)


	for i in range (0, len(optimizer)):
	    for j in range (0, len(objectivefunc)):
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
                    print("Dataset: " + dataset_List[h])
                    print("Run no.: " + str(z)) 
                    print("Population Size: " + str(PopulationSize)) 
                    print("Iterations: " + str(Iterations)) 
                    
                    objective_name=objectivefunc[j]
                    x=selector(optimizer[i],objective_name, k[h], f[h], PopulationSize,Iterations, points[h])
                    
                    
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



                    if(Export_details_labels==True):
                    	ExportToFileDetailsLabels=results_directory + "experiment_details_Labels.csv"
                    	with open(ExportToFileDetailsLabels, 'a',newline='\n') as out_details_labels:
                            writer_details = csv.writer(out_details_labels,delimiter=',')
                            if (Flag_details_Labels==False): # just one time to write the header of the CSV file
                                header_details= numpy.concatenate([["Dataset", "Optimizer","objfname"]])
                                writer_details.writerow(header_details)
                                Flag_details_Labels = True
                            a=numpy.concatenate([[dataset_List[h], optimizerName, objfname],x.labelsPred])  
                            writer_details.writerow(a)
                    	out_details_labels.close()                            

                    if(Export_details==True):
                        ExportToFileDetails=results_directory + "experiment_details.csv"
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
                    
            
                if(Export==True):
                	ExportToFile=results_directory + "experiment.csv"

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

	if Export_convergence == True:
		conv_plot.run(results_directory, optimizer, objectivefunc, dataset_List, Iterations)
    
	
	if Export_boxplot == True:
		ev_measures=['SSE','Purity','Entropy', 'HS', 'CS', 'VM', 'AMI', 'ARI', 'Fmeasure', 'TWCV', 'SC', 'Accuracy', 'DI', 'DB', 'STDev']
		box_plot.run(results_directory, optimizer, objectivefunc, dataset_List, ev_measures, Iterations)

	if (Flag==False): # Faild to run at least one experiment
	    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
