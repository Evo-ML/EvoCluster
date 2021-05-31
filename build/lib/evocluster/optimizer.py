# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""

from sklearn import preprocessing
from pathlib import Path

from .optimizers import CSSA as cssa
from .optimizers import CPSO as cpso
from .optimizers import CGA as cga
from .optimizers import CBAT as cbat
from .optimizers import CFFA as cffa
from .optimizers import CGWO as cgwo
from .optimizers import CWOA as cwoa
from .optimizers import CMVO as cmvo
from .optimizers import CMFO as cmfo
from .optimizers import CCS as ccs
from . import objectives as objectives
from . import measures as measures
import os
import sys
import numpy
import warnings
import time
import csv
from . import plot_convergence as conv_plot
from . import plot_boxplot as box_plot
from . import cluster_detection as clus_det

warnings.simplefilter(action='ignore')
		
def run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags, 
	auto_cluster = True, n_clusters = 'supervised', labels_exist = True, metric='euclidean'):
	
	"""
	It serves as the main interface of the framework for running the experiments.

	Parameters
	----------    
	optimizer : list
	    The list of optimizers names
	objectivefunc : list
	    The list of objective functions
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
	auto_cluster : boolean, default = True
		Choose whether the number of clusters is detected automatically. 
		If True, select one of the following: 'supervised', 'CH', 'silhouette', 'elbow', 'gap', 'min', 'max', 'median' for n_clusters. 
		If False, specify a list of integers for n_clusters. 
	n_clusters : string, or list, default = 'supervised'
		A list of the number of clusters for the datasets in dataset_List
		Other values can be considered instead of specifying the real value, which are as follows:
		- supervised: The number of clusters is derived from the true labels of the datasets
		- elbow: The number of clusters is automatically detected by elbow method
		- gap: The number of clusters is automatically detected by gap analysis methos
		- silhouette: The number of clusters is automatically detected by silhouette coefficient method
		- CH: The number of clusters is automatically detected by Calinski-Harabasz index
		- DB: The number of clusters is automatically detected by Davies Bouldin index
		- BIC: The number of clusters is automatically detected by Bayesian Information Criterion score
		- min: The number of clusters is automatically detected by the minimum value of the number of clusters detected by all detection techniques
		- max: The number of clusters is automatically detected by the maximum value of the number of clusters detected by all detection techniques
		- median: The number of clusters is automatically detected by the median value of the number of clusters detected by all detection techniques
		- majority: The number of clusters is automatically detected by the majority vote of the number of clusters detected by all detection techniques
	labels_exist : boolean, default = True
		Specify if labels exist as the last column of the csv file of the datasets in dataset_List
		if the value is False, the following hold:
		- supervised value for n_clusters is not allowed
		- experiments, and experiments_details files contain only the evaluation measures for 
		  "SSE","TWCV","SC","DB","DI","STDev"
		- Export_boxplot is set for "SSE","TWCV","SC","DB","DI","STDev"   
	metric : string, default = 'euclidean'
		The metric to use when calculating the distance between points if applicable for the objective function selected. 
		It must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter

	
	Returns
	-----------
	N/A
	"""

	if not labels_exist and n_clusters == 'supervised':
		print('Syupervised value for n_clusters is not allowed when labels_exist value is false')
		sys.exit()

	if isinstance(n_clusters, list):
		if len(n_clusters) != len(dataset_List):
			print('Length of n_clusters list should equal the length of dataset_List list')
			sys.exit()
		if min(n_clusters) < 2:
			print('n_clusters value should be larger than 2')
			sys.exit()
		if auto_cluster == True:
			print('n_clusters should be string if auto_cluster is true')
			sys.exit()
	else:
		if auto_cluster == False:
			print('n_clusters should be a list of integers if auto_cluster is false')
			sys.exit()

	
	# Select general parameters for all optimizers (population size, number of iterations) ....
	PopulationSize = params['PopulationSize']
	Iterations= params['Iterations']

	#Export results ?
	Export=export_flags['Export_avg']
	Export_details=export_flags['Export_details']
	Export_details_labels = export_flags['Export_details_labels']
	Export_convergence = export_flags['Export_convergence']
	Export_boxplot = export_flags['Export_boxplot']

	# Check if it works at least once
	Flag=False
	Flag_details=False
	Flag_details_Labels=False

	# CSV Header for for the cinvergence 
	CnvgHeader=[]
	
	if labels_exist:
		datasets_directory = "../datasets/" # the directory where the dataset is stored
	else:
		datasets_directory = "../datasets/unsupervised/" # the directory where the dataset is stored
		
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

		if labels_exist:			
			f[h] = nValues - 1 #Dimension value
			points[h] = data[:,:-1].tolist() #list of points
			labelsTrue[h] = data[:,-1].tolist() #List of actual cluster of each points (last field)
		else:
			f[h] = nValues#Dimension value
			points[h] = data.copy().tolist() #list of points		
			labelsTrue[h] = None #List of actual cluster of each points (last field)

		points[h] =preprocessing.normalize(points[h], norm='max', axis=0)

		if n_clusters == 'supervised':		
			k[h] = len(numpy.unique(data[:,-1]))#k: Number of clusters
		elif n_clusters == 'elbow':
			k[h] = clus_det.ELBOW(points[h])#k: Number of clusters		
		elif n_clusters == 'gap':
			k[h] = clus_det.GAP_STATISTICS(points[h])#k: Number of clusters		
		elif n_clusters == 'silhouette':
			k[h] = clus_det.SC(points[h])#k: Number of clusters		
		elif n_clusters == 'DB':
			k[h] = clus_det.DB(points[h])#k: Number of clusters		
		elif n_clusters == 'CH':
			k[h] = clus_det.CH(points[h])#k: Number of clusters		
		elif n_clusters == 'DB':
			k[h] = clus_det.DB(points[h])#k: Number of clusters		
		elif n_clusters == 'BIC':
			k[h] = clus_det.BIC(points[h])#k: Number of clusters	
		elif n_clusters == 'min':
			k[h] = clus_det.min_clusters(points[h])#k: Number of clusters
		elif n_clusters == 'max':
			k[h] = clus_det.max_clusters(points[h])#k: Number of clusters		
		elif n_clusters == 'median':
			k[h] = clus_det.median_clusters(points[h])#k: Number of clusters
		elif n_clusters == 'majority':
			k[h] = clus_det.majority_clusters(points[h])#k: Number of clusters			
		else:
			k[h] = n_clusters[h]#k: Number of clusters	


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
                    print("k: " + str(k[h])) 
                    print("Run no.: " + str(z)) 
                    print("Population Size: " + str(PopulationSize)) 
                    print("Iterations: " + str(Iterations)) 
                    
                    objective_name=objectivefunc[j]
                    x=selector(optimizer[i],objective_name, k[h], f[h], PopulationSize,Iterations, points[h], metric)
                    
                    if labels_exist:
	                    HS[z] = measures.HS(labelsTrue[h],x.labelsPred)
	                    CS[z] = measures.CS(labelsTrue[h],x.labelsPred)
	                    VM[z] = measures.VM(labelsTrue[h],x.labelsPred)
	                    AMI[z] = measures.AMI(labelsTrue[h],x.labelsPred)
	                    ARI[z] = measures.ARI(labelsTrue[h],x.labelsPred)
	                    Fmeasure[z] = measures.Fmeasure(labelsTrue[h],x.labelsPred)
	                    accuracy[z] = measures.accuracy(labelsTrue[h],x.labelsPred)
	                    purity[z] = measures.purity(labelsTrue[h],x.labelsPred)
	                    entropy[z] = measures.entropy(labelsTrue[h],x.labelsPred)
	                    #Agg[z] = float("%0.2f"%(float("%0.2f"%(HS[z] + CS[z] + VM[z] + AMI[z] + ARI[z])) / 5))
                    
                    SC[z] = measures.SC(points[h],x.labelsPred)
                    DI[z] = measures.DI(points[h], x.labelsPred)
                    DB[z] = measures.DB(points[h], x.labelsPred)
                    stdev[z] = measures.stdev(x.bestIndividual,x.labelsPred, k[h], points[h])
                    exSSE[z] = measures.SSE(x.bestIndividual, x.labelsPred, k[h], points[h])
                    exTWCV[z] = measures.TWCV(x.bestIndividual, x.labelsPred, k[h], points[h])

                    executionTime[z] = x.executionTime
                    convergence[z] = x.convergence
                    optimizerName = x.optimizer
                    objfname = x.objfname

                    if(Export_details_labels==True):
                    	ExportToFileDetailsLabels=results_directory + "experiment_details_Labels.csv"
                    	with open(ExportToFileDetailsLabels, 'a',newline='\n') as out_details_labels:
                            writer_details = csv.writer(out_details_labels,delimiter=',')
                            if (Flag_details_Labels==False): # just one time to write the header of the CSV file
                                header_details= numpy.concatenate([["Dataset", "Optimizer","objfname","k"]])
                                writer_details.writerow(header_details)
                                Flag_details_Labels = True
                            a=numpy.concatenate([[dataset_List[h], optimizerName, objfname, k[h]],x.labelsPred])  
                            writer_details.writerow(a)
                    	out_details_labels.close()                            

                    if(Export_details==True):
                        ExportToFileDetails=results_directory + "experiment_details.csv"
                        with open(ExportToFileDetails, 'a',newline='\n') as out_details:
                            writer_details = csv.writer(out_details,delimiter=',')
                            if (Flag_details==False): # just one time to write the header of the CSV file
                            	if labels_exist:
                            		header_details= numpy.concatenate([["Dataset", "Optimizer","objfname","k","ExecutionTime","SSE","Purity","Entropy","HS","CS","VM","AMI","ARI","Fmeasure","TWCV","SC","Accuracy","DI","DB","STDev"],CnvgHeader])
                            	else:
                                	header_details= numpy.concatenate([["Dataset", "Optimizer","objfname","k","ExecutionTime","SSE","TWCV","SC","DI","DB","STDev"],CnvgHeader])
                            	writer_details.writerow(header_details)
                            	Flag_details = True
                            if labels_exist:
                            	a=numpy.concatenate([[dataset_List[h], optimizerName, objfname, k[h], float("%0.2f"%(executionTime[z])), float("%0.2f"%(exSSE[z])), float("%0.2f"%(purity[z])), float("%0.2f"%(entropy[z])), float("%0.2f"%(HS[z])), float("%0.2f"%(CS[z])),  float("%0.2f"%(VM[z])),  float("%0.2f"%(AMI[z])),  float("%0.2f"%(ARI[z])), float("%0.2f"%(Fmeasure[z])),  float("%0.2f"%(exTWCV[z])),  float("%0.2f"%(SC[z])),  float("%0.2f"%(accuracy[z])),  float("%0.2f"%(DI[z])), float("%0.2f"%(DB[z])), float("%0.2f"%(stdev[z]))],numpy.around(convergence[z],decimals=2)])
                            else:
	                            a=numpy.concatenate([[dataset_List[h], optimizerName, objfname, k[h], float("%0.2f"%(executionTime[z])), float("%0.2f"%(exSSE[z])), float("%0.2f"%(exTWCV[z])),  float("%0.2f"%(SC[z])),  float("%0.2f"%(DI[z])), float("%0.2f"%(DB[z])), float("%0.2f"%(stdev[z]))],numpy.around(convergence[z],decimals=2)])  

                            writer_details.writerow(a)
                        out_details.close()
            
                if(Export==True):
                	ExportToFile=results_directory + "experiment.csv"

                	with open(ExportToFile, 'a',newline='\n') as out:
                		writer = csv.writer(out,delimiter=',')
                		if (Flag==False): # just one time to write the header of the CSV file
                			if labels_exist:
                				header= numpy.concatenate([["Dataset", "Optimizer","objfname","k","ExecutionTime","SSE","Purity","Entropy","HS","CS","VM","AMI","ARI","Fmeasure","TWCV","SC","Accuracy","DI","DB","STDev"],CnvgHeader])
                			else:
                				header= numpy.concatenate([["Dataset", "Optimizer","objfname","k","ExecutionTime","SSE","TWCV","SC","DI","DB","STDev"],CnvgHeader])
                			writer.writerow(header)
                			Flag=True # at least one experiment

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
                		if labels_exist:
                			a=numpy.concatenate([[dataset_List[h], optimizerName,objfname,k[h], avgExecutionTime,avgSSE,avgPurity,avgEntropy,avgHomo, avgComp, avgVmeas, avgAMI, avgARI, avgFmeasure, avgTWCV, avgSC, avgAccuracy, avgDI, avgDB, avgStdev],avgConvergence])
                		else:
                			a=numpy.concatenate([[dataset_List[h], optimizerName,objfname,k[h], avgExecutionTime,avgSSE,avgTWCV, avgSC, avgDI, avgDB, avgStdev],avgConvergence])
                		writer.writerow(a)
                	out.close()
    	        

	if Export_convergence == True:
		conv_plot.run(results_directory, optimizer, objectivefunc, dataset_List, Iterations)
	
	if Export_boxplot == True:
		if labels_exist:
			ev_measures=['SSE','Purity','Entropy', 'HS', 'CS', 'VM', 'AMI', 'ARI', 'Fmeasure', 'TWCV', 'SC', 'Accuracy', 'DI', 'DB', 'STDev']
		else:
			ev_measures=['SSE','TWCV', 'SC', 'DI', 'DB', 'STDev']
		box_plot.run(results_directory, optimizer, objectivefunc, dataset_List, ev_measures, Iterations)

	print("Execution completed") 


def selector(algo,objective_name, k, f, popSize,Iter, points, metric):
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
		x=cssa.SSA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="PSO"):
		x=cpso.PSO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="GA"):
		x=cga.GA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="BAT"):
		x=cbat.BAT(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="FFA"):
		x=cffa.FFA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="GWO"):
		x=cgwo.GWO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="WOA"):
		x=cwoa.WOA(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="MVO"):
		x=cmvo.MVO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="MFO"):
		x=cmfo.MFO(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	if(algo=="CS"):
		x=ccs.CS(getattr(objectives, objective_name),lb,ub,k * f,popSize,Iter, k, points, metric)        
	return x
