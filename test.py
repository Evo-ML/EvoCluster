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
import sys
import numpy
import warnings
import time
import csv
import plot_convergence as conv_plot
import plot_boxplot as box_plot
import cluster_detection as clus_det

warnings.simplefilter(action='ignore')

# Select data sets
#"aggregation","aniso","appendicitis","balance","banknote","blobs","Blood","circles","diagnosis_II","ecoli","flame","glass","heart","ionosphere","iris","iris2D","jain","liver","moons","mouse","pathbased","seeds","smiley","sonar","varied","vary-density","vertebral2","vertebral3","wdbc","wine"
dataset_List = ["iris","aggregation"]

datasets_directory = "datasets/" # the directory where the dataset is stored
dataset_len = len(dataset_List)
k = [-1] * dataset_len
f = [-1] * dataset_len
points= [0] * dataset_len
labelsTrue = [0] * dataset_len

#read all datasets
for h in range(dataset_len):
		
	dataset_filename = dataset_List[h] + '.csv'			
	# Read the dataset file and generate the points list and true values 
	rawData = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), datasets_directory + dataset_filename), 'rt')
	data = numpy.loadtxt(rawData, delimiter=",")
	
	nPoints, nValues = data.shape #Number of points and Number of values for each point

	f[h] = nValues - 1 #Dimension value
	points[h] = data[:,:-1].tolist() #list of points
	labelsTrue[h] = data[:,-1].tolist() #List of actual cluster of each points (last field)

	points[h] =preprocessing.normalize(points[h], norm='max', axis=0)
		
	k[h] = clus_det.ELBOW(points[h])#k: Number of clusters
	print("ELBOW for " + dataset_List[h] + ":" + str(k[h]))		
	k[h] = clus_det.SC(points[h])		
	print("SC for " + dataset_List[h] + ":" + str(k[h]))		
	k[h] = clus_det.DB(points[h])		
	print("DB for " + dataset_List[h] + ":" + str(k[h]))		
	k[h] = clus_det.CH(points[h])		
	print("CH for " + dataset_List[h] + ":" + str(k[h]))		
	k[h] = clus_det.GAP_STATISTICS(points[h])	
	print("GAP for " + dataset_List[h] + ":" + str(k[h]))
	k[h] = clus_det.min_clusters(points[h])	
	print("min_clusters for " + dataset_List[h] + ":" + str(k[h]))
	k[h] = clus_det.max_clusters(points[h])	
	print("max_clusters for " + dataset_List[h] + ":" + str(k[h]))
	k[h] = clus_det.median_clusters(points[h])	
	print("median_clusters for " + dataset_List[h] + ":" + str(k[h]))




