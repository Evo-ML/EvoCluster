# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:50:48 2019

@author: Raneem
"""

from sklearn import preprocessing

from optimizer_run import run

# Select optimizers
CSSA= False
CPSO= True
CGA= True
CBAT= False
CFFA=False
CGWO=False
CWOA=False
CMVO=False
CMFO=False
CCS=False
optimizer=[CSSA, CPSO, CGA, CBAT, CFFA, CGWO, CWOA, CMVO, CMFO, CCS]

# Select objective function
SSE=True
TWCV=True
SC=False
DB=False
DI=False
objectivefunc=[SSE, TWCV, SC, DB, DI] 

# Select data sets
dataset_List = ["iris.csv","aggregation.csv"]
'''
dataset_List = ["aggregation.csv","aniso.csv","appendicitisNorm.csv", "balance.csv",
                "banknote.csv", "blobs.csv","Blood.csv","circles.csv","diagnosis_II.csv",
                "ecoli.csv","flame.csv","glass.csv","heart.csv","ionosphere.csv",
                "iris.csv","iris2D.csv","jain.csv","liver.csv","moons.csv",
                "mouse.csv","pathbased.csv","seeds.csv","smiley.csv","sonar.csv",
                "varied.csv","vary-density.csv","vertebral2.csv","vertebral3.csv",
                "wdbc.csv","wine.csv"]
'''

# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=3

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {'PopulationSize' : 50, 'Iterations' : 100}

#Export results?
export_flags = {'Export_avg':True,'Export_details':True}

run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags)