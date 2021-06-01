# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:06:19 2021

@author: Dang Trung Anh
"""

from EvoCluster import EvoCluster

optimizer = ["SSA", "PSO", "GA", "GWO"]
objective_func = ["SSE", "TWCV"]
dataset_list = ["iris", "aggregation"]
num_of_runs = 3
params = {'PopulationSize': 30, 'Iterations': 50}
export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                'Export_convergence': True, 'Export_boxplot': True}

ec = EvoCluster(
    optimizer,
    objective_func,
    dataset_list,
    num_of_runs,
    params,
    export_flags,
    auto_cluster=True,
    n_clusters='supervised',
    labels_exist=True,
    metric='euclidean'
)

ec.run()
