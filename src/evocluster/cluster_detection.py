'''
ref: https://jtemporal.com/kmeans-and-elbow-method/
ref: https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
ref: https://github.com/minddrummer/gap/edit/master/gap/gap.py
ref: https://www.tandfonline.com/doi/pdf/10.1080/03610927408827101
ref: https://www.sciencedirect.com/science/article/pii/S0952197618300629?casa_token=W6QEUM7YA2cAAAAA:jtbAvDYF8axr8ghhr92aCnhXJ71wtCJ1tEZFHAjBUBbLrbJ8wdLHG0d4HIhDN5ICZJmrEGQ71vQ
ref: https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb
'''

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from math import sqrt
from sklearn import metrics 
import numpy as np
import random
import pandas as pd
import scipy
import sys

def min_clusters(points):
	k_list = all_methods(points)
	return min(k_list)

def max_clusters(points):
	k_list = all_methods(points)
	return max(k_list)

def median_clusters(points):
	k_list = all_methods(points)
	return int(np.median(np.array(k_list)))

def majority_clusters(points):
    k_list = all_methods(points)
    return max(set(k_list), key = k_list.count) 

def all_methods(points):
    k_list = [0] * 6
    k_list[0] = ELBOW(points)
    k_list[1] = GAP_STATISTICS(points)
    k_list[2] = SC(points)
    k_list[3] = CH(points)
    k_list[4] = DB(points)
    k_list[5] = BIC(points)
    return k_list


### Calinski-Harabasz Index
def CH(data):
	ch_max = 0
	ch_max_clusters = 2
	for n_clusters in range(2,10):
		model = KMeans(n_clusters = n_clusters)
		labels = model.fit_predict(data)
		ch_score = metrics.calinski_harabaz_score(data, labels)
		if ch_score > ch_max:
			ch_max = ch_score
			ch_max_clusters = n_clusters
	return ch_max_clusters
### END Calinski-Harabasz Index


### silhouette score 
def SC(data):
	sil_max = 0
	sil_max_clusters = 2
	for n_clusters in range(2,10):
		model = KMeans(n_clusters = n_clusters)
		labels = model.fit_predict(data)
		sil_score = metrics.silhouette_score(data, labels)
		if sil_score > sil_max:
			sil_max = sil_score
			sil_max_clusters = n_clusters
	return sil_max_clusters
### END silhouette score 

### DB score 
def DB(data):
    db_min = sys.float_info.max
    db_min_clusters = 2
    for n_clusters in range(2,10):
        model = KMeans(n_clusters = n_clusters)
        labels = model.fit_predict(data)
        db_score = metrics.davies_bouldin_score(data, labels)
        if db_score < db_min:
            db_min = db_score
            db_min_clusters = n_clusters
    return db_min_clusters
### END DB score 

### Bayesian Information Criterion
def BIC(data):
    bic_max = 0
    bic_max_clusters = 2
    for n_clusters in range(2,10):
        gm = GaussianMixture(n_components=n_clusters,n_init=10,tol=1e-3,max_iter=1000).fit(data)
        bic_score = -gm.bic(data)
        if bic_score > bic_max:
            bic_max = bic_score
            bic_max_clusters = n_clusters
    return bic_max_clusters
### END Bayesian Information Criterion


### ELBOW
def ELBOW(data):
	wcss = calculate_wcss(data)
	n_clusters = optimal_number_of_clusters(wcss)
	return n_clusters

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

def calculate_wcss(data):
    wcss = []
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)    
    return wcss
### END ELBOW

### gap statistics
def GAP_STATISTICS(data):
	gaps, s_k, K = gap_statistic(data, refs=None, B=10, K=range(1,11), N_init = 10) 
	bestKValue = find_optimal_k(gaps, s_k, K)	
	
	return bestKValue

def short_pair_wise_D(each_cluster):
    '''
    this function computes the sum of the pairwise distance(repeatedly) of all points in one cluster;
    each pair be counting twice here; using the short formula below instead of the original meaning of pairwise distance of all points

    each_cluster: np.array, with containing all points' info within the array
    '''
    mu = each_cluster.mean(axis = 0)
    total = sum(sum((each_cluster - mu)**2)) * 2.0 * each_cluster.shape[0]
    return total

def compute_Wk(data, classfication_result):
    '''
    this function computes the Wk after each clustering

    data:np.array, containing all the data
    classfication_result: np.array, containing all the clustering results for all the data
    '''
    Wk = 0
    label_set = set(classfication_result.tolist())
    for label in label_set:
        each_cluster = data[classfication_result == label, :]
        D = short_pair_wise_D(each_cluster)
        Wk = Wk + D/(2.0*each_cluster.shape[0])
    return Wk
 
def gap_statistic(X, refs=None, B=10, K=range(2,11), N_init = 10):
    '''
    this function first generates B reference samples; for each sample, the sample size is the same as the original datasets;
    the value for each reference sample follows a uniform distribution for the range of each feature of the original datasets;
    using a simplify formula to compute the D of each cluster, and then the Wk; K should be a increment list, 1-10 is fair enough;
    the B value is about the number of replicated samples to run gap-statistics, it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    
    X: np.array, the original data;
    refs: np.array or None, it is the replicated data that you want to compare with if there exists one; if no existing replicated/proper data, just use None, and the function
    will automatically generates them; 
    B: int, the number of replicated samples to run gap-statistics; it is recommended as 10, and it should not be changed/decreased that to a smaller value;
    K: list type, the range of K values to test on;
    N_init: int, states the number of initial starting points for each K-mean running under sklearn, in order to get stable clustering result each time; 
    you may not need such many starting points, so it can be reduced to a smaller number to fasten the computation;
    n_jobs below is not an argument for this function,but it clarifies the parallel computing, could fasten the computation, this can be only changed inside the script, not as an argument of the function;
    '''
    shape = X.shape
    if refs==None:
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        rands = scipy.random.random_sample(size=(shape[0],shape[1],B))
        for i in range(B):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs

    gaps = np.zeros(len(K))
    Wks = np.zeros(len(K))
    Wkbs = np.zeros((len(K),B))

    for indk, k in enumerate(K):
        # #setup the kmean clustering instance
        #n_jobs set up for parallel:1 mean No Para-computing; -1 mean all parallel computing
        #n_init is the number of times each Kmeans running to get stable clustering results under each K value
        k_means =  KMeans(n_clusters=k, init='k-means++', n_init=N_init, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        k_means.fit(X)
        classfication_result = k_means.labels_
        #compute the Wk for the classification result
        Wks[indk] = compute_Wk(X,classfication_result)
        
        # clustering on B reference datasets for each 'k' 
        for i in range(B):
            Xb = rands[:,:,i]
            k_means.fit(Xb)
            classfication_result_b = k_means.labels_
            Wkbs[indk,i] = compute_Wk(Xb,classfication_result_b)

    #compute gaps and sk
    gaps = (np.log(Wkbs)).mean(axis = 1) - np.log(Wks)        
    sd_ks = np.std(np.log(Wkbs), axis=1)
    sk = sd_ks*np.sqrt(1+1.0/B)
    return gaps, sk, K

def find_optimal_k(gaps, s_k, K):
    '''
    this function is finding the best K value given the computed results of gap-statistics

    gaps: np.array, containing all the gap-statistics results;
    s_k: float, the baseline value to minus with; say reference paper for detailed meaning;
    K: list, containing all the tested K values;
    '''
    gaps_thres = gaps - s_k
    below_or_above = (gaps[0:-1] >= gaps_thres[1:])
    if below_or_above.any():
        optimal_k = K[below_or_above.argmax()]
    else:
        optimal_k = K[-1]
    return optimal_k

### END gap statistics
