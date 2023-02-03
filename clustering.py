"""
CLUSTERING
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m' 
    

import warnings
warnings.filterwarnings('ignore')
import random
import pandas as pd
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
import argparse
from operator import index
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from collections import Counter

from sklearn.cluster import *
from sklearn.mixture import *
from lightgbm import LGBMClassifier
from logitboost import LogitBoost
from tabulate import tabulate
from time import time

parser = argparse.ArgumentParser(description='BIOMAT 2022 Workbench')
parser.add_argument('-i', "--inputfile", dest='inputfile', metavar='<inputfile>', nargs="+", type=str, help='inputfile dataset', required=True)
parser.add_argument('-b', "--seed", dest='seed', metavar='<seed>', type=int, help='seed (default: 1)' , default=1, required=False)
parser.add_argument('-n', "--nclusters", dest='nclusters', metavar='<nclusters>', type=int, help='n. clusters (default: 2)' , default=2, required=False)
parser.add_argument('-O', "--tuneparams", dest='tuneparams',  action='store_true', required=False)
parser.add_argument('-f', "--labelfile", dest='labelfile', metavar='<labelfile>', type=str, help='label filename', required=True)
parser.add_argument('-m', "--method", dest='method', metavar='<method>', type=str, help='clustering name (default: KMeans, choice: KMrans|DBSCAN)', default='KMeans', required=False)
parser.add_argument('-V', "--verbose", action='store_true', required=False)
parser.add_argument('-D', "--tocsv", dest="tocsv", type=str, required=False)
args = parser.parse_args()

seed=args.seed

"""
Load attributes
"""

attr_file = args.inputfile[0]
print(bcolors.HEADER + f'Loading attribute matrix "{attr_file}" ... ' + bcolors.ENDC, end='')
x = pd.read_csv(attr_file, index_col=0)
x = x.select_dtypes(include=np.number)     # drop non numeric attributes
print(bcolors.HEADER + f'{x.shape}' + bcolors.ENDC)
for attr_file in args.inputfile[1:]:
  print(bcolors.HEADER + f'Loading attribute matrix "{attr_file}" ... ' + bcolors.ENDC, end='')
  df = pd.read_csv(attr_file, index_col=0)
  df = df.select_dtypes(include=np.number)     # drop non numeric attributes
  print(bcolors.HEADER + f'{df.shape}' + bcolors.ENDC)
  x = x.join(df)

"""
Load labels
"""

label_file = args.labelfile #@param {type:"string"}
df_label = pd.read_csv(label_file, sep=',', index_col=0)
labelname = "label_wo_outliers"
print(bcolors.HEADER + f'Loading label "{labelname}" from file "{label_file}"...' + bcolors.ENDC)
print(bcolors.OKGREEN + f'- label distribution: {dict(df_label[labelname].value_counts())}' + bcolors.ENDC)


### set balancing in method paramters
clustering_map = {'KMeans' : 'KMeans', 
                  'DBSCAN': 'DBSCAN',
                  'SC' : 'SpectralClustering',
                  'GM' : 'GaussianMixture',
                  "MS" : "MeanShift",
                  "AP" : "AffinityPropagation",
                  "AC" : "AgglomerativeClustering",
                  "MBKMeans" : "MiniBatchKMeans"
                } 

set_seed(seed)
if args.tuneparams:
  clustering_args = {
    'KMeans' : {"n_clusters": args.nclusters,"n_init": "auto", "random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'MBKMeans' : {"n_clusters": args.nclusters,"n_init": "auto", "random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'DBSCAN' : {"eps" : 0.3, "min_samples": 2},
    'SC' : {"n_clusters": args.nclusters, "random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'MS' :{},
    'AP' : {"random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'AC' : {"n_clusters": args.nclusters},
  }
else:
  clustering_args = {
    'KMeans' : {"n_clusters": args.nclusters,"random_state": seed, 'verbose' : 1 if args.verbose else 0}, 
    'MBKMeans' : {"n_clusters": args.nclusters,"random_state": seed, 'verbose' : 1 if args.verbose else 0}, 
    'DBSCAN' : {"eps" : 0.3, "min_samples": 2},
    'SC' : {"n_clusters": args.nclusters, "random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'MS' :{},
    'AP' : {"random_state": seed, 'verbose' : 1 if args.verbose else 0},
    'AC' : {},
  }


"""
Clustering
"""

method = args.method 
genes = x.index.values

X = x.to_numpy()
y = df_label[labelname].values
print(bcolors.HEADER + f'Clustering with method "{clustering_args[method]}"...' + bcolors.ENDC)
print(bcolors.OKGREEN + f'- method params: {clustering_map[method]}' + bcolors.ENDC)
t0 = time()
clst = globals()[clustering_map[args.method]](**clustering_args[args.method]).fit(X)
fit_time = time() - t0
results = [method, fit_time, getattr(clst, 'inertia_', 0)]
labels = clst.labels_
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#n_noise_ = list(labels).count(-1)
#print(bcolors.OKGREEN + f'Labels: {labels}' + bcolors.ENDC)
#print(bcolors.OKGREEN + f'n. clusters: {n_clusters_}' + bcolors.ENDC)
#print(bcolors.OKGREEN + f'Noise: {n_noise_}' + bcolors.ENDC)

from sklearn import metrics
clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
results += [m(y, clst.labels_) for m in clustering_metrics]
results += [
        metrics.silhouette_score(
            X,
            clst.labels_,
            metric="euclidean",
            sample_size=1000,
        )
    ]
print(82 * bcolors.OKGREEN + "_"+ bcolors.ENDC)
print(bcolors.OKGREEN + "init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette"+ bcolors.ENDC)
print(bcolors.OKGREEN + formatter_result.format(*results)+ bcolors.ENDC)
print(82 * bcolors.OKGREEN + "_"+ bcolors.ENDC)
