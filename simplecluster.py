import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import *
from sklearn.metrics import *
import numpy as np
from sklearn.datasets import load_iris

if False:
    x = pd.read_csv("KIDNEY/x_best_EvsNE.csv", index_col=0)
    y = pd.read_csv("KIDNEY/y_EvsNE.csv", index_col=0)
    y = y["label"].values
    X  = x.to_numpy()
else:
    X, y = load_iris(return_X_y=True)

print(y)
#clst = KMeans(random_state=1, n_clusters=len(np.unique(y))).fit(X)
clst = MeanShift().fit(X)
print(f'Looking for {len(np.unique(y))} clusters')
clustering_metrics = [homogeneity_score,
         completeness_score,
         v_measure_score,
         adjusted_rand_score,
         adjusted_mutual_info_score]
results = [m(y, clst.labels_) for m in clustering_metrics]
results += [silhouette_score(X,clst.labels_,metric="euclidean",sample_size=300,)]
formatter_result = ("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
print("homo\tcompl\tv-meas\tARI\tAMI\tsilhouette\n" + formatter_result.format(*results))

#predict the labels of clusters.
label = clst.predict(X)
 
#Getting unique labels
u_labels = np.unique(label)
 
#plotting the results:
for i in u_labels:
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
plt.legend()
plt.show()