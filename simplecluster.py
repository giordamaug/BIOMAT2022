import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import *

x = pd.read_csv("KIDNEY/x_best_EvsAE.csv", index_col=0)
y = pd.read_csv("KIDNEY/y_EvsAE.csv", index_col=0)
y = y["label_wo_outliers"].values
X = x.to_numpy()
clst = KMeans(n_clusters=2).fit(X)
clustering_metrics = [homogeneity_score,
         completeness_score,
         v_measure_score,
         adjusted_rand_score,
         adjusted_mutual_info_score]
results = [m(y, clst.labels_) for m in clustering_metrics]
results += [silhouette_score(X,clst.labels_,metric="euclidean",sample_size=300,)]
formatter_result = ("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
print("homo\tcompl\tv-meas\tARI\tAMI\tsilhouette\n" + formatter_result.format(*results))