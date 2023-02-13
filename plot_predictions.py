import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

res= pd.read_csv("results_avg.csv", index_col=0)
scores = pd.read_csv("KIDNEY/node_targets.csv", index_col=0)
th0 = -3.54
th1 = -1.056
th2 = -0.508
th3 = 0.01

scores["mean"] = scores.mean(axis=1)

res = res.join(scores)

cl = ['cyan', 'orange', 'green', 'brown']
cmap = ListedColormap(cl)

gene_list = res.index.values
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("Label")
plt.xlim(th0, 0.5)
plt.ylim([0, len(gene_list)])
plt.axvline(x=th0, color='b', ls=':', lw=2)
plt.axvline(x=th1, color='b', ls=':', lw=2)
plt.axvline(x=th2, color='r', ls=':', lw=2)
plt.axvline(x=th3, color='g', ls=':', lw=2)
scatter = plt.scatter(res['mean'].values, np.arange(len(gene_list)), c=res['label'].values, cmap=cmap)
plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.xlim(-3.55, 0.5)
plt.ylim([0, len(gene_list)])
plt.axvline(x=-3.54, color='b', ls=':', lw=2)
plt.axvline(x=-1.056, color='b', ls=':', lw=2)
scatter = plt.scatter(res['mean'].values, np.arange(len(gene_list)), c=res['prediction'].values, cmap=cmap)
plt.show()