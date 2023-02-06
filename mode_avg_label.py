

import pandas as pd
import numpy as np

scores = pd.read_csv("KIDNEY/node_targets.csv", index_col=0)
th1 = -1.056
th2 = -0.508
th3 = 0.01

def relab(x):
    res = "E" if x < th1 else "aE" if x < th2 else "aNE" if x < th3 else "NE"
    return res

# calculate mode
scores.apply(np.vectorize(relab)).mode(axis=1)[0].to_csv("KIDNEY/node_labels_modescore.csv", index=True)

# calculate mean
scores["mean"] = scores.mean(axis=1)
scores.apply(lambda x: "E" if x["mean"] < th1 else "aE" if x["mean"] < th2 else "aNE" if x["mean"] < th3 else "NE", axis=1).to_csv("KIDNEY/node_labels_avgscore.csv", index=True)