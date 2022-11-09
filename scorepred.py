import pandas as pd
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser(description='score and predictions')
parser.add_argument('-d', "--datadir", dest='datadir', metavar='<data-dir>', type=str, help='data directory (default KIDNEY)', default='KIDNEY', required=False)
parser.add_argument('-f', "--scorefile", dest='scorefile', metavar='<score-file>', type=str, help='score filename (default scorepred.csv)', default='scorepred.csv', required=False)
parser.add_argument('-x', "--xfile", dest='xfile', metavar='<x-file>', type=str, help='x filename (default x.csv)', default='x.csv', required=False)
parser.add_argument('-y', "--yfile", dest='yfile', metavar='<y-file>', type=str, help='y filename (default y.csv)', default='y.csv', required=False)
args = parser.parse_args()

x = pd.read_csv(f'{args.datadir}/{args.xfile}', index_col=0)
targets = pd.read_csv(f'{args.datadir}/{args.yfile}', index_col=0)

import lightgbm as  lgb
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from sklearn.metrics import *
seed = 1
random.seed(seed)
np.random.seed(seed)
nfolds = 5
kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
regressor_args = {'random_state' : seed}
X = x.to_numpy()
y = targets.to_numpy()
genelist = targets.index.values
columns = list(targets.columns)
result = pd.DataFrame()
for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), desc=f"{nfolds}-fold")):
  train_x, train_y, test_x, test_y, genes = X[train_idx], y[train_idx], X[test_idx], y[test_idx], genelist[test_idx]
  data = {'name' : list(genes)}
  for i in range(len(y[0])):
      prediction = lgb.LGBMRegressor(**regressor_args).fit(train_x, train_y[:,i]).predict(test_x)
      data[f'pred_{columns[i]}'] = prediction
      data[f'score_{columns[i]}'] = list(test_y[:,i])
  result = result.append(pd.DataFrame(data))

print(result)
result.set_index('name').to_csv(f'{args.datadir}/{args.scorefile}')