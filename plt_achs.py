import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import argparse
parser = argparse.ArgumentParser(description='plot score and predictions')
parser.add_argument('-d', "--datadir", dest='datadir', metavar='<data-dir>', type=str, help='data directory (default KIDNEY)', default='KIDNEY', required=False)
parser.add_argument('-f', "--scorefile", dest='scorefile', metavar='<score-file>', type=str, help='score filename (default scorepred.csv)', default='scorepred.csv', required=False)
parser.add_argument('-u', "--upto", dest='upto', metavar='<upto>', type=int, help='number of genes displayed (default: None)' , default=None, required=False)
parser.add_argument('-g', "--genelist", dest='genelist', metavar='<genelist>', nargs="+", default=None, help='gene to display (default None, values any list)', required=False)
args = parser.parse_args()

targets = pd.read_csv(f'{args.datadir}/{args.scorefile}', index_col=0)
num_bins=81
interval=[-2.5, 2.5]
#interval=[y.min(numeric_only=True).min(), y.max(numeric_only=True).max()]
size = args.upto if args.upto is not None else len(targets)
if args.genelist is None:
    for i in range(size):
        ax = targets.filter(regex='score_').iloc[i].hist(bins=num_bins, range=interval, alpha=0.5, label='score')        
        ax = targets.filter(regex='pred_').iloc[i].hist(bins=num_bins, range=interval, alpha=0.5, label='preds')        
        plt.plot()
        plt.legend(loc='upper right')
        plt.title(f'{targets.index[i]}')
        plt.ylim(0, len(targets.filter(regex='score_').columns))
        plt.pause(0.20)
        plt.close()
else:
    for gene in args.genelist:
        ax = targets.filter(regex='score_').loc[gene].hist(bins=num_bins, range=interval, alpha=0.5, label='score')        
        ax = targets.filter(regex='pred_').loc[gene].hist(bins=num_bins, range=interval, alpha=0.5, label='preds')        
        plt.plot()
        plt.legend(loc='upper right')
        plt.title(f'{gene}')
        plt.ylim(0, len(targets.filter(regex='score_').columns))
        #plt.pause(0.50)
        plt.show()
plt.show()


#for i in range(50):
#   ax = y.iloc[i].hist(bins=num_bins, range=interval)        
#   plt.show()
