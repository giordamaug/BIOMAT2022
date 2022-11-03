import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
x = pd.read_csv("KIDNEY/x.csv", index_col=0)
y = pd.read_csv("KIDNEY/y.csv", index_col=0)
num_bins=10
interval=[-3.5, 3.5]
#interval=[y.min(numeric_only=True).min(), y.max(numeric_only=True).max()]

# Create subplots

for i in range(len(y)):
    ax = y.iloc[i].hist(bins=num_bins, range=interval)        
    plt.plot()
    plt.title(f'{y.index[i]}')
    plt.ylim(0, 32)
    plt.pause(0.20)
    plt.close()

plt.show()


#for i in range(50):
#   ax = y.iloc[i].hist(bins=num_bins, range=interval)        
#   plt.show()
