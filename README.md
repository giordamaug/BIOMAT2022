# BIOMAT2022
This repo is the experimental workbench for paper

```
Network-Based Computational Modeling to Unravel
Gene Essentiality. I. GRANATA et al.
To appear in BIOMAT2022
```

# How to reproduce experiments

Just open the `BIOMAT2022_workbench.ipynb` notebook and click on the Google Colab launch button.

# Results

## PPI network

#### BIO+GTEX

|------+------------+----------+---------------+---------------+----------+---------------|
|      |  Accuracy  |       BA |   Sensitivity |   Specificity |      MCC | CM            |
|------+------------+----------+---------------+---------------+----------+---------------|
| LGBM |   0.904825 | 0.819903 |      0.680537 |      0.959269 | 0.68234  | [[ 507  238]  |
|      |            |          |               |               |          |  [ 125 2944]] |
| XGB  |   0.895646 | 0.806577 |      0.660403 |      0.952751 | 0.651845 | [[ 492  253]  |
|      |            |          |               |               |          |  [ 145 2924]] |
| MLP  |   0.855275 | 0.690011 |      0.418792 |      0.961229 | 0.475601 | [[ 312  433]  |
|      |            |          |               |               |          |  [ 119 2950]] |
| RF   |   0.879919 | 0.727178 |       0.47651 |      0.977846 | 0.574159 | [[ 355  390]  |
|      |            |          |               |               |          |  [  68 3001]] |
|------+------------+----------+---------------+---------------+----------+---------------|

#### BIO+GTEX+Node2Vec

|     |   Accuracy |       BA |   Sensitivity |   Specificity |      MCC | CM            |
|-----+------------+----------+---------------+---------------+----------+---------------|
| LGBM |   0.927635 | 0.849832 |      0.722148 |      0.977516 | **0.758463** | [[ 538  207]  |
|      |            |          |               |               |          |  [  69 3000]] |
| XGB |   0.922654 | 0.842672 |      0.711409 |      0.973935 | 0.741349 | [[ 530  215]  |
|     |            |          |               |               |          |  [  80 2989]] |
| MLP |   0.918984 | **0.856145** |       0.75302 |       0.95927 | 0.735039 | [[ 561  184]  |
|     |            |          |               |               |          |  [ 125 2944]] |
| RF |   0.899582 | 0.762774 |      0.538255 |      0.987293 | 0.652522 | [[ 401  344]  |
|    |            |          |               |               |          |  [  39 3030]] |

#### BIO+GTEX+DeepWalk

|     |   Accuracy |       BA |   Sensitivity |   Specificity |      MCC | CM            |
|-----+------------+----------+---------------+---------------+----------+---------------|
| LGBM |    0.92554 | 0.83989 |      0.699329 |      0.980452 | **0.749946** | [[ 521  224]  |
|      |            |         |               |               |          |  [  60 3009]] |
| XGB |   0.923177 | 0.839947 |      0.703356 |      0.976538 | 0.742353 | [[ 524  221]  |
|     |            |          |               |               |          |  [  72 2997]] |
| MLP |   0.917932 | **0.851427** |      0.742282 |      0.960571 | 0.730933 | [[ 553  192]  |
|     |            |          |               |               |          |  [ 121 2948]] |
| RF |   0.896962 | 0.755046 |      0.522148 |      0.987944 | 0.641838 | [[ 389  356]  |
|    |            |          |               |               |          |  [  37 3032]] |

#### BIO+GTEX+HOPE

|     |   Accuracy |      BA |   Sensitivity |   Specificity |     MCC | CM            |
|-----+------------+---------+---------------+---------------+---------+---------------|
| LGBM |   0.905873 | 0.819539 |      0.677852 |      0.961227 | 0.685103 | [[ 505  240]  |
|      |            |          |               |               |          |  [ 119 2950]] |
| XGB |   0.904826 | 0.81838 |       0.67651 |       0.96025 | 0.68186 | [[ 504  241]  |
|     |            |         |               |               |         |  [ 122 2947]] |
| MLP |   0.850291 | 0.740279 |      0.559732 |      0.920827 | 0.504984 | [[ 417  328]  |
|     |            |          |               |               |          |  [ 243 2826]] |
| RF |   0.869166 | 0.721006 |      0.477852 |       0.96416 | 0.535904 | [[ 356  389]  |
|    |            |          |               |               |          |  [ 110 2959]] |
