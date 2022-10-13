---
  title: BIOMAT sperimentazione
  author: Maurizio Giordano
  date: 10-07-2022
---

  
# KIDNEY


## Node2Vec

| embed | Accuracy    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| 64 | 0.931±0.008 | 0.819±0.020 | 0.658±0.039   | 0.979±0.005   | **0.709±0.036** | [[ 574  298]  |
|      |             |             |               |               |             |  [ 104 4881]] |
| 128 | 0.931±0.004 | 0.812±0.015 | 0.642±0.033   | 0.982±0.005   | 0.708±0.018 | [[ 560  312]  |
|      |             |             |               |               |             |  [  90 4895]] |

## GLEE

| embed | Accuracy    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| 64 | 0.922±0.010 | 0.805±0.023 | 0.639±0.044   | 0.971±0.007   | 0.670±0.045 | [[ 557  315]  |
|      |             |             |               |               |             |  [ 143 4842]] |
| 128 | 0.928±0.008 | 0.811±0.019 | 0.643±0.037   | 0.978±0.005   | 0.694±0.034 | [[ 561  311]  |
|      |             |             |               |               |             |  [ 111 4874]] |
| 256 | 0.928±0.006 | 0.810±0.014 | 0.643±0.026   | 0.977±0.004   | 0.692±0.028 | [[ 561  311]  |
|      |             |             |               |               |             |  [ 113 4872]] |
| 512 | 0.927±0.007 | 0.809±0.018 | 0.640±0.034   | 0.977±0.004   | 0.689±0.032 | [[ 558  314]  |
|      |             |             |               |               |             |  [ 114 4871]] |

# LUNG

## GLEE

| embed | Accuracy    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| 64 | 0.925±0.007 | 0.810±0.010 | 0.648±0.016   | 0.972±0.007   | 0.678±0.030 | [[ 559  304]  |
|      |             |             |               |               |             |  [ 141 4909]] |
| 128 | 0.928±0.005 | 0.815±0.014 | 0.655±0.030   | 0.975±0.006   | 0.692±0.024 | [[ 565  298]  |
|      |             |             |               |               |             |  [ 127 4923]] |
| 256 | 0.931±0.003 | 0.819±0.010 | 0.660±0.022   | 0.977±0.004   | 0.702±0.015 | [[ 570  293]  |
|      |             |             |               |               |             |  [ 117 4933]] |
| 512 | 0.934±0.002 | 0.825±0.011 | 0.671±0.022   | 0.979±0.002   | **0.718±0.012** | [[ 579  284]  |
|      |             |             |               |               |             |  [ 105 4945]] |

## Node2Vec 

| embed | Accuracy    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| 64 | 0.935±0.004 | 0.821±0.007 | 0.659±0.014   | 0.982±0.004   | **0.720±0.016** | [[ 569  294]  |
|      |             |             |               |               |             |  [  90 4960]] |
| 128 | 0.935±0.002 | 0.819±0.007 | 0.655±0.016   | 0.983±0.004   | 0.719±0.010 | [[ 565  298]  |
|      |             |             |               |               |             |  [  86 4964]] |