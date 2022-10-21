
  
# KIDNEY


Combining `BIO`logical attributes and `EMB`edding features learned by Node2Vec (with 64-size)

```python
$ python nodeclassification2.py -d <dataset> 
     -a BIO EMBED  
     -A node_attributes.csv -Z zscore
     -e Node2Vec -L -c <dataset>/embeddings -s 64
     -E CS0 -N CS6 CS7 CS8 CS9
     -l label_wo_outliers -x ND -F node_labels2.csv 
     -m LGBM -O
```

Using only  `BIO`logical attributes.

```python
$ python nodeclassification2.py -d <dataset> 
     -a BIO  
     -A node_attributes.csv -Z zscore
     -E CS0 -N CS6 CS7 CS8 CS9
     -l label_wo_outliers -x ND -F node_labels2.csv 
     -m LGBM -O
```

Using only `EMB`edding features learned by Node2Vec (with 64-size)

```python
$ python nodeclassification2.py -d <dataset> -Z        
     -a EMBED 
     -e Node2Vec -L -c <dataset>/embeddings -s 64 
     -E CS0 -N CS6 CS7 CS8 CS9
     -l label_wo_outliers -x ND -F node_labels2.csv
     -m LGBM -O
```

| Attr | Acc    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| EMB+BIO | 0.948±0.002 | 0.898±0.007 | 0.826±0.019   | 0.970±0.005   | 0.797±0.007 | [[ 727  153]  |
| |             |             |               |               |             |  [ 150 4832]] |
| BIO | 0.907±0.008 | 0.852±0.021 | 0.773±0.042   | 0.931±0.004   | 0.662±0.033 | [[ 680  200]  |
|      |             |             |               |               |             |  [ 343 4639]] |
| EMB | 0.932±0.002 | 0.858±0.010 | 0.752±0.024   | 0.964±0.005   | 0.730±0.009 | [[ 662  218]  |
|      |             |             |               |               |             |  [ 179 4803]] |

# LUNG

|      | Acc    | BA          | Sensitivity   | Specificity   | MCC         | CM            |
|------+-------------+-------------+---------------+---------------+-------------+---------------|
| EMB+BIO | 0.950±0.004 | 0.896±0.013 | 0.820±0.029   | 0.972±0.005   | 0.799±0.017 | [[ 708  155]  |
|      |             |             |               |               |             |  [ 140 4910]] |
| BIO | 0.911±0.014 | 0.862±0.016 | 0.791±0.028   | 0.932±0.015   | 0.675±0.041 | [[ 683  180]  |
|      |             |             |               |               |             |  [ 345 4705]] |
| EMB | 0.940±0.009 | 0.871±0.017 | 0.773±0.031   | 0.969±0.007   | 0.756±0.034 | [[ 667  196]  |
|      |             |             |               |               |             |  [ 158 4892]] |
