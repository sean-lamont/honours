## file list

* CodaPCA.py              the implementation of dimensionality reduction techniques
* measure.py              the distance measurements
* runpca.py               the benchmark scripts
* fig_benchmark_curves.py generate the figure of training/testing curves
* fig_legend.py           generate the figure of legend
* run.sh                  contains example commands to run our codes

## dependencies

```
$ pip install -r requirements.txt
```

## to perform PCA benchmarking
```
$ ./runpca.py dataset.csv
```
this will generate a npz file containing PCA results

## to visualize the results
```
$ ./fig_benchmark_curves.py 1.npz 2.npz
```
this will generate a pdf figure corresponding to the two input npz files

## to run the arms example
```
$ ./runpca.py --visualize 20arms
```
this will generate a pdf figure
