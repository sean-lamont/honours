#!/bin/sh

# the 10 arms example
#./runpca.py --visualize 20arms

# atlas2 benchmarking
#./runpca.py data/atlas2.csv --lrate 0.005 --batchsize 32 --epochs 300

# dietswap benchmarking
#./runpca.py data/dietswap.csv --lrate 0.005 --batchsize 32 --epochs 500

# make the figures
#./fig_benchmark_curves.py data/atlas2.csv_lrate0.005_batchsize32_epochs300_maxdim10_test0.1.npz data/dietswap.csv_lrate0.005_batchsize32_epochs500_maxdim10_test0.1.npz
