import time
import matplotlib.pyplot as plt
import numpy as np

#from sklearn.cluster import MeanShift
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler

from  meanshift.mean_shift_gpu  import  MeanShiftEuc

def main():
    # Generate a blob dataset.
    n_samples = 100000
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=9)

    # Normalize dataset for easier parameter selection
    X, y = blobs
    X = StandardScaler().fit_transform(X)

    # Estimate bandwidth for mean shift(Select 1000 points)
    bandwidth = cluster.estimate_bandwidth(X[0:999])
    bandwidth_gpu = 2*bandwidth/(X.max()-X.min())

    # Obtain results
    ms = MeanShiftEuc(bandwidth=bandwidth_gpu, cluster_all=True, GPU=True)
    ms.fit(X)
    labels  =  ms.labels_

    return 0

if __name__ == "__main__":
    # execute only if run as a script
    main()
