# Faster-Mean-Shift-Euc
Faster Mean-shift algorithm with Euclidean Distance Metrics. The algorithm is based on GPU acceleration, which can achieve satisfactory speedup with optimized GPU memory consumption. Here is a brief introduction to how to run it. And the details of our algorithm, please refer to our [paper1](https://arxiv.org/abs/2112.13891), [paper2](https://doi.org/10.1016/j.media.2021.102048).

The Cosine Metrics version is provided in another repository [Faster-Mean-Shift](https://github.com/masqm/Faster-Mean-Shift).


##  Environment
Win10

VS2019

Anacoda 2020.11

The packages requirement please see [requirements.txt](https://github.com/masqm/Faster-Mean-Shift-Euc/blob/main/requirements.txt)

Please make sure to use a compatible pytorch-gpu version.

## Example
Using our algorithm is similar to calling the meanshift in sklearn. 
An example of how to run the algorithm is given in [main.py](https://github.com/masqm/Faster-Mean-Shift-Euc/blob/main/FMS-Euc-git/main.py)：

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
    
Here is a helpful Q&A for image segmentation and bandwidth setting: https://github.com/masqm/Faster-Mean-Shift-Euc/issues/1
    
If you encounter any problem or find a bug during using, you are very welcome to contact me by (Mengyang.Zhao.TH@dartmouth.edu). If you use this code for your research, please cite our [paper1](https://arxiv.org/abs/2112.13891), [paper2](https://doi.org/10.1016/j.media.2021.102048). Thanks!
