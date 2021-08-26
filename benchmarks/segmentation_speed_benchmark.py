#!/usr/bin/env python3

import sys
import os
import timeit
from multiprocessing import Process, Queue, set_start_method

import numpy as np
import scipy
import pandas as pd
import scanpy as sc
import anndata as ad

def read_data():
    data = sc.read_visium("mouse_brain_visium_wo_cloupe_data/rawdata/ST8059048")
    data.var_names_make_unique()
    data.var["mt"] = data.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(data, min_counts=4000)
    data = data[data.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(data, min_cells=100)

    coords = data.obsm["spatial"].max(axis=0)
    data1 = data.copy()
    data1.obsm["spatial"][:, 0] += coords[0]
    data2 = data.copy()
    data2.obsm["spatial"][:, 1] += coords[1]
    data3 = data.copy()
    data3.obsm["spatial"] += coords[np.newaxis, :]

    data = data.concatenate(data1, data2, data3)
    coords = data.obsm["spatial"].max(axis=0)
    data1 = data.copy()
    data1.obsm["spatial"][:, 0] += coords[0]

    return data.concatenate(data1)

def bench_cpu(ordering, npoints, vargenes, ncores, number, q):
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(ncores)
    tf.config.threading.set_intra_op_parallelism_threads(ncores)

    aff = os.sched_getaffinity(0)
    if ncores < len(aff):
        os.sched_setaffinity(0, [a for i, a in enumerate(aff) if i < ncores])


    import numpy as np
    import scanpy as sc
    import SpatialDE

    data = read_data()[ordering, :]

    results = []
    for n in npoints:
        cdata = data[:n, :]
        def leiden():
            data_normalized = cdata.copy()
            sc.pp.normalize_total(data_normalized, target_sum=1e4, key_added="scaling_factor")
            sc.pp.log1p(data_normalized)
            data_normalized = data_normalized[:, vargenes]
            sc.pp.regress_out(data_normalized, ['total_counts'])
            sc.pp.scale(data_normalized, max_value=10)
            n_comps = min(100, data_normalized.n_obs - 1)
            sc.tl.pca(data_normalized, n_comps=n_comps)
            sc.pp.neighbors(data_normalized, n_neighbors=20, n_pcs=n_comps)
            sc.tl.leiden(data_normalized, random_state=42)

        def spde():
            with tf.device("/device:CPU:0"):
                SpatialDE.tissue_segmentation(cdata, genes=vargenes, rng=np.random.default_rng(seed=42))

        leiden_res = timeit.repeat(leiden, repeat=1, number=number)[0]

        spatialde_cpu_res = timeit.repeat(spde, repeat=1, number=number)[0]

        df = pd.DataFrame({"method": ["leiden", "spatialde2"], "time_seconds": [leiden_res, spatialde_cpu_res]})
        df["ncores"] = str(ncores)
        df["npoints"] = n
        df.time_seconds = df.time_seconds / number
        results.append(df)
    q.put(pd.concat(results, axis=0, ignore_index=True))

if __name__ == "__main__":
    set_start_method("spawn")
    import tensorflow as tf
    import SpatialDE

    svg_full = pd.read_pickle("ST8059048_svg_full.pkl")
    vargenes = svg_full[svg_full.padj < 0.001].sort_values("total_counts", ascending=False).gene[:2000]

    data = read_data()
    centroid = np.mean(data.obsm["spatial"], axis=0)
    radii = np.sqrt(np.sum((data.obsm["spatial"] - centroid) ** 2, axis=1))
    ordering = np.argsort(radii)
    npoints = (200, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000)

    data = data[ordering, :]
    results = []
    number=10

    q = Queue()
    for ncores in (1, 10):
        p = Process(target=bench_cpu, args=(ordering, npoints, vargenes, ncores, number, q))
        p.start()
        df = q.get()
        p.join()
        results.append(df)

    for n in npoints:
        cdata = data[:n, :]
        def spde():
            with tf.device("/device:GPU:0"):
                SpatialDE.tissue_segmentation(cdata, genes=vargenes, rng=np.random.default_rng(seed=42))

        spatialde_gpu_res = timeit.repeat(spde, setup="import tensorflow as tf", repeat=1, number=number)[0]
        results.append(pd.DataFrame({"method": ["spatialde2"],
                                    "time_seconds": spatialde_gpu_res / number,
                                    "ncores": "gpu",
                                    "npoints": n}))

    results = pd.concat(results, axis=0, ignore_index=True)
    results.to_pickle("clustering_speed_benchmark.pkl")
    results.to_feather("clustering_speed_benchmark.feather")
