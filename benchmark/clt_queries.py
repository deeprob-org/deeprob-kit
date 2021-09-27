import gc
import time
import numpy as np

from utils import compute_mean_stddev_times, load_dataset

from spn.structure.leaves.cltree.CLTree import CLTree
from spn.structure.leaves.cltree.MLE import update_cltree_parameters_mle
from spn.structure.leaves.cltree.Inference import cltree_log_likelihood
from spn.structure.leaves.cltree.MPE import cltree_mpe
from spn.structure.leaves.cltree.Sampling import sample_cltree_node

import deeprob.spn.structure as spn


def deeprob_learn_binary_clt(data: np.ndarray) -> spn.BinaryCLT:
    n_features = data.shape[1]
    scope = list(range(n_features))
    domain = [[0, 1]] * n_features
    clt = spn.BinaryCLT(scope, root=0)
    clt.fit(data, domain, alpha=0.1, random_state=42)
    return clt


def spflow_learn_binary_clt(data: np.ndarray) -> CLTree:
    n_features = data.shape[1]
    scope = list(range(n_features))
    clt = CLTree(scope, data)
    update_cltree_parameters_mle(clt, data, alpha=0.1)
    return clt


if __name__ == '__main__':
    n_repetitions = 20
    data_train, evi_data, mar_data = load_dataset('ad', n_samples=5000, seed=42)
    evi_data = evi_data.astype(np.int64)
    n_samples = 1000

    # ---- DeeProb-kit Learn Binary CLT -------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        clt = deeprob_learn_binary_clt(data_train)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[FIT] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))

    # ---- DeeProb-kit EVI Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = clt.log_likelihood(evi_data)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[EVI] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit MAR Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = clt.log_likelihood(mar_data)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MAR] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit MPE Inference ----------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        mpe_data = clt.mpe(mar_data)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    lls = clt.log_likelihood(mpe_data)
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MPE] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- DeeProb-kit Sampling ---------------------------------------------------------------------------------------
    gc.collect()
    mis_data = np.full(shape=(n_samples, mar_data.shape[1]), fill_value=np.nan, dtype=np.float32)
    elapsed_times = list()
    for _ in range(n_repetitions):
        mis_data[:] = np.nan
        start_time = time.perf_counter()
        _ = clt.sample(mis_data)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[SMP] DeeProb-kit Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))

    print()

    # ---- SPFlow Learn Binary CLT ------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        clt = spflow_learn_binary_clt(data_train)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[FIT] SPFlow Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))

    # ---- SPFlow EVI Inference ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = cltree_log_likelihood(clt, evi_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[EVI] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- SPFlow MAR Inference ---------------------------------------------------------------------------------------
    gc.collect()
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        lls = cltree_log_likelihood(clt, mar_data, dtype=np.float32)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MAR] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    # ---- SPFlow MPE Inference ---------------------------------------------------------------------------------------
    gc.collect()
    logprobs = np.empty(len(mar_data), dtype=np.float32)
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        mar_data_copied = mar_data.copy()
        cltree_mpe(clt, mar_data_copied, logprobs=logprobs)
        end_time = time.perf_counter()
        mpe_data = mar_data_copied
        elapsed_times.append(1e3 * (end_time - start_time))
    mpe_data = mpe_data.astype(np.int64)
    lls = cltree_log_likelihood(clt, mpe_data, dtype=np.float32)
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[MPE] SPFlow Avg. Time: {:.2f}ms (+- {:.2f}) - LL: {:.6f}".format(
        mean_time, stddev_time, np.mean(lls)
    ))

    """
    # The implementation of Sampling in SPFlow have an infinite loop
    # ---- SPFlow Sampling --------------------------------------------------------------------------------------------
    gc.collect()
    rand_gen = np.random.RandomState(42)
    elapsed_times = list()
    for _ in range(n_repetitions):
        start_time = time.perf_counter()
        _ = sample_cltree_node(clt, n_samples=n_samples, data=mar_data, rand_gen=rand_gen)
        end_time = time.perf_counter()
        elapsed_times.append(1e3 * (end_time - start_time))
    mean_time, stddev_time = compute_mean_stddev_times(elapsed_times)
    print("[SMP] SPFlow Avg. Time: {:.2f}ms (+- {:.2f})".format(
        mean_time, stddev_time
    ))
    """

    print()
