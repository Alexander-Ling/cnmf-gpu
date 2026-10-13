"""Parallel CPU post-processing: cNMF ``consensus`` and the ``k_selection_plot`` statistics in worker processes.

cNMF runs ``consensus`` for one K after another in a single process, and nothing in it uses the GPU. Every K is
independent, so this module distributes the work over a pool of worker processes:
  * tasks are single consensus passes: (K, density threshold) and, with ``k_selection``, (K, statistics). cNMF's per-K
    local-density cache is written by the parent beforehand (cNMF's own few lines, seconds per K), so all passes are
    independent and no file has two writers.
  * the normalized counts and the TPM matrix are read once by the parent and written as plain arrays to a shared-memory
    directory (``/dev/shm`` where it exists). Workers map them read-only, so all workers share one copy of each matrix
    and a worker costs only its working buffers. Read-only also means cNMF cannot modify them unnoticed: ``consensus``
    only reads both matrices (its one in-place scaling acts on a gene-subset copy), and a write would raise.
  * workers start with "spawn" (fresh interpreters: forking a process that has used OpenMP aborts). Memory grows with
    the number of workers (each holds transient copies inside cNMF/sklearn, up to ~1.5x the matrices), not with their
    threads, so the worker count is capped by memory and spare cores become BLAS/OpenMP threads per worker.
The work per pass is cNMF's own ``consensus`` code, so the files written are the same as cNMF's. Results can differ
from a multi-threaded serial run at rounding level only (floating-point summation order of the BLAS calls).

    from cnmf_gpu import cNMFGPU, run_consensus
    obj = cNMFGPU(output_dir="runs", name="s1")
    run_consensus(obj, ks=range(5, 41), density_thresholds=[0.1], workers=0, k_selection=True)

"spawn" re-imports the calling script in every worker: call ``run_consensus`` from a script only under
``if __name__ == "__main__":`` (the ``cnmf-gpu`` CLI and notebooks need nothing).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import h5py
import numpy as np
import pandas as pd

_W = {}     # worker state: the cNMF object, normalized counts, thread limiter


# ---------------------------------------------------------------------------------------------- shared matrices
def _export(adata, d):
    """Write an AnnData's X (dense or scipy sparse) and obs/var names as arrays that workers can memory-map."""
    import scipy.sparse as sp
    os.makedirs(d)
    X = adata.X
    if sp.issparse(X):
        fmt = X.format
        for part in ("data", "indices", "indptr"):
            np.save(os.path.join(d, part + ".npy"), getattr(X, part))
    else:
        fmt = "dense"
        np.save(os.path.join(d, "X.npy"), np.asarray(X))
    pd.Series({"format": fmt, "n_obs": X.shape[0], "n_vars": X.shape[1]}).to_pickle(os.path.join(d, "meta.pkl"))
    pd.Index(adata.obs_names).to_series().to_pickle(os.path.join(d, "obs.pkl"))
    pd.Index(adata.var_names).to_series().to_pickle(os.path.join(d, "var.pkl"))


def _attach(d):
    """AnnData whose X is backed by the read-only memory-mapped arrays of ``_export`` (no copy)."""
    import anndata as ad
    import scipy.sparse as sp
    meta = pd.read_pickle(os.path.join(d, "meta.pkl"))
    load = lambda part: np.load(os.path.join(d, part + ".npy"), mmap_mode="r")
    shape = (int(meta["n_obs"]), int(meta["n_vars"]))
    if meta["format"] == "dense":
        X = load("X")
    else:
        cls = {"csr": sp.csr_matrix, "csc": sp.csc_matrix}[meta["format"]]
        X = cls((load("data"), load("indices"), load("indptr")), shape=shape, copy=False)
    obs, var = pd.read_pickle(os.path.join(d, "obs.pkl")), pd.read_pickle(os.path.join(d, "var.pkl"))
    return ad.AnnData(X=X, obs=pd.DataFrame(index=obs.index), var=pd.DataFrame(index=var.index))


class _ReadCache:
    """Stands in for the ``scanpy`` module inside ``cnmf.cnmf``: ``read`` of the TPM file returns the shared matrix;
    every other attribute and path goes to scanpy."""

    def __init__(self, sc, tpm_path, tpm):
        self._sc, self._tpm_path, self._tpm = sc, os.path.abspath(tpm_path), tpm

    def read(self, path, *args, **kwargs):
        if os.path.abspath(str(path)) != self._tpm_path:
            return self._sc.read(path, *args, **kwargs)
        return self._tpm

    def __getattr__(self, name):
        return getattr(self._sc, name)


def _init_worker(output_dir, name, threads, shm_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    from threadpoolctl import threadpool_limits
    _W["limits"] = threadpool_limits(limits=threads)
    import cnmf.cnmf as cc
    import scanpy as sc
    from .core import cNMFGPU
    obj = cNMFGPU(output_dir=output_dir, name=name)
    _W["obj"], _W["norm_counts"] = obj, _attach(os.path.join(shm_dir, "norm_counts"))
    cc.sc = _ReadCache(sc, obj.paths["tpm"], _attach(os.path.join(shm_dir, "tpm")))


def _task(kind, k, dt, local_neighborhood_size, show_clustering, build_ref):
    obj, nc = _W["obj"], _W["norm_counts"]
    t0 = time.perf_counter()
    stats = None
    if kind == "consensus":
        obj.consensus(int(k), dt, local_neighborhood_size, show_clustering, build_ref, close_clustergram_fig=True,
                      norm_counts=nc)
    else:               # exactly what cNMF's k_selection_plot computes for this K
        stats = obj.consensus(int(k), skip_density_and_return_after_stats=True, show_clustering=False,
                              close_clustergram_fig=True, norm_counts=nc).stats
    return kind, int(k), dt, stats, time.perf_counter() - t0


def _write_density_cache(obj, k, local_neighborhood_size):
    """cNMF 1.7's local-density cache for K (the same lines as in ``cNMF.consensus``), unless it exists."""
    from cnmf.cnmf import load_df_from_npz, save_df_to_npz
    from sklearn.metrics.pairwise import euclidean_distances
    if os.path.isfile(obj.paths["local_density_cache"] % k):
        return
    merged_spectra = load_df_from_npz(obj.paths["merged_spectra"] % k)
    n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0] / k)
    l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra ** 2).sum(axis=1))).T
    topics_dist = euclidean_distances(l2_spectra.values)
    partitioning_order = np.argpartition(topics_dist, n_neighbors + 1)[:, :n_neighbors + 1]
    distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
    local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1) / (n_neighbors), columns=["local_density"],
                                 index=l2_spectra.index)
    save_df_to_npz(local_density, obj.paths["local_density_cache"] % k)


def _clean_stale(shm_root):
    """Remove shared-memory directories left by runs whose process no longer exists (e.g. killed with SIGKILL)."""
    root = shm_root or tempfile.gettempdir()
    for d in os.listdir(root) if os.path.isdir(root) else []:
        parts = d.split("_")
        if d.startswith("cnmf_gpu_") and len(parts) > 2 and parts[2].isdigit():
            try:
                os.kill(int(parts[2]), 0)
            except ProcessLookupError:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            except PermissionError:
                pass


def _raise_sigterm(signum, frame):
    raise SystemExit("SIGTERM")


# ---------------------------------------------------------------------------------------------- sizing
def _matrix_gb(path) -> float:
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group):                      # sparse: data + indices + indptr
            return (X["data"].size * (X["data"].dtype.itemsize + X["indices"].dtype.itemsize) + X["indptr"].size * 8) / 2**30
        return X.size * X.dtype.itemsize / 2**30


def estimate_memory_gb(obj) -> tuple[float, float]:
    """(GiB shared, GiB per worker): the two matrices once in shared memory; per worker the transient copies cNMF and
    sklearn make (gene subsets, a CSR copy of the transposed TPM in the spectra refit, dense regression blocks), measured
    at up to 1.5x the matrices, plus 0.5 GiB of interpreter and libraries."""
    m = _matrix_gb(obj.paths["normalized_counts"]) + _matrix_gb(obj.paths["tpm"])
    return m, 1.5 * m + 0.5


def _available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            return int(next(l for l in f if l.startswith("MemAvailable")).split()[1]) / 2**20
    except (OSError, StopIteration, ValueError):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        except (ValueError, OSError, AttributeError):
            return 8.0


def _cores() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def auto_workers(obj, n_tasks: int, threads_per_worker: int = 1, memory_fraction: float = 0.7, max_memory_gb=None) -> int:
    """Workers = min(tasks, usable cores / threads per worker, what fits in the memory budget), where the budget is
    memory_fraction x available memory, capped at max_memory_gb."""
    shared, per = estimate_memory_gb(obj)
    budget = memory_fraction * _available_gb()
    if max_memory_gb:
        budget = min(budget, float(max_memory_gb))
    by_mem = int((budget - shared) // per)
    return max(1, min(n_tasks, max(1, (_cores() - 1) // max(1, threads_per_worker)), by_mem))


def save_k_selection(obj, stats: pd.DataFrame, close_fig: bool = True):
    """Write the k-selection statistics and plot exactly as cNMF 1.7's ``k_selection_plot`` does."""
    import matplotlib.pyplot as plt
    from cnmf.cnmf import save_df_to_npz
    stats = stats.sort_values("k").reset_index(drop=True)
    save_df_to_npz(stats, obj.paths["k_selection_stats"])
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(stats.k, stats.silhouette, "o-", color="b")
    ax1.set_ylabel("Stability", color="b", fontsize=15)
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    ax2.plot(stats.k, stats.prediction_error, "o-", color="r")
    ax2.set_ylabel("Error", color="r", fontsize=15)
    for tl in ax2.get_yticklabels():
        tl.set_color("r")
    ax1.set_xlabel("Number of Components", fontsize=15)
    ax1.grid("on")
    plt.tight_layout()
    fig.savefig(obj.paths["k_selection_plot"], dpi=250)
    if close_fig:
        plt.close(fig)
    return stats


# ---------------------------------------------------------------------------------------------- driver
def run_consensus(obj, ks=None, density_thresholds=(0.5,), workers=0, threads_per_worker=0, k_selection=False,
                  local_neighborhood_size=0.30, show_clustering=True, build_ref=True, max_memory_gb=None, shm_root=None,
                  log=print):
    """cNMF ``consensus`` for every K in ``ks`` (default: all K of the run) at every density threshold, and optionally
    the ``k_selection_plot`` statistics, with ``workers`` processes (0 = ``auto_workers`` within ``max_memory_gb``)
    of ``threads_per_worker`` BLAS/OpenMP threads (0 = spare cores / workers). The matrices are shared through
    ``shm_root`` (default /dev/shm, else the temp directory). Returns the seconds per K, summed over its passes."""
    import signal
    import threading
    import scanpy as sc
    from cnmf.cnmf import load_df_from_npz
    if ks is None:
        ks = sorted(set(load_df_from_npz(obj.paths["nmf_replicate_parameters"]).n_components))
    ks = sorted((int(k) for k in ks), reverse=True)            # largest K first: they take longest
    density_thresholds = [float(d) for d in np.atleast_1d(density_thresholds)]
    # largest K first; for each K its consensus passes, then its statistics pass
    tasks = [t for k in ks for t in [("consensus", k, d) for d in density_thresholds] + ([("stats", k, None)] if k_selection else [])]
    if workers <= 0:
        workers = auto_workers(obj, len(tasks), max(1, threads_per_worker), max_memory_gb=max_memory_gb)
    if threads_per_worker <= 0:
        threads_per_worker = max(1, (_cores() - 1) // workers)
    shared, per = estimate_memory_gb(obj)
    log(f"[cnmf-gpu] consensus: {len(ks)} K x {len(density_thresholds)} density threshold(s)"
        f"{' + k-selection stats' if k_selection else ''} = {len(tasks)} passes; {workers} worker(s) x {threads_per_worker} "
        f"thread(s); memory ~{shared + workers * per:.0f} GiB ({shared:.1f} shared + {per:.1f} per worker)")
    t0, done, stats_rows = time.perf_counter(), [], []
    shm_root = shm_root or ("/dev/shm" if os.path.isdir("/dev/shm") else None)
    _clean_stale(shm_root)
    shm_dir = tempfile.mkdtemp(prefix=f"cnmf_gpu_{os.getpid()}_{obj.name}_", dir=shm_root)
    on_term = None
    if threading.current_thread() is threading.main_thread():   # SIGTERM -> SystemExit, so the cleanup below runs
        on_term = signal.signal(signal.SIGTERM, _raise_sigterm)
    try:
        if density_thresholds:
            for k in ks:
                _write_density_cache(obj, k, local_neighborhood_size)
        for key, path in (("norm_counts", obj.paths["normalized_counts"]), ("tpm", obj.paths["tpm"])):
            _export(sc.read(path), os.path.join(shm_dir, key))
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(obj.output_dir, obj.name, threads_per_worker, shm_dir)) as ex:
            pending = {ex.submit(_task, *t, local_neighborhood_size, show_clustering, build_ref) for t in tasks}
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    try:
                        kind, k, dt, stats, sec = f.result()
                    except BrokenProcessPool as e:
                        raise RuntimeError(f"a consensus worker died (most often out of memory: {workers} workers x "
                                           f"~{per:.1f} GiB); rerun with fewer workers or a lower max_memory_gb") from e
                    done.append(dict(k=k, kind=kind, density_threshold=dt, seconds=sec))
                    if stats is not None:
                        stats_rows.append(stats)
                    log(f"[cnmf-gpu]   K={k} {kind}{'' if dt is None else f' dt={dt}'} done ({sec:.0f} s)")
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
        if on_term is not None:
            signal.signal(signal.SIGTERM, on_term)
    if k_selection:
        save_k_selection(obj, pd.DataFrame(stats_rows))
    log(f"[cnmf-gpu] consensus finished in {time.perf_counter() - t0:.0f} s")
    return pd.DataFrame(done).groupby("k", as_index=False)["seconds"].sum().round(1)
