"""End-to-end: cNMF prepare -> factorize (cpu and gpu) -> combine -> consensus on small synthetic counts, and
the two solvers' consensus programs must agree. Also solver selection and the recorded parameters."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from cnmf_gpu import cNMFGPU, gpu_available, select_solver

K, N, G, N_ITER = 5, 2500, 80, 8


@pytest.fixture(scope="module")
def counts(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    H = rng.gamma(0.4, 1.0, (K, G))
    H /= H.sum(1, keepdims=True)
    W = rng.gamma(0.6, 1.0, (N, K)) * 60
    X = rng.poisson(W @ H + 0.05).astype(np.float32)
    X[X.sum(1) == 0, 0] = 1
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(N)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(G)]))
    a.write_h5ad(d / "counts.h5ad")
    (d / "genes.txt").write_text("\n".join(a.var.index))
    return d


def _run(counts, workdir, name, solver):
    obj = cNMFGPU(output_dir=str(workdir), name=name)
    obj.prepare(str(counts / "counts.h5ad"), components=[K], n_iter=N_ITER, seed=1,
                genes_file=str(counts / "genes.txt"), solver=solver)
    obj.factorize()
    obj.combine()
    obj.consensus(K, density_threshold=2.0, show_clustering=False, close_clustergram_fig=True)
    spec = pd.read_csv(Path(workdir) / name / f"{name}.spectra.k_{K}.dt_2_0.consensus.txt", sep="\t", index_col=0)
    return obj, spec


def _matched_cos(a, b):
    An = a / np.linalg.norm(a, axis=1, keepdims=True)
    Bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    S = An @ Bn.T
    r, c = linear_sum_assignment(-S)
    return S[r, c]


def test_cpu_path_is_plain_cnmf(counts, tmp_path):
    obj, spec = _run(counts, tmp_path, "cpu", "cpu")
    assert obj.solver_params()["solver"] == "cpu"
    assert spec.shape == (K, G)
    assert not Path(obj.paths["gpu_factorize_log"]).exists()


def test_gpu_matches_cpu_consensus(counts, tmp_path):
    ok, why = gpu_available()
    if not ok:
        pytest.skip(why)
    _, spec_cpu = _run(counts, tmp_path, "cpu", "cpu")
    obj, spec_gpu = _run(counts, tmp_path, "gpu", "gpu")
    p = obj.solver_params()
    assert p["solver"] == "gpu" and p["requested"] == "gpu" and "gpu" in p
    log = pd.read_csv(obj.paths["gpu_factorize_log"])
    assert len(log) == N_ITER and log["converged"].all()
    cos = _matched_cos(spec_cpu.values, spec_gpu.values)
    assert cos.min() > 0.95, cos


def test_auto_and_gpu_validation(counts, tmp_path):
    obj = cNMFGPU(output_dir=str(tmp_path), name="auto")
    obj.prepare(str(counts / "counts.h5ad"), components=[K], n_iter=2, seed=1,
                genes_file=str(counts / "genes.txt"), solver="auto")
    p = obj.solver_params()
    assert p["requested"] == "auto" and p["solver"] in ("cpu", "gpu") and p["reason"].startswith("auto")
    # gpu is refused for settings the GPU solver does not implement
    kw = dict(beta_loss="kullback-leibler", init="random", alpha_W=0.0, alpha_H=0.0)
    with pytest.raises((ValueError, RuntimeError)):
        select_solver("gpu", kw, 100, 10, 5, 4)
    assert select_solver("auto", kw, 100, 10, 5, 4)[0] == "cpu"
    assert select_solver("cpu", {}, 100, 10, 5, 4) == ("cpu", "requested")


def test_set_solver_after_prepare_and_skip_completed(counts, tmp_path):
    ok, why = gpu_available()
    if not ok:
        pytest.skip(why)
    obj = cNMFGPU(output_dir=str(tmp_path), name="later")
    obj.prepare(str(counts / "counts.h5ad"), components=[K], n_iter=4, seed=2, genes_file=str(counts / "genes.txt"))
    assert obj.solver_params()["solver"] == "cpu"
    obj.set_solver("gpu", gpu_batch_size=2)
    obj.factorize(worker_i=0, total_workers=2)          # half the jobs
    obj.update_nmf_iter_params()
    obj.factorize(skip_completed_runs=True)             # the rest
    obj.combine()
    assert len(pd.read_csv(obj.paths["gpu_factorize_log"])) == 4
