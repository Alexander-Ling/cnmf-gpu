"""Parallel consensus (cnmf_gpu.parallel) must reproduce cNMF's serial consensus and k_selection_plot: same files,
same numbers up to floating-point summation order. CPU only (the factorize step uses cNMF's CPU solver)."""
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from cnmf.cnmf import load_df_from_npz

from cnmf_gpu import cNMFGPU
from cnmf_gpu.parallel import _ReadCache, _attach, _export, auto_workers, estimate_memory_gb, run_consensus

KS, N, G, N_ITER, DTS = [3, 4, 5], 1200, 60, 8, [0.5, 2.0]
METHODS = ["workers2", "workers5"]                                   # fewer and more workers than K values


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("par")
    rng = np.random.default_rng(0)
    H = rng.gamma(0.4, 1.0, (5, G)); H /= H.sum(1, keepdims=True)
    X = rng.poisson(rng.gamma(0.6, 1.0, (N, 5)) * 60 @ H + 0.05).astype(np.float32)
    X[X.sum(1) == 0, 0] = 1
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(N)]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(G)]))
    a.write_h5ad(d / "counts.h5ad"); (d / "genes.txt").write_text("\n".join(a.var.index))
    obj = cNMFGPU(output_dir=str(d / "serial"), name="r")
    obj.prepare(str(d / "counts.h5ad"), components=KS, n_iter=N_ITER, seed=1, genes_file=str(d / "genes.txt"), solver="cpu")
    obj.factorize(); obj.combine()
    for m in METHODS:                                                    # identical inputs for every consensus path
        shutil.copytree(d / "serial" / "r", d / m / "r")
    for dt in DTS:
        for k in KS:
            obj.consensus(k, dt, show_clustering=True, close_clustergram_fig=True)
    obj.k_selection_plot(close_fig=True)
    tables = {m: run_consensus(cNMFGPU(output_dir=str(d / m), name="r"), KS, DTS, workers=int(m[-1]), k_selection=True,
                               show_clustering=True) for m in METHODS}
    return d, tables


def _files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("method", METHODS)
def test_same_files(runs, method):
    d, _ = runs
    assert _files(d / "serial" / "r") == _files(d / method / "r")


@pytest.mark.parametrize("method", METHODS)
def test_same_numbers(runs, method):
    d, _ = runs
    s, p = d / "serial" / "r", d / method / "r"
    npz = sorted(f for f in _files(s) if f.endswith(".df.npz"))
    assert any("consensus" in f for f in npz) and any("gene_spectra_score" in f for f in npz)
    for f in npz:
        a, b = load_df_from_npz(str(s / f)), load_df_from_npz(str(p / f))
        assert a.index.equals(b.index) and a.columns.equals(b.columns), f
        np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-5, atol=1e-8, err_msg=f)


@pytest.mark.parametrize("method", METHODS)
def test_k_selection_stats(runs, method):
    d, tables = runs
    a = load_df_from_npz(str(d / "serial" / "r" / "r.k_selection_stats.df.npz"))
    b = load_df_from_npz(str(d / method / "r" / "r.k_selection_stats.df.npz"))
    assert list(b["k"].astype(int)) == KS
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-6)
    assert list(tables[method]["k"]) == KS


def test_consensus_on_read_only_shared_matrices(runs, tmp_path):
    """Workers run consensus on read-only memory-mapped matrices: any in-place write by cNMF would raise here, and the
    results must equal those from the matrices read from disk."""
    import cnmf.cnmf as cc
    import scanpy as sc
    d, _ = runs
    obj = cNMFGPU(output_dir=str(d / "serial"), name="r")
    _export(sc.read(obj.paths["normalized_counts"]), str(tmp_path / "nc")); _export(sc.read(obj.paths["tpm"]), str(tmp_path / "tpm"))
    nc, tpm = _attach(str(tmp_path / "nc")), _attach(str(tmp_path / "tpm"))
    assert not (tpm.X.data if sp.issparse(tpm.X) else tpm.X).flags.writeable
    shutil.copytree(d / "serial" / "r", tmp_path / "ro" / "r")
    ro = cNMFGPU(output_dir=str(tmp_path / "ro"), name="r")
    keep = cc.sc
    try:
        cc.sc = _ReadCache(sc, ro.paths["tpm"], tpm)
        ro.consensus(KS[-1], DTS[0], show_clustering=False, close_clustergram_fig=True, norm_counts=nc)
    finally:
        cc.sc = keep
    f = f"cnmf_tmp/r.gene_spectra_score.k_{KS[-1]}.dt_{str(DTS[0]).replace('.', '_')}.df.npz"
    np.testing.assert_allclose(load_df_from_npz(str(tmp_path / "ro" / "r" / f)).to_numpy(float),
                               load_df_from_npz(str(d / "serial" / "r" / f)).to_numpy(float), rtol=1e-6, atol=1e-10)


def test_worker_estimates(runs):
    d, _ = runs
    par = cNMFGPU(output_dir=str(d / "workers2"), name="r")
    shared, per = estimate_memory_gb(par)
    assert 0 < shared < 0.1 and 0.5 <= per < 1.0
    assert 1 <= auto_workers(par, n_tasks=3) <= 3
