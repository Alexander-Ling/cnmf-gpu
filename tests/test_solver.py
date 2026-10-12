"""Solver-level tests: kernel equivalence, optimality of the stopping rule, batching, reproducibility."""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from cnmf_gpu.solver import TorchHALS, choose_batch, gpu_memory_needed

K_TRUE, N, G = 6, 1500, 60


def _planted(seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    H = rng.gamma(0.3, 1.0, (K_TRUE, G))
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    W = rng.gamma(0.5, 1.0, (N, K_TRUE))
    X = W @ H
    return np.clip(X + noise * X.mean() * rng.standard_normal(X.shape), 0, None).astype(np.float32), H


def _cos_match(A, B):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    S = An @ Bn.T
    r, c = linear_sum_assignment(-S)
    return S[r, c]


def _gpu():
    ok, why = TorchHALS.available()
    if not ok:
        pytest.skip(why)
    return TorchHALS()


def test_choose_batch_logic():
    n, g, k = 165_553, 366, 30
    assert gpu_memory_needed(n, g, k, 16) > 4 * n * g
    assert choose_batch(n, g, k, 16, free_bytes=12 * 2**30) == 16
    assert 0 < choose_batch(n, g, k, 16, free_bytes=1 * 2**30) < 16
    assert choose_batch(n, g, k, 16, free_bytes=200 * 2**20) == 0


def test_valid_factors_and_recovery():
    be = _gpu()
    X, H_true = _planted()
    fits = be.fit_batch(X, K_TRUE, [1, 2], max_iter=500, tol=1e-4)
    assert len(fits) == 2
    for f in fits:
        assert f.usage.shape == (N, K_TRUE) and f.spectra.shape == (K_TRUE, G)
        assert np.isfinite(f.usage).all() and np.isfinite(f.spectra).all()
        assert (f.usage >= 0).all() and (f.spectra >= 0).all()
        assert f.converged and 1 < f.n_iter < 500
        assert np.median(_cos_match(f.spectra, H_true)) > 0.9


def test_matches_sklearn_cd_error():
    from sklearn.decomposition import non_negative_factorization
    be = _gpu()
    X, _ = _planted(seed=1)
    W, H, _ = non_negative_factorization(X, n_components=K_TRUE, init="random", solver="cd", tol=1e-5,
                                         max_iter=1000, random_state=5)
    cpu_err = np.linalg.norm(X - W @ H)
    f = be.fit_batch(X, K_TRUE, [5], max_iter=1000, tol=1e-5)[0]
    gpu_err = np.linalg.norm(X - f.usage @ f.spectra)
    assert gpu_err < cpu_err * 1.10 + 1e-3


def test_cd_rule_reaches_kkt_tolerance():
    """The returned factors satisfy sklearn's projected-gradient criterion (recomputed in float64)."""
    be = _gpu()
    X, _ = _planted(seed=5)
    tol = 1e-4
    be.trace = []
    f = be.fit_batch(X, K_TRUE, [9], max_iter=2000, tol=tol)[0]
    ratios = np.array([v[0] for _, v in be.trace])
    assert ratios[-1] <= tol and (ratios[:-1] > tol).all()      # stopped at the first iteration below tol
    import torch

    def viol64(W, H):
        X64 = X.astype(np.float64)
        gW = W @ (H @ H.T) - X64 @ H.T
        gH = (W.T @ W) @ H - W.T @ X64
        return np.abs(np.where(W == 0, np.minimum(gW, 0), gW)).sum() + np.abs(np.where(H == 0, np.minimum(gH, 0), gH)).sum()

    W0, H0 = TorchHALS._init(float(X.mean()), N, G, K_TRUE, [9], "cuda", torch.float32)
    v0 = viol64(W0[0].cpu().numpy().astype(np.float64), H0[0].cpu().numpy().astype(np.float64))
    v1 = viol64(f.usage.astype(np.float64), f.spectra.astype(np.float64))
    assert v1 / v0 <= 5 * tol        # post-hoc float64 check of the same optimality statistic


@pytest.mark.parametrize("k", [7, 33, 70])     # KP = 8, 64, 128: the >32 cases exercise the reduced row block
def test_fused_w_sweep_matches_reference(k):
    _gpu()
    import torch
    from cnmf_gpu.kernels import HAVE_TRITON, w_sweep_fused, w_sweep_reference
    if not HAVE_TRITON:
        pytest.skip("no triton")
    torch.manual_seed(0)
    B, n, g = 3, 1037, 80
    X = torch.rand(n, g, device="cuda")
    H = torch.rand(B, k, g, device="cuda")
    W = torch.rand(B, n, k, device="cuda")
    W[W < 0.3] = 0
    active = torch.tensor([True, False, True], device="cuda")
    XHt = X @ H.permute(2, 0, 1).reshape(g, B * k)
    HHt = torch.matmul(H, H.transpose(1, 2)).contiguous()
    W_ref, W_fused = W.clone(), W.clone()
    v_ref = w_sweep_reference(W_ref, XHt, HHt, active, 1e-7)
    v_fused = w_sweep_fused(W_fused, XHt, HHt, active, 1e-7)
    assert torch.equal(W_fused[1], W[1])
    assert torch.allclose(W_fused, W_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(v_fused[active], v_ref[active], rtol=1e-4)
    assert float(v_fused[1]) == 0.0


def test_fused_and_reference_solver_agree():
    _gpu()
    from cnmf_gpu.kernels import HAVE_TRITON
    if not HAVE_TRITON:
        pytest.skip("no triton")
    X, _ = _planted(seed=8)
    a = TorchHALS(fused=True).fit_batch(X, K_TRUE, [21, 22], max_iter=500, tol=1e-4)
    b = TorchHALS(fused=False).fit_batch(X, K_TRUE, [21, 22], max_iter=500, tol=1e-4)
    for fa, fb in zip(a, b):
        assert abs(fa.n_iter - fb.n_iter) <= 2
        assert np.min(_cos_match(fa.spectra, fb.spectra)) > 0.9999


def test_batch_matches_single():
    be = _gpu()
    X, _ = _planted(seed=2)
    seeds = [11, 12, 13]
    batched = be.fit_batch(X, K_TRUE, seeds, max_iter=300, tol=1e-4, batch_size=3)
    for s, b in zip(seeds, batched):
        single = be.fit_batch(X, K_TRUE, [s], max_iter=300, tol=1e-4, batch_size=1)[0]
        assert np.min(_cos_match(b.spectra, single.spectra)) > 0.999
        assert abs(b.n_iter - single.n_iter) <= 2


def test_same_seed_reproducible():
    be = _gpu()
    X, _ = _planted(seed=3)
    a = be.fit_batch(X, K_TRUE, [7], max_iter=200)[0]
    b = be.fit_batch(X, K_TRUE, [7], max_iter=200)[0]
    assert np.allclose(a.spectra, b.spectra, rtol=1e-3, atol=1e-4) and a.n_iter == b.n_iter


def test_sparse_input_accepted():
    be = _gpu()
    X, _ = _planted(seed=4)
    X[X < np.quantile(X, 0.6)] = 0
    f = be.fit_batch(sp.csr_matrix(X.astype(np.float64)), K_TRUE, [1], max_iter=200)[0]
    assert np.isfinite(f.spectra).all()
