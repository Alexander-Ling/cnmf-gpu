"""Batched GPU HALS for Frobenius NMF, matching cNMF's default CPU solver.

cNMF with ``beta_loss='frobenius'`` runs sklearn ``non_negative_factorization(solver='cd', init='random',
tol=1e-4, max_iter=1000)`` once per replicate. ``TorchHALS`` runs the same algorithm for many replicates at once:

* update rule: Cichocki-Phan HALS, i.e. sklearn's ``_update_cdnmf_fast``; usage (W) sweep first, then spectra (H),
  with the gradient for component j using the components already updated in the sweep;
* stopping rule: sum of |projected gradient| accumulated over both sweeps, relative to its value after
  iteration 1, must fall to ``tol``; checked every iteration; converged replicates are frozen;
* init: sklearn's 'random' scheme, sqrt(mean(X)/k) * |N(0,1)|, but drawn from torch's generator on the device,
  so a seed does not reproduce the CPU replicate (same distribution, different stream);
* arithmetic: float32 on a dense copy of X. The two products that reduce over cells are 2D GEMMs (batched
  matmul accumulates those sums at ~2e-5 relative error, enough to stall the stopping rule).

Validated against cNMF's CPU solver at consensus level on Xenium data (see the package README).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Fit:
    usage: np.ndarray      # cells x k
    spectra: np.ndarray    # k x genes
    n_iter: int
    seed: int
    converged: bool
    extra: dict = field(default_factory=dict)


def gpu_memory_needed(n_cells: int, n_genes: int, k: int, batch: int) -> int:
    """Rough device bytes for one ``fit_batch`` call: dense float32 X plus four (batch, n, k) float32 tensors
    (W, X H^T, the (n, B*k) copy of W for the GEMMs, and a work copy in the non-Triton path)."""
    return 4 * n_cells * n_genes + 16 * n_cells * k * batch + 256 * 2**20


def free_device_memory(device="cuda") -> int:
    """Free bytes on ``device`` ('cuda', 'cuda:1', torch.device or int)."""
    import torch
    dev = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    free, _total = torch.cuda.mem_get_info(idx)
    return int(free)


def choose_batch(n_cells: int, n_genes: int, k: int, requested: int, free_bytes: int, headroom: float = 0.85) -> int:
    """Largest batch <= requested that fits in ``headroom`` of ``free_bytes``; 0 if even one replicate does not."""
    budget = headroom * free_bytes - 4 * n_cells * n_genes - 256 * 2**20
    if budget <= 0:
        return 0
    return int(max(0, min(requested, budget // (16 * n_cells * k))))


class TorchHALS:
    """Batched HALS on a CUDA device. ``fit_batch(X, k, seeds, max_iter, tol)`` -> list[Fit]."""

    def __init__(self, batch_size: int = 16, device: str = "cuda", fused: bool | None = None):
        self.batch_size = int(batch_size)
        self.device = device
        if fused is None:
            from .kernels import HAVE_TRITON
            fused = HAVE_TRITON
        self.fused = bool(fused)
        self.trace = None  # set to a list to collect (iteration, per-replicate violation ratio)

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import torch
        except Exception as e:
            return False, f"torch import failed: {e}"
        if not torch.cuda.is_available():
            return False, "torch has no CUDA device"
        return True, ""

    def describe(self) -> dict:
        import torch
        info = {"backend": "torch_hals", "fused_kernel": bool(self.fused), "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda), "device": str(self.device)}
        try:
            info["gpu"] = str(torch.cuda.get_device_name(self.device))
        except Exception:
            pass
        return info

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init(X_mean, n, g, k, seeds, dev, dtype):
        import torch
        avg = float(np.sqrt(X_mean / k))
        Ws, Hs = [], []
        for s in seeds:
            gen = torch.Generator(device=dev).manual_seed(int(s))
            Hs.append(torch.randn(k, g, generator=gen, dtype=dtype, device=dev).abs_() * avg)
            Ws.append(torch.randn(n, k, generator=gen, dtype=dtype, device=dev).abs_() * avg)
        return torch.stack(Ws), torch.stack(Hs)

    def _solve(self, Xt, k, seeds, max_iter, tol):
        import torch
        from .kernels import w_sweep_fused, w_sweep_reference
        dev, dtype = Xt.device, Xt.dtype
        n, g = Xt.shape
        B = len(seeds)
        eps = torch.finfo(dtype).eps
        W, H = self._init(float(Xt.mean()), n, g, k, seeds, dev, dtype)
        active = torch.ones(B, dtype=torch.bool, device=dev)
        n_iter = torch.zeros(B, dtype=torch.int64, device=dev)
        viol_init = None
        sweep = w_sweep_fused if self.fused else w_sweep_reference

        for it in range(1, max_iter + 1):
            act = active.view(B, 1, 1)
            # usage sweep (in place, frozen replicates skipped)
            XHt = Xt @ H.permute(2, 0, 1).reshape(g, B * k)                 # (n, B*k) one 2D GEMM
            HHt = torch.matmul(H, H.transpose(1, 2)).contiguous()          # (B, k, k)
            viol = sweep(W, XHt, HHt, active, eps)
            # spectra sweep
            W2 = W.permute(1, 0, 2).reshape(n, B * k)                       # (n, B*k) copy
            WtX = (Xt.T @ W2).T.reshape(B, k, g)
            WtW = (W2.T @ W2).view(B, k, B, k).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
            Hn = H.clone()
            for j in range(k):
                grad = torch.matmul(WtW[:, j:j + 1, :], Hn)[:, 0, :] - WtX[:, j, :]
                hj = Hn[:, j, :]
                viol = viol + torch.where(hj == 0, grad.clamp_max(0), grad).abs().sum(1)
                Hn[:, j, :] = (hj - grad / WtW[:, j, j:j + 1].clamp_min(eps)).clamp_min(0)
            H = torch.where(act, Hn, H)

            n_iter += active.to(n_iter.dtype)
            if it == 1:
                viol_init = viol.clone()
            if self.trace is not None:
                self.trace.append((it, (viol / viol_init).cpu().numpy()))
            done = (viol <= tol * viol_init) | (viol_init == 0)
            active = active & ~done
            if not bool(active.any()):
                break
        return W, H, n_iter, ~active

    # ------------------------------------------------------------------ #
    def fit_batch(self, X, k: int, seeds, max_iter: int = 1000, tol: float = 1e-4, batch_size: int | None = None):
        """Fit every seed in ``seeds`` at rank ``k``; replicates are solved ``batch_size`` at a time."""
        import torch
        dev = self.device
        bs = int(batch_size or self.batch_size)
        Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
        Xt = torch.as_tensor(np.ascontiguousarray(Xd, dtype=np.float32), device=dev)
        del Xd
        out = []
        for i in range(0, len(seeds), bs):
            chunk = [int(s) for s in seeds[i:i + bs]]
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            W, H, n_iter, conv = self._solve(Xt, k, chunk, max_iter, tol)
            torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            Wn, Hn = W.cpu().numpy(), H.cpu().numpy()
            ni, cv = n_iter.cpu().numpy(), conv.cpu().numpy()
            for b, s in enumerate(chunk):
                out.append(Fit(Wn[b], Hn[b], int(ni[b]), s, bool(cv[b]),
                               extra={"batch_seconds": dt, "batch_size": len(chunk)}))
            del W, H
        del Xt
        torch.cuda.empty_cache()
        return out
