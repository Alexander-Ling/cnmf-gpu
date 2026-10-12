"""Fused HALS usage (W) sweep for the batched GPU solver, written in Triton, plus a pure-torch reference.

The sweep updates usage components one at a time in sklearn's cd order (``_update_cdnmf_fast``):

    for j in 0..k-1:                         # component
        grad_j = W @ HHt[:, j] - XHt[:, j]   # uses the components already updated in this sweep
        pg     = grad_j, or min(grad_j, 0) where W[:, j] == 0   (projected gradient, for the stopping rule)
        W[:, j] = max(W[:, j] - grad_j / HHt[j, j], 0)

Rows (cells) are independent, so one Triton program loads a block of rows with all k columns into registers,
runs the k column updates there and writes the block back once. The torch loop it replaces re-streams the whole
(B, n, k) tensor for every component. Finished replicates (``active == 0``) are skipped, leaving their W as is.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - depends on platform
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _w_sweep_kernel(W_ptr, XHt_ptr, HHt_ptr, active_ptr, viol_ptr,
                        n, k, sWb, sWn, sXn, sXb, n_blocks, eps,
                        BLOCK: tl.constexpr, KP: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)
        if tl.load(active_ptr + pid_b) == 0:
            return
        rows = pid_r * BLOCK + tl.arange(0, BLOCK)
        cols = tl.arange(0, KP)
        rmask = rows < n
        cmask = cols < k
        mask = rmask[:, None] & cmask[None, :]
        w_off = pid_b * sWb + rows[:, None] * sWn + cols[None, :]
        W = tl.load(W_ptr + w_off, mask=mask, other=0.0)
        X = tl.load(XHt_ptr + rows[:, None] * sXn + pid_b * sXb + cols[None, :], mask=mask, other=0.0)
        hbase = HHt_ptr + pid_b * k * k
        viol = tl.zeros([BLOCK], dtype=tl.float32)
        for j in range(0, k):
            sel = cols[None, :] == j
            h = tl.load(hbase + j * k + cols, mask=cmask, other=0.0)          # row j of HHt (symmetric)
            grad = tl.sum(W * h[None, :], axis=1) - tl.sum(tl.where(sel, X, 0.0), axis=1)
            wj = tl.sum(tl.where(sel, W, 0.0), axis=1)
            pg = tl.where(wj == 0.0, tl.minimum(grad, 0.0), grad)
            viol += tl.abs(pg)
            hjj = tl.maximum(tl.load(hbase + j * k + j), eps)
            wnew = tl.maximum(wj - grad / hjj, 0.0)
            W = tl.where(sel, wnew[:, None], W)
        tl.store(W_ptr + w_off, W, mask=mask)
        tl.store(viol_ptr + pid_b * n_blocks + pid_r, tl.sum(tl.where(rmask, viol, 0.0), axis=0))


def w_sweep_fused(W: torch.Tensor, XHt: torch.Tensor, HHt: torch.Tensor, active: torch.Tensor,
                  eps: float, block: int | None = None, num_warps: int = 4) -> torch.Tensor:
    """In-place HALS W sweep on ``W`` (B, n, k, contiguous) for replicates with ``active`` set.

    ``XHt`` is (n, B*k) (output of the 2D GEMM X @ [H_1^T ... H_B^T]); ``HHt`` is (B, k, k) contiguous.
    Returns the per-replicate sum of |projected gradient| (B,), as sklearn's cd rule accumulates it.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not available; use w_sweep_reference")
    B, n, k = W.shape
    assert W.is_contiguous() and HHt.is_contiguous() and XHt.shape == (n, B * k) and XHt.is_contiguous()
    kp = max(2, triton.next_power_of_2(k))
    if block is None:
        # W and XHt blocks ([BLOCK, KP] fp32 each) live in registers. Keep the per-thread footprint constant as KP
        # grows, otherwise KP=64 (k 33..64) spills to local memory and the sweep runs 4-6x slower than at k<=32
        # (and far worse on very large n). 128 rows for KP<=32, 64 rows for KP=64, 32 rows for KP=128.
        block = max(16, 128 * 32 // kp) if kp > 32 else 128
    n_blocks = triton.cdiv(n, block)
    viol = torch.zeros((B, n_blocks), dtype=torch.float32, device=W.device)
    act = active.to(torch.int32)
    _w_sweep_kernel[(B, n_blocks)](W, XHt, HHt, act, viol, n, k, n * k, k, B * k, k, n_blocks, eps,
                                   BLOCK=block, KP=kp, num_warps=num_warps)
    return viol.sum(1)


def w_sweep_reference(W: torch.Tensor, XHt: torch.Tensor, HHt: torch.Tensor, active: torch.Tensor,
                      eps: float) -> torch.Tensor:
    """Plain-torch version of the same sweep (one column at a time). Slower; used when Triton is missing."""
    B, n, k = W.shape
    X = XHt.view(n, B, k).permute(1, 0, 2)
    Wn = W.clone()
    viol = torch.zeros(B, dtype=W.dtype, device=W.device)
    for j in range(k):
        grad = torch.matmul(Wn, HHt[:, :, j:j + 1])[:, :, 0] - X[:, :, j]
        wj = Wn[:, :, j]
        viol += torch.where(wj == 0, grad.clamp_max(0), grad).abs().sum(1)
        Wn[:, :, j] = (wj - grad / HHt[:, j:j + 1, j].clamp_min(eps)).clamp_min(0)
    W.copy_(torch.where(active.view(B, 1, 1), Wn, W))
    return viol
