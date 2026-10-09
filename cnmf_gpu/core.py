"""``cNMFGPU``: cNMF with a selectable factorization solver.

Only ``prepare`` (one extra argument group, recorded to disk) and ``factorize`` (dispatch) are overridden.
Everything else, including the files written, is cNMF's own code, so ``combine``, ``consensus``,
``k_selection_plot`` and ``load_results`` behave exactly as in cNMF.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from cnmf import cNMF
from cnmf.cnmf import load_df_from_npz, save_df_to_npz, worker_filter

GPU_SOLVER_VERSION = 1          # bump when the GPU solver's numerics change
SOLVERS = ("cpu", "gpu", "auto")


def gpu_available() -> tuple[bool, str]:
    from .solver import TorchHALS
    return TorchHALS.available()


def _gpu_compatible(nmf_kwargs: dict) -> tuple[bool, str]:
    """The GPU solver implements exactly cNMF's default: Frobenius loss, random init, no regularization."""
    if nmf_kwargs.get("beta_loss", "frobenius") != "frobenius":
        return False, f"beta_loss={nmf_kwargs.get('beta_loss')!r} (GPU solver supports 'frobenius' only)"
    if nmf_kwargs.get("init", "random") != "random":
        return False, f"init={nmf_kwargs.get('init')!r} (GPU solver supports 'random' only)"
    if float(nmf_kwargs.get("alpha_W", 0.0)) != 0.0 or float(nmf_kwargs.get("alpha_H", 0.0)) != 0.0:
        return False, "alpha_W / alpha_H != 0 (GPU solver has no regularization)"
    return True, ""


def select_solver(solver: str, nmf_kwargs: dict, n_cells: int, n_genes: int, max_k: int, batch_size: int,
                  device: str = "cuda") -> tuple[str, str]:
    """Resolve 'auto' and validate 'gpu'. Returns (solver, reason)."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if solver == "cpu":
        return "cpu", "requested"
    ok, why = gpu_available()
    if not ok:
        if solver == "gpu":
            raise RuntimeError(f"solver='gpu' requested but no usable GPU: {why}")
        return "cpu", f"auto: no GPU ({why})"
    ok, why = _gpu_compatible(nmf_kwargs)
    if not ok:
        if solver == "gpu":
            raise ValueError(f"solver='gpu' is incompatible with the NMF settings: {why}")
        return "cpu", f"auto: {why}"
    from .solver import choose_batch, free_device_memory
    free = free_device_memory(device)
    b = choose_batch(n_cells, n_genes, max_k, batch_size, free)
    if b < 1:
        msg = (f"matrix ({n_cells:,} cells x {n_genes} genes, K<={max_k}) does not fit in free GPU memory "
               f"({free / 2**30:.1f} GiB) even one replicate at a time")
        if solver == "gpu":
            raise MemoryError(msg)
        return "cpu", "auto: " + msg
    return "gpu", "requested" if solver == "gpu" else "auto: GPU available and matrix fits"


class cNMFGPU(cNMF):
    """cNMF whose ``factorize`` step can run on the GPU. See the module docstring."""

    def _initialize_dirs(self):
        super()._initialize_dirs()
        self.paths.setdefault("gpu_run_parameters", os.path.join(
            self.output_dir, self.name, "cnmf_tmp", self.name + ".solver_params.yaml"))
        self.paths.setdefault("gpu_factorize_log", os.path.join(
            self.output_dir, self.name, "cnmf_tmp", self.name + ".gpu_factorize_log.csv"))

    # ------------------------------------------------------------------ #
    def prepare(self, counts_fn, components, n_iter=100, densify=False, tpm_fn=None, seed=None,
                beta_loss="frobenius", num_highvar_genes=2000, genes_file=None, alpha_usage=0.0,
                alpha_spectra=0.0, init="random", max_NMF_iter=1000,
                solver: str = "cpu", gpu_batch_size: int = 16, gpu_device: str = "cuda"):
        """cNMF ``prepare`` plus the solver choice, which is resolved now and written to
        ``<name>.solver_params.yaml`` so that ``factorize`` on any worker uses the same solver.

        solver : 'cpu' (cNMF's sklearn cd, the default), 'gpu' (error if unusable), or 'auto'
                 (gpu if a CUDA device is present, the NMF settings are supported and the matrix fits).
        gpu_batch_size : replicates solved simultaneously; reduced automatically to fit device memory.
        """
        super().prepare(counts_fn, components, n_iter=n_iter, densify=densify, tpm_fn=tpm_fn, seed=seed,
                        beta_loss=beta_loss, num_highvar_genes=num_highvar_genes, genes_file=genes_file,
                        alpha_usage=alpha_usage, alpha_spectra=alpha_spectra, init=init,
                        max_NMF_iter=max_NMF_iter)
        self.set_solver(solver, gpu_batch_size=gpu_batch_size, gpu_device=gpu_device)

    def set_solver(self, solver: str = "cpu", gpu_batch_size: int = 16, gpu_device: str = "cuda"):
        """Choose the factorize solver for an already-prepared run (what ``prepare(solver=...)`` calls)."""
        nmf_kwargs = yaml.load(open(self.paths["nmf_run_parameters"]), Loader=yaml.FullLoader)
        run_params = load_df_from_npz(self.paths["nmf_replicate_parameters"])
        nc = sc.read_h5ad(self.paths["normalized_counts"], backed="r")
        n_cells, n_genes = nc.shape
        max_k = int(run_params["n_components"].max())
        resolved, reason = select_solver(solver, nmf_kwargs, n_cells, n_genes, max_k, gpu_batch_size, gpu_device)
        info = {"requested": solver, "solver": resolved, "reason": reason, "gpu_batch_size": int(gpu_batch_size),
                "gpu_device": gpu_device, "gpu_solver_version": GPU_SOLVER_VERSION}
        if resolved == "gpu":
            from .solver import TorchHALS
            info.update(TorchHALS(batch_size=gpu_batch_size, device=gpu_device).describe())
        with open(self.paths["gpu_run_parameters"], "w") as f:
            yaml.safe_dump(info, f)
        print(f"[cnmf-gpu] factorize solver: {resolved} ({reason})", flush=True)
        return resolved

    def solver_params(self) -> dict:
        p = self.paths["gpu_run_parameters"]
        if not os.path.exists(p):
            return {"solver": "cpu", "reason": "no solver_params.yaml (run prepared without cnmf-gpu)"}
        return yaml.load(open(p), Loader=yaml.FullLoader)

    # ------------------------------------------------------------------ #
    def factorize(self, worker_i=0, total_workers=1, skip_completed_runs=False):
        """Dispatch to cNMF's CPU ``factorize`` or the GPU solver according to the recorded choice.

        With the GPU solver, this worker's jobs are grouped by K and solved in batches. Several workers can
        share the jobs as in cNMF (``--total-workers``), e.g. one per GPU with different ``gpu_device``.
        """
        params = self.solver_params()
        if params.get("solver", "cpu") != "gpu":
            return super().factorize(worker_i=worker_i, total_workers=total_workers,
                                     skip_completed_runs=skip_completed_runs)
        return self._factorize_gpu(worker_i, total_workers, skip_completed_runs, params)

    def _factorize_gpu(self, worker_i, total_workers, skip_completed_runs, params):
        from .solver import TorchHALS, choose_batch, free_device_memory

        run_params = load_df_from_npz(self.paths["nmf_replicate_parameters"])
        nmf_kwargs = yaml.load(open(self.paths["nmf_run_parameters"]), Loader=yaml.FullLoader)
        ok, why = _gpu_compatible(nmf_kwargs)
        if not ok:
            raise ValueError(f"GPU solver cannot run this prepare(): {why}")
        norm_counts = sc.read(self.paths["normalized_counts"])
        X = norm_counts.X
        n_cells, n_genes = X.shape
        tol, max_iter = float(nmf_kwargs.get("tol", 1e-4)), int(nmf_kwargs.get("max_iter", 1000))
        device = params.get("gpu_device", "cuda")

        if not skip_completed_runs:
            jobs = list(worker_filter(range(len(run_params)), worker_i, total_workers))
        else:
            jobs = list(worker_filter(run_params.index[run_params["completed"] == False], worker_i, total_workers))
        if not jobs:
            print(f"[Worker {worker_i}] nothing to do", flush=True)
            return
        sub = run_params.iloc[jobs]
        solver = TorchHALS(batch_size=int(params.get("gpu_batch_size", 16)), device=device)
        log_rows = []
        for k in sorted(sub["n_components"].unique()):
            rows = sub[sub["n_components"] == k]
            free = free_device_memory(device)
            bs = choose_batch(n_cells, n_genes, int(k), solver.batch_size, free)
            if bs < 1:
                raise MemoryError(f"K={k}: matrix does not fit in free GPU memory ({free / 2**30:.1f} GiB)")
            if bs < solver.batch_size:
                print(f"[Worker {worker_i}] K={k}: batch size reduced to {bs} to fit GPU memory", flush=True)
            t0 = time.perf_counter()
            fits = solver.fit_batch(X, int(k), [int(s) for s in rows["nmf_seed"]], max_iter=max_iter, tol=tol,
                                    batch_size=bs)
            dt = time.perf_counter() - t0
            for (_, p), f in zip(rows.iterrows(), fits):
                spectra = pd.DataFrame(f.spectra.astype(np.float64), index=np.arange(1, int(k) + 1),
                                       columns=norm_counts.var.index)
                save_df_to_npz(spectra, self.paths["iter_spectra"] % (int(p["n_components"]), int(p["iter"])))
                log_rows.append(dict(worker=worker_i, n_components=int(k), iter=int(p["iter"]),
                                     nmf_seed=int(p["nmf_seed"]), n_iter=f.n_iter, converged=f.converged,
                                     batch_size=f.extra["batch_size"], batch_seconds=f.extra["batch_seconds"]))
            n_unconv = sum(not f.converged for f in fits)
            iters = [f.n_iter for f in fits]
            print(f"[Worker {worker_i}] K={k}: {len(fits)} replicates in {dt:.1f}s "
                  f"(iterations median {int(np.median(iters))}, max {max(iters)})", flush=True)
            if n_unconv:
                warnings.warn(f"K={k}: {n_unconv} replicate(s) hit max_iter={max_iter} without reaching tol={tol}")
        log_path = self.paths["gpu_factorize_log"]
        log = pd.DataFrame(log_rows)
        if os.path.exists(log_path):
            log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
        log.to_csv(log_path, index=False)
