"""cnmf-gpu: GPU factorization backend for dylkot/cNMF.

``cNMFGPU`` subclasses ``cnmf.cNMF`` and replaces only the ``factorize`` step. ``prepare``, ``combine``,
``consensus`` and ``k_selection_plot`` are cNMF's own code, so results are read with the usual cNMF tools.

    from cnmf_gpu import cNMFGPU
    obj = cNMFGPU(output_dir="runs", name="sample1")
    obj.prepare("counts.h5ad", components=[10, 20, 30], n_iter=100, seed=14, solver="gpu")
    obj.factorize()
    obj.combine()
    obj.consensus(k=20, density_threshold=0.1)

The solver choice ("cpu", "gpu" or "auto") is made at ``prepare`` time and recorded next to the run parameters,
so ``factorize`` on any machine does what the run was prepared for.
"""
from .core import cNMFGPU, GPU_SOLVER_VERSION, gpu_available, select_solver

__version__ = "0.1.1"
__all__ = ["cNMFGPU", "GPU_SOLVER_VERSION", "gpu_available", "select_solver", "__version__"]
