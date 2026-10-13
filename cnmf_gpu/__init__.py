"""cnmf-gpu: GPU factorization backend for dylkot/cNMF.

``cNMFGPU`` subclasses ``cnmf.cNMF`` and replaces only the ``factorize`` step. ``prepare``, ``combine``,
``consensus`` and ``k_selection_plot`` are cNMF's own code, so results are read with the usual cNMF tools.

    from cnmf_gpu import cNMFGPU
    obj = cNMFGPU(output_dir="runs", name="sample1")
    obj.prepare("counts.h5ad", components=[10, 20, 30], n_iter=100, seed=14, solver="gpu")
    obj.factorize()
    obj.combine()
    obj.consensus(k=20, density_threshold=0.1)

``cnmf_gpu.parallel.run_consensus`` runs ``consensus`` (and the ``k_selection_plot`` statistics) for many K in parallel
worker processes on the CPU; the work per K is cNMF's own code.

The solver choice ("cpu", "gpu" or "auto") is made at ``prepare`` time and recorded next to the run parameters,
so ``factorize`` on any machine does what the run was prepared for.
"""
from .core import cNMFGPU, GPU_SOLVER_VERSION, gpu_available, select_solver
from .parallel import run_consensus

__version__ = "0.2.0"
__all__ = ["cNMFGPU", "GPU_SOLVER_VERSION", "gpu_available", "select_solver", "run_consensus", "__version__"]
