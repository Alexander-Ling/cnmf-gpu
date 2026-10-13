# cnmf-gpu

GPU factorization backend for [dylkot/cNMF](https://github.com/dylkot/cNMF). It replaces the `factorize` step
(the slow one: hundreds of NMF fits per dataset) with a batched GPU solver and leaves `prepare`, `combine`,
`consensus` and `k_selection_plot` to cNMF's own code, so the outputs are ordinary cNMF outputs. Those CPU steps can
run one K per worker process (`--workers`), which makes post-processing of large runs several times faster.

- **Same algorithm as cNMF's default.** cNMF runs sklearn `non_negative_factorization(solver='cd')`, i.e.
  Cichocki-Phan HALS with a projected-gradient stopping rule. `cnmf-gpu` runs that algorithm, in the same
  sweep order with the same stopping rule, for many replicates at once in float32 on the GPU.
- **Explicit, recorded solver choice.** `prepare(..., solver="cpu" | "gpu" | "auto")` resolves the choice once
  and writes it to `<name>.solver_params.yaml` next to cNMF's run parameters. `factorize` reads it. The default
  is `cpu`, which is plain cNMF.
- **Speed.** On an RTX A4000 (16 GB) with a 165k-cell x 366-gene Xenium sample: 450 / 105 / 42 fits per minute
  at K = 10 / 20 / 30, versus 43 / 12 / 3.2 for cNMF with 16 CPU processes (Ryzen 9950X3D).

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build matching your driver
pip install git+https://github.com/Alexander-Ling/cnmf-gpu.git                  # pulls in cnmf and the rest
```

Or, for development, clone and `pip install -e ".[test]"`. Tested with Python 3.12, cnmf 1.7.1, torch 2.4.1
(CUDA 12.1), Triton 3.0, scikit-learn 1.9.

Requirements: Linux (including WSL2) with an NVIDIA GPU of compute capability 7.0 or newer for the fused
kernel. The torch Linux wheel brings Triton; without it the solver still works through a slower pure-torch
path. Without a CUDA device, `solver="auto"` falls back to the CPU and `solver="gpu"` raises. Native Windows
and macOS can run the CPU path only.

## Use

Python:

```python
from cnmf_gpu import cNMFGPU

obj = cNMFGPU(output_dir="runs", name="sample1")
obj.prepare("counts.h5ad", components=[10, 20, 30], n_iter=100, seed=14, solver="gpu")   # or "auto"
obj.factorize()                       # all replicates, batched per K
obj.combine()
obj.consensus(k=20, density_threshold=0.1)
obj.k_selection_plot()
```

Command line (same options as `cnmf`, plus `--solver`, `--gpu-batch-size`, `--gpu-device`):

```bash
cnmf-gpu prepare   --output-dir runs --name sample1 -c counts.h5ad -k 10 20 30 -n 100 --seed 14 --solver gpu
cnmf-gpu factorize --output-dir runs --name sample1
cnmf-gpu combine   --output-dir runs --name sample1
cnmf-gpu consensus --output-dir runs --name sample1 --local-density-threshold 0.1
```

`cnmf-gpu set_solver --solver gpu` switches an already-prepared run. `--total-workers N --worker-index i`
splits the replicates across workers as in cNMF; give each a different `--gpu-device` to use several GPUs.
`<name>.gpu_factorize_log.csv` records iterations, convergence and batch timing per replicate.

## Parallel consensus and k-selection (CPU)

cNMF's `consensus` and `k_selection_plot` run one K after another in one process, and none of it uses the GPU. Per K
they refit usages (NNLS), refit TPM spectra over all genes, regress z-scored TPM on the usages over all genes, and draw
a clustergram, so with a full transcriptome and many K this post-processing can take longer than the GPU factorization.
Every pass is independent, so `--workers` runs them in worker processes (`cnmf_gpu.parallel.run_consensus`):

```bash
cnmf-gpu consensus        --output-dir runs --name sample1 --local-density-threshold 0.1 0.2 --workers 0 --max-memory-gb 40
cnmf-gpu k_selection_plot --output-dir runs --name sample1 --workers 0
```

```python
from cnmf_gpu import run_consensus
run_consensus(obj, ks=None, density_thresholds=[0.1], workers=0, k_selection=True, max_memory_gb=40)
```

- **Same results.** Each task is one pass of cNMF's own `consensus` code ((K, density threshold) or the K's
  k-selection statistics), so the files are the same as cNMF's. The numbers match the serial run to floating-point
  summation order (about 1e-16 relative in the benchmarks below), not bit for bit.
- **One copy of the data.** The parent reads the normalized counts and the TPM matrix once and writes them as plain
  arrays to shared memory (`/dev/shm`); workers map them read-only, so cNMF cannot modify them unnoticed. The parent
  also writes cNMF's per-K local-density cache beforehand, so all passes are independent.
- **Memory decides the worker count, cores become threads.** A worker's memory is the transient copies cNMF and sklearn
  make (measured at up to 1.5x the two matrices) plus 0.5 GiB; threads add none. `--workers 0` takes as many workers as
  fit in `--max-memory-gb` (default 70% of available memory, minus the shared copy) and at most cores - 1;
  `--threads-per-worker 0` (default) gives the spare cores to each worker as BLAS/OpenMP threads. The estimate is printed.
- **Robust to interruption.** SIGTERM cleans up the shared-memory directory; directories left by killed runs are
  removed at the next start. A worker that dies (usually out of memory) stops the run with a message naming the sizing.
- Workers start with the `spawn` method (forking a process that has used OpenMP aborts), which re-imports the calling
  script: call `run_consensus` from a script only under `if __name__ == "__main__":`. The CLI and notebooks need nothing.
- `--workers 1` (the default) is cNMF's serial loop, unchanged.

Benchmark (Ryzen 9950X3D, 32 threads, WSL2, `--workers 0 --max-memory-gb 40`; serial = cNMF's own loop with all BLAS
threads; all outputs compared with the serial run's files):

| Run | Passes | Workers x threads | Peak RAM | Parallel | Serial | Max relative difference |
|---|---|---|---|---|---|---|
| scRNA, 64k cells x 2,000 genes, 10 K x 2 thresholds + stats | 30 | 12 x 2 | 30.7 GB | 122 s | 840 s | 9e-17 |
| scRNA, 157k cells x 2,000 genes, 13 K x 2 thresholds + stats | 39 | 5 x 6 | 30.5 GB | 667 s | 2,280 s | 2e-16 |
| Xenium, 24.7k cells x 366 genes, 36 K x 1 threshold + stats | 72 | 31 x 1 | 16.4 GB | 18 s | 81 s | 7e-16 |

The speed-up is bounded by the slowest single pass and by memory bandwidth: a pass runs 1.5-3x slower when many run
at once, and the sparse products inside cNMF's refits are single-threaded. A larger memory budget buys more workers;
beyond about 10 concurrent passes the gain is small.

## What is and is not the same as cNMF

Same: input handling, gene selection and scaling, replicate seeds table, file layout, consensus, statistics.

Different, by necessity:

- **Random stream.** Initial factors follow sklearn's `random` scheme (`sqrt(mean(X)/k) * |N(0,1)|`) but are
  drawn with torch's generator on the device. A given seed therefore does not reproduce the CPU replicate.
  Compare runs at the consensus level, not replicate by replicate.
- **Reproducibility.** The same seed on the same GPU and torch version reproduces a replicate; different GPU
  models or CUDA/torch versions can differ at float32 rounding level, which can change where a fit stops and,
  rarely, which side of a bistable program the consensus lands on. `solver_params.yaml` records the GPU,
  torch and CUDA versions and a `gpu_solver_version` that is bumped whenever the solver's numerics change.
- **Precision.** float32 on a dense copy of the normalized matrix (cNMF uses float64, sparse). The two
  products that reduce over cells are done as 2D GEMMs, which keeps the stopping statistic accurate to ~1e-6.
- **Supported settings.** `beta_loss='frobenius'`, `init='random'`, no regularization (cNMF's defaults).
  Other settings raise with `solver="gpu"` and fall back to the CPU with `solver="auto"`.
- **Memory.** The dense float32 matrix plus about 16 bytes per cell per component per batched replicate must
  fit in free GPU memory. The batch size is reduced automatically; if even one replicate does not fit the run
  stops with a clear error. There is no cell-dimension chunking yet. For reference, 339k cells x 2000 genes
  (2.7 GB) runs at full batch size on a 16 GB card; 671k x 366 also does.

## Validation

Every arm below was run through the `cnmf-gpu` CLI end to end (prepare, factorize, combine, consensus,
k_selection_plot): two independent CPU runs (`--solver cpu`, 16 workers, seeds 14 and 15) and one GPU run
(`--solver gpu`, seed 14), K = 10, 20, 30, 50 replicates per K, density threshold 0.10. Consensus programs were
matched one-to-one by cosine; a "mismatch" is a matched pair with cosine below 0.9. The CPU-versus-CPU column is
the reproducibility of cNMF with itself and is the yardstick for the GPU columns.

| Dataset | cells x genes | K=10: CPU-CPU, GPU-CPU | K=20 | K=30 |
|---|---|---|---|---|
| Xenium Patient 2 | 55,541 x 366 | 0, 0 / 0 | 0, 0 / 0 | 0, 0 / 0 |
| Xenium Patient 4 | 165,553 x 366 | 0, 0 / 0 | 0, 0 / 0 | 1, 0 / 2 |
| Xenium patient 3 | 238,194 x 366 | 0, 0 / 0 | 2, 2 / 0 | 2, 0 / 2 |
| Xenium Q044 | 670,872 x 366 | 0, 0 / 0 | 0, 0 / 0 | 0, 0 / 0 |
| scRNA-seq GBmap balanced | 59,234 x 2000 (cNMF HVG) | 1, 0 / 1 | 0, 1 / 1 | 1, 0 / 1 |

"GPU-CPU a / b" = mismatches against CPU run A (same replicate seeds) / CPU run B. In 14 of 15 cells of the
table the GPU count is within the CPU-versus-CPU range. The exception (GBmap, K=20) and every other mismatch
seen are the same phenomenon: a program that is bistable across random starts (for GBmap K=20, an erythroid
HBD/HBM/AHSP program versus a fibroblast DCN/C7/COL6A3 one), found by both solvers in a third to a half of
their replicates, where the consensus picks whichever wins the density filter and k-means. Silhouette stability
and prediction error of the GPU runs fall inside the range of the two CPU runs on every dataset.

An scRNA-seq run at scale, the full GBmap core atlas (338,564 cells x 2000 HVG), ran GPU-only: all 150
replicates converged at batch size 16 (2.7 GB dense matrix) in 416 s; no CPU comparison was made.

Wall time of the factorize step for the 150 replicates, each arm run alone on an otherwise idle machine
(CPU: 16 worker processes on a Ryzen 9 9950X3D, 106 GB RAM; GPU: RTX A4000 16 GB, batch size 16):

| Dataset | CPU, 16 processes | GPU | speed-up |
|---|---|---|---|
| Patient 2 (55k cells x 366) | 209 s | 38 s | 5.5x |
| patient 3 (238k x 366) | 1421 s | 130 s | 10.9x |
| Q044 (671k x 366) | 4760 s | 517 s | 9.2x |
| GBmap balanced (59k x 2000 genes) | 271 s | 70 s | 3.9x |
| GBmap core (339k x 2000 genes) | not run | 292 s | |

The GPU advantage grows with cell count (the batched solver amortizes the matrix across replicates) and is
smallest on small or sparse inputs, where the CPU's sparse arithmetic is efficient and fixed costs dominate.
Per-K iteration counts and batch timings for every run are in each run's `gpu_factorize_log.csv`.
Dedicated, uncontended benchmarks on Patient 4 gave 450 / 105 / 42 fits per minute at K = 10 / 20 / 30 versus
43 / 12 / 3.2 for the 16-process CPU. Study scripts and full reports are in the companion
`GPU_Solver_Benchmark` folder (`results/validation_<dataset>/`).

Not validated: matrices above ~1M cells on a 16 GB card (no cell chunking), non-default NMF settings.

## Why the stopping rule matters

A progress-based rule (stop when the objective barely changes), which sklearn uses for its `mu` solver and
which this solver used in an early version, stops on plateaus where programs are still splitting and merging,
and produced consensus programs with near-zero cosine to the CPU's. The projected-gradient rule is an
optimality test and does not. Do not swap it for a cheaper rule.

## Tests

```bash
pytest -q tests
```

Covers: Triton kernel against the torch reference sweep, KKT optimality at the stop, batched versus single
fits, reproducibility, sparse input, solver selection, and an end-to-end prepare/factorize/combine/consensus
run where GPU and CPU consensus programs must agree, and parallel consensus / k-selection against cNMF's serial
code (same files, same numbers). GPU tests skip automatically without a CUDA device; the
GitHub workflow runs the CPU-only subset.

## Citing

This package only replaces the factorization step. If you use it, cite cNMF:
Kotliar D. et al., *Identifying gene expression programs of cell-type identity and cellular activity with
single-cell RNA-Seq*, eLife 2019, https://doi.org/10.7554/eLife.43803. The solver is the HALS algorithm of
Cichocki & Phan (IEICE Trans. 2009), as implemented in scikit-learn's coordinate-descent NMF.

## License

MIT. cNMF itself is MIT licensed (dylkot/cNMF).
