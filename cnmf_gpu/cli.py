"""``cnmf-gpu`` command line: cNMF's commands and options plus ``--solver``.

    cnmf-gpu prepare   --output-dir runs --name s1 -c counts.h5ad -k 10 20 30 -n 100 --seed 14 --solver gpu
    cnmf-gpu factorize --output-dir runs --name s1
    cnmf-gpu combine   --output-dir runs --name s1
    cnmf-gpu consensus --output-dir runs --name s1 --local-density-threshold 0.1
    cnmf-gpu k_selection_plot --output-dir runs --name s1

Options mirror ``cnmf`` 1.7 (same names and defaults); the three ``--solver``/``--gpu-*`` options are new and
only read by ``prepare`` (``set_solver`` lets you change the choice of an existing run).
"""
from __future__ import annotations

import argparse

from cnmf.cnmf import load_df_from_npz

from .core import SOLVERS, cNMFGPU


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cnmf-gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "set_solver", "factorize", "combine", "consensus", "k_selection_plot"])
    p.add_argument("--name", type=str, nargs="?", default="cNMF", help="[all] analysis name; output in [output-dir]/[name]/")
    p.add_argument("--output-dir", type=str, nargs="?", default=".", help="[all] output directory")
    p.add_argument("-c", "--counts", type=str, help="[prepare] cell x gene counts: .h5ad, .mtx, df.npz or tab-delimited")
    p.add_argument("-k", "--components", type=int, nargs="+", help="[prepare/consensus] K values, e.g. -k 8 9 10")
    p.add_argument("-n", "--n-iter", type=int, default=100, help="[prepare] factorization replicates per K")
    p.add_argument("--total-workers", type=int, default=1, help="[factorize] workers sharing the jobs")
    p.add_argument("--seed", type=int, default=None, help="[prepare] seed for replicate seeds")
    p.add_argument("--genes-file", type=str, default=None, help="[prepare] genes to use, one per line")
    p.add_argument("--numgenes", type=int, default=2000, help="[prepare] number of high-variance genes")
    p.add_argument("--tpm", type=str, default=None, help="[prepare] precomputed TPM matrix")
    p.add_argument("--max-nmf-iter", type=int, default=1000, help="[prepare] max iterations per NMF run")
    p.add_argument("--beta-loss", choices=["frobenius", "kullback-leibler", "itakura-saito"], default="frobenius",
                   help="[prepare] NMF loss (GPU solver: frobenius only)")
    p.add_argument("--init", choices=["random", "nndsvd"], default="random", help="[prepare] NMF init (GPU solver: random only)")
    p.add_argument("--densify", action="store_true", default=False, help="[prepare] treat input as dense")
    p.add_argument("--worker-index", type=int, default=0, help="[factorize] index of this worker (from 0)")
    p.add_argument("--skip-completed-runs", action="store_true", default=False, help="[factorize] skip finished replicates")
    p.add_argument("--local-density-threshold", type=float, default=0.5, help="[consensus] local density filter (2 = none)")
    p.add_argument("--local-neighborhood-size", type=float, default=0.30, help="[consensus] neighbor fraction for density")
    p.add_argument("--show-clustering", action="store_true", help="[consensus] write the clustergram figure")
    p.add_argument("--build-reference", action="store_true", default=True, help="[consensus] write starCAT reference spectra")
    p.add_argument("--solver", choices=SOLVERS, default="cpu",
                   help="[prepare/set_solver] factorize solver: cpu (cNMF sklearn cd), gpu, or auto (default cpu)")
    p.add_argument("--gpu-batch-size", type=int, default=16, help="[prepare/set_solver] replicates solved at once on the GPU")
    p.add_argument("--gpu-device", type=str, default="cuda", help="[prepare/set_solver] torch device, e.g. cuda:1")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    obj = cNMFGPU(output_dir=args.output_dir, name=args.name)

    if args.command == "prepare":
        obj.prepare(args.counts, components=args.components, n_iter=args.n_iter, densify=args.densify,
                    tpm_fn=args.tpm, seed=args.seed, beta_loss=args.beta_loss, max_NMF_iter=args.max_nmf_iter,
                    num_highvar_genes=args.numgenes, genes_file=args.genes_file, init=args.init,
                    solver=args.solver, gpu_batch_size=args.gpu_batch_size, gpu_device=args.gpu_device)
    elif args.command == "set_solver":
        obj.set_solver(args.solver, gpu_batch_size=args.gpu_batch_size, gpu_device=args.gpu_device)
    elif args.command == "factorize":
        obj.factorize(worker_i=args.worker_index, total_workers=args.total_workers,
                      skip_completed_runs=args.skip_completed_runs)
    elif args.command == "combine":
        obj.combine(components=args.components)
    elif args.command == "consensus":
        run_params = load_df_from_npz(obj.paths["nmf_replicate_parameters"])
        ks = args.components if args.components else sorted(set(run_params.n_components))
        for k in ks:
            obj.consensus(int(k), args.local_density_threshold, args.local_neighborhood_size, args.show_clustering,
                          args.build_reference, close_clustergram_fig=True)
    elif args.command == "k_selection_plot":
        obj.k_selection_plot(close_fig=True)


if __name__ == "__main__":
    main()
