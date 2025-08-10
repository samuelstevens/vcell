# scripts/submit_vcc.py

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys

import anndata as ad
import beartype
import numpy as np
import tyro


@beartype.beartype
def _print(msg: str) -> None:
    """Print a message with [submit] prefix."""
    print(f"[submit] {msg}")


@beartype.beartype
def _err(msg: str) -> None:
    print(f"[submit] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


@beartype.beartype
def _run(cmd: list[str]) -> None:
    _print(f"$ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise
    except subprocess.CalledProcessError as e:
        _err(f"command failed with code {e.returncode}")


@beartype.beartype
def _find_and_run_cell_eval(args: list[str]) -> None:
    # Try common invocations in preferred order
    candidates = []
    if shutil.which("uv"):
        candidates.append(["uv", "run", "cell-eval"])
        candidates.append(["uv", "run", "python", "-m", "cell-eval"])
    if shutil.which("cell-eval"):
        candidates.append(["cell-eval"])
    if shutil.which("uvx"):
        candidates.append(["uvx", "cell-eval"])
    # Last resort: python -m cell_eval (may or may not exist depending on package name)
    candidates.append([sys.executable, "-m", "cell_eval"])
    for base in candidates:
        try:
            _run(base + args)
            return
        except FileNotFoundError:
            continue
    _err(
        "Could not find a way to run cell-eval. Install it (e.g., `uv pip install -U cell-eval`) and try again."
    )


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for VCC submission preparation."""

    pred: pathlib.Path = pathlib.Path("pred_raw.h5ad")
    """Path to predictions h5ad file"""
    training_h5ad: pathlib.Path = pathlib.Path("data/inputs/vcc/adata_Training.h5ad")
    """Path to training h5ad file"""
    outdir: pathlib.Path = pathlib.Path("submissions")
    """Output directory for submission files"""
    genes: pathlib.Path | None = None
    """Optional preexisting genes.txt"""


def main():
    args = tyro.cli(Config)

    if not args.pred.exists():
        _err(f"missing predictions file: {args.pred}")
    if not args.training_h5ad.exists():
        _err(f"missing training h5ad: {args.training_h5ad}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Build genes.txt from training gene order (exactly 18,080)
    _print("loading training gene list…")
    A = ad.read_h5ad(args.training_h5ad, backed="r")
    genes = np.array(A.var_names)
    if genes.shape[0] != 18080:
        _print(f"WARN: training genes len={genes.shape[0]} (expected 18080)")

    genes_txt = args.genes or (args.outdir / "genes.txt")
    if args.genes is None:
        with open(genes_txt, "w") as f:
            for g in genes:
                f.write(f"{g}\n")
        _print(f"wrote {genes_txt} ({len(genes)} genes)")

    # Quick validation of pred_raw.h5ad
    _print("validating predictions file…")
    P = ad.read_h5ad(str(args.pred), backed="r")
    if "target_gene" not in P.obs.columns:
        _err("obs['target_gene'] column missing")
    if P.X is None:
        _err("X matrix missing")
    n_cells, n_genes = P.shape
    if n_genes != genes.shape[0]:
        _err(f"gene count mismatch: pred has {n_genes}, training has {genes.shape[0]}")
    if n_cells > 100_000:
        _err(f"too many rows: {n_cells} (cap is 100,000)")
    # dtype check (allow sparse); cast check via a small slice
    x0 = np.asarray(P.X[:1]).astype(np.float32)
    if x0.dtype != np.float32:
        _print(
            "WARN: X not float32 on read; cell-eval prep will coerce, but consider writing float32 upstream."
        )
    # exact gene order check
    if not np.array_equal(np.array(P.var_names), genes):
        _err("gene order mismatch between predictions and training var_names")

    # Run cell-eval prep in outdir; capture output filename by mtimes
    before = {p.name: p.stat().st_mtime for p in args.outdir.glob("*")}
    # Convert paths to absolute before changing directory
    pred_abs = args.pred.resolve()
    genes_txt_abs = genes_txt.resolve()
    prep_args = ["prep", "-i", str(pred_abs), "-g", str(genes_txt_abs)]
    # run in outdir so the output lands here
    cwd = os.getcwd()
    os.chdir(args.outdir)
    try:
        _find_and_run_cell_eval(prep_args)
    finally:
        os.chdir(cwd)

    # Find the newest file created in outdir
    after = list(args.outdir.glob("*"))
    newfiles = [
        p
        for p in after
        if p.name not in before or p.stat().st_mtime > before.get(p.name, 0)
    ]
    newfiles.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not newfiles:
        _err("cell-eval prep did not create an output file in outdir")
    out = newfiles[0]
    _print(f"success. Submission file: {out}")
    _print("upload this file on the Evaluation page. One submission per 24h.")


if __name__ == "__main__":
    main()
