# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata",
#     "beartype",
#     "h5py",
#     "mudata",
#     "numpy",
#     "pandas",
#     "scipy",
#     "tyro",
# ]
# ///

import dataclasses
import pathlib

import anndata as ad
import beartype
import h5py
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    dataset: pathlib.Path = pathlib.Path(
        "/Volumes/samuel-stevens-2TB/datasets/KOLF_Pan_Genome_Aggregate.h5mu"
    )
    """Path to the input .h5mu dataset."""

    output: pathlib.Path = pathlib.Path("./data/sampled_dataset.h5mu")
    """Path to save the sampled dataset."""

    n_samples: int = 10000
    """Number of samples to extract."""

    modality: str | None = None
    """Specific modality to sample (default: all modalities)."""

    seed: int = 42
    """Random seed for reproducibility."""


@beartype.beartype
def main(cfg: Config):
    print(f"Opening dataset: {cfg.dataset}")

    # Set random seed
    np.random.seed(cfg.seed)

    with h5py.File(cfg.dataset, "r") as f:
        # Navigate to modalities
        if "mod" not in f:
            print("Error: No 'mod' group found in file")
            return

        mod_names = list(f["mod"].keys())
        print(f"Found modalities: {mod_names}")

        # Filter modality if specified
        if cfg.modality:
            if cfg.modality in mod_names:
                mod_names = [cfg.modality]
                print(f"Processing only '{cfg.modality}' modality")
            else:
                print(f"Error: Modality '{cfg.modality}' not found")
                return

        # Get dimensions from first modality
        first_mod = f[f"mod/{mod_names[0]}"]
        if "X" in first_mod:
            X_group = first_mod["X"]
            n_obs = X_group.attrs.get("shape", [0, 0])[0]
        else:
            print("Could not determine number of observations")
            return

        print(f"Total observations: {n_obs}")

        # Generate sample indices
        n_samples = min(cfg.n_samples, n_obs)
        sample_indices = np.sort(np.random.choice(n_obs, size=n_samples, replace=False))
        print(f"Sampling {n_samples} observations")

        # Process each modality
        all_modalities = {}

        for mod_name in mod_names:
            print(f"\nProcessing modality: {mod_name}")
            mod_group = f[f"mod/{mod_name}"]

            # Get X matrix
            if "X" not in mod_group:
                print("  No X matrix found, skipping")
                continue

            X_group = mod_group["X"]

            # Handle sparse matrix
            if "data" in X_group:
                n_vars = X_group.attrs.get("shape", [0, 0])[1]
                print(f"  Matrix shape: {n_obs} x {n_vars} (sparse)")

                # Skip if matrix is too large (e.g., RNA with millions of genes)
                data_size_gb = (
                    X_group["data"].size * X_group["data"].dtype.itemsize / (1024**3)
                )
                if data_size_gb > 10:  # Skip if data array is >10GB
                    print(f"  Skipping - data array too large ({data_size_gb:.1f} GB)")
                    continue

                # Read sparse components
                data = X_group["data"][:]
                indices = X_group["indices"][:]
                indptr = X_group["indptr"][:]

                # Determine format and create matrix
                if len(indptr) == n_vars + 1:
                    # CSC format
                    full_matrix = sp.csc_matrix(
                        (data, indices, indptr), shape=(n_obs, n_vars)
                    )
                else:
                    # CSR format
                    full_matrix = sp.csr_matrix(
                        (data, indices, indptr), shape=(n_obs, n_vars)
                    )

                # Sample rows
                X_sampled = full_matrix[sample_indices, :]
            else:
                # Dense matrix
                shape = X_group.shape
                n_vars = shape[1]
                print(f"  Matrix shape: {n_obs} x {n_vars} (dense)")
                X_sampled = X_group[sample_indices, :]

            # Get metadata
            obs_df = pd.DataFrame()
            var_df = pd.DataFrame()

            # Process obs
            if "obs" in mod_group:
                obs_group = mod_group["obs"]
                if "_index" in obs_group:
                    obs_index = obs_group["_index"][:]
                    if len(obs_index) > 0 and isinstance(obs_index[0], bytes):
                        obs_index = [x.decode() for x in obs_index]
                    obs_df.index = pd.Index([obs_index[i] for i in sample_indices])

            # Process var
            if "var" in mod_group:
                var_group = mod_group["var"]
                if "_index" in var_group:
                    var_index = var_group["_index"][:]
                    if len(var_index) > 0 and isinstance(var_index[0], bytes):
                        var_index = [x.decode() for x in var_index]
                    var_df.index = pd.Index(var_index)

            # Create AnnData
            adata = ad.AnnData(X=X_sampled, obs=obs_df, var=var_df)
            all_modalities[mod_name] = adata
            print(f"  Sampled shape: {adata.shape}")

    if not all_modalities:
        print("No modalities could be sampled")
        return

    # Create MuData and save
    output_path = cfg.output
    if len(all_modalities) > 1:
        print(f"\nCreating MuData with {len(all_modalities)} modalities...")
        mdata = md.MuData(all_modalities)

        print(f"Saving to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mdata.write_h5mu(output_path)
    else:
        # Single modality - save as h5ad
        output_path = cfg.output.with_suffix(".h5ad")
        print(f"Saving single modality to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list(all_modalities.values())[0].write_h5ad(output_path)

    # Report size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput size: {actual_size_mb:.2f} MB")
    print(f"Successfully created sample with {n_samples} observations")


if __name__ == "__main__":
    main(tyro.cli(Config))
