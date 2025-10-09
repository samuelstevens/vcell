# TPU VM Setup Guide

This guide explains how to create and run experiments on Google Cloud TPU VMs using the automated initialization script.

## Prerequisites

**GCS Bucket Setup**: Create a Google Cloud Storage bucket and upload your dataset. Pick a bucket name.

```sh
gcloud storage buckets create gs://$BUCKET_NAME --location us-central1
```

> [!TIP]
> Sam calls his bucket `sam-vcc-us-central1` because it's sam's bucket for the virtual cell challenge and it's in us-central1.

Then copy some data to the bucket.

```sh
gcloud storage cp -r $FILES gs://$BUCKET_NAME
```

> [!TIP]
> Sam ran `gcloud storage rsync -r ./bucket gs://sam-vcc-us-central1/bucket`.
>
> ```
> [I] samstevens@localhoster /V/s/d/vcc> pwd
> /Volumes/samuel-stevens-2TB/datasets/vcc
> [I] samstevens@localhoster /V/s/d/vcc> tree
> .
> └── bucket
>     ├── NadigOConner2024_hepg2.csv
>     ├── NadigOConner2024_hepg2.h5ad
>     ├── NadigOConner2024_jurkat.csv
>     ├── NadigOConner2024_jurkat.h5ad
>     ├── ReplogleWeissman2022_K562_essential.csv
>     ├── ReplogleWeissman2022_K562_essential.h5ad
>     ├── ReplogleWeissman2022_K562_gwps.csv
>     ├── ReplogleWeissman2022_K562_gwps.h5ad
>     ├── ReplogleWeissman2022_rpe1.h5ad
>     ├── adata_Training.csv
>     ├── adata_Training.h5ad
>     └── pert_counts_Validation.csv
> 
> 2 directories, 12 files
> [I] samstevens@localhoster /V/s/d/vcc> gcloud storage rsync -r ./bucket gs://sam-vcc-us-central1/bucket
> ```
>
> This synced all files from `bucket` to GCS.

## Variable Reference

### gcloud CLI Variables (passed via `--metadata`)

| Variable | Example | Description |
|----------|---------|-------------|
| `git-commit` | `6976699` | Git commit hash to checkout from the vcell repo |
| `gcs-bucket` | `gs://sam-vcc-us-central1/bucket` | Full GCS path to your data bucket |
| `exp-script` | `experiments/14_repro_st_rn.py` | Path to Python experiment script (relative to repo root) |
| `exp-args` | `'--cfg configs/14-repro-st-rn.toml --vcc-root $DATA_ROOT'` | Arguments passed to the experiment script |
| `wandb-api-key` | `11b55b27...` | WandB API key for experiment tracking |
| `wandb-project` | `vcell` | WandB project name |
| `wandb-entity` | `samuelstevens` | WandB entity/username |

### Runtime Variables (automatically set by scripts/tpu-init.sh)

| Variable | Value | Description |
|----------|-------|-------------|
| `DATA_ROOT` | `~/data` | Directory where GCS data is downloaded to |
| `CODE_ROOT` | `~/vcell` | Directory where the vcell repo is cloned to |

### gcloud Standard Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `--zone` | `us-central1-a` | GCP zone (must match your TRC allocation) |
| `--accelerator-type` | `v5litepod-16` | TPU type (must match your TRC allocation) |
| `--version` | `v2-alpha-tpuv5-lite` | TPU runtime image version |

## How It Works

1. **You run** `gcloud compute tpus tpu-vm create` with metadata
2. **scripts/tpu-init.sh** runs automatically on VM startup:
   - Clones the vcell repo → checks out `$git-commit`
   - Downloads data from `$gcs-bucket` → `~/data` (sets `DATA_ROOT`)
   - Installs `uv` package manager
   - Runs: `uv run --extra tpu "$exp-script" $exp-args`
3. **Experiment script** loads config file and runs training
4. **Config file** (e.g., `configs/14-repro-st-rn.toml`) uses `$DATA_ROOT` for data paths

## Config File Setup

Config files can reference `$DATA_ROOT` which expands to `~/data` on the TPU VM:

```toml
# configs/14-repro-st-rn.toml
[vcc_dataset]
h5ad_fpath = "$DATA_ROOT/adata_Training.h5ad"
hvgs_csv = "$DATA_ROOT/adata_Training.csv"
pert_col = "target_gene"
ctrl_label = "non-targeting"
group_by = ["batch"]

[[datasets]]
h5ad_fpath = "$DATA_ROOT/ReplogleWeissman2022_K562_essential.h5ad"
hvgs_csv = "$DATA_ROOT/ReplogleWeissman2022_K562_essential.csv"
pert_col = "gene"
ctrl_label = "non-targeting"
group_by = ["batch"]
```

## Experiment Script CLI

Experiment scripts use `tyro` for CLI argument parsing and accept:

- `--cfg PATH`: Path to TOML config file
- CLI overrides for any config value (e.g., `--seed 123`, `--batch-size 64`)

Example:
```sh
uv run experiments/14_repro_st_rn.py --cfg configs/14-repro-st-rn.toml --vcc-root $DATA_ROOT --seed 42
```

## Monitoring

- **Startup logs**: `/tmp/startup.log` on the TPU VM
- **View logs**:
  ```sh
  gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE
  tail -f /tmp/startup.log
  ```
- **List TPU VMs**:
  ```sh
  gcloud compute tpus tpu-vm list --zone $ZONE
  ```

## Troubleshooting

### TPU creation fails

- Check your TRC allocation email for correct `--zone` and `--accelerator-type`
- Retry the command (TPU quota can be temporarily unavailable)
- Note: You can only have 2 TPU VMs at a time with TRC; delete old ones first

### Data download fails

- Check bucket path is correct: `gcloud storage ls gs://your-bucket-name/`
- SSH into VM and check `/tmp/startup.log`

### Experiment doesn't start

- SSH into VM: `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE`
- Check startup logs: `cat /tmp/startup.log`
- Verify experiment script path is correct relative to repo root
- Verify commit hash exists: `git log --oneline | grep $COMMIT`

### Config file path issues

- Config paths in `exp-args` must be relative to repo root
- `$DATA_ROOT` is only expanded on the TPU VM, not locally

## Tips

- **Set default project**: `gcloud config set project PROJECT_ID` to avoid `--project` flag
- **Use spot instances**: Add `--spot` flag (can be preempted)
- **Delete old VMs**: `gcloud compute tpus tpu-vm delete $NAME --zone $ZONE`
- **Check runtime versions**: `gcloud compute tpus tpu-vm versions list`
