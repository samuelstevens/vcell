# 06/29/2025

LambdaAI application

Related Publications (if any)
Provide a link to your own relevant papers, project pages, or other supporting materials such as demos or prototypes which may have shown good results on small datasets

BioCLIP: foundation vision model for all living organisms (university project, https://imageomics.github.io/bioclip/)
BioCLIP 2: sequel to BioCLIP, demonstrates that scale leads to emergent understanding of biological traits (university project, https://imageomics.github.io/bioclip-2/)
Mind the Gap and BioBench: evaluates both MLLMs and vision models on a wide spread of biology-relevant tasks with application-specific metrics (https://github.com/samuelstevens/mindthegap, https://samuelstevens.me/biobench)

Research Problem
What problem does your research address, why is it important, and how success looks like

Problem: Current AI models cannot reliably predict how cells respond to genetic perturbations across different cell types. Despite massive single-cell datasets, we lack standardized benchmarks to evaluate whether models capture generalizable biological mechanisms or just dataset artifacts.

Importance: Virtual cells that accurately predict perturbation responses would transform biology—replacing expensive experiments, accelerating drug discovery, and revealing causal gene-function relationships. The field currently wastes resources on models that fail to generalize beyond training contexts.

Success metrics: The Virtual Cell Challenge defines success through three complementary metrics:
1. Differential expression accuracy - predicting which genes change expression after perturbation
2. Perturbation discrimination - distinguishing between different genetic interventions by their effects  
3. Global expression prediction (MAE) - capturing the full transcriptomic response

Success means building models that excel across all three metrics when predicting responses to held-out perturbations in new cell types with minimal adaptation data. The immediate benchmark: accurately predicting 100 unseen genetic perturbations in H1 stem cells given only 150 training perturbations in that cell type, leveraging cross-cell-type knowledge from 350M+ cells in public datasets.

Long-term success: models that replace bench experiments for perturbation screening, enabling in silico exploration of genetic interventions across any human cell type.

Relevance and Novelty
What are the key related works and trends in this field, and how does your approach offer something new?

---

Based on this, it seems like we need an actual idea before getting compute grants.

# 07/20/2025

Some links:

Jax code that I know works:

- https://github.com/samuelstevens/mlm-pretraining
- https://github.com/samuelstevens/frx

Tutorials on anndata:

- https://anndata.readthedocs.io/en/stable/tutorials/notebooks/getting-started.html
- https://scverse-tutorials.readthedocs.io/en/latest/notebooks/anndata_getting_started.html

Biological primer for ML folks:

- https://fleetwood.dev/posts/virtual-cell-challenge

# 07/22/2025

- https://biothings-clientpy.readthedocs.io/en/latest/doc/quickstart.html#use-the-client-for-mygene-info-api-genes
- https://nbviewer.org/gist/newgene/6771106

# 07/23/2025

Submission is still not passing.
Desperately working with the arc folks on that.

Other than that, I now have compute with Google TRC.
So I need to get up to speed on downloading the STATE public data, then we can train a clone of the model and demonstrate that we're not imcompentent.

What are some public TODOs for everyone to work on?

1. Download the data. Look at it.
2. Make a random baseline submission, or a mean expression baseline submission. If you can successfully get a score, you've made progress for the team.
3. Get Google TRC access. This seems to be a reliable source of TPUs/compute for researchers. We could all take advantage of this. Seems like it's ~$30-60 a month for incidentals besides the TPU VM (static IPs, storage, etc).
4. Find other free compute grants. I have some at the top of my logbook.

# 07/28/2025

Looking at the dataloading in the STATE model, we need:

1. A dataloader that batches by batch and cell line. This is in Arc institute's [PerturbationBatchSampler](https://github.com/ArcInstitute/cell-load/blob/367708ca193860d79a04f0a02c2d6eb128bbf80b/src/cell_load/data_modules/samplers.py#L16): 
```py
class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same
    (cell_type, perturbation) combination, using only H5 codes.

    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.

    Supports distributed training.
    """
```

However, their implementation uses pytorch samplers to handle this.
I think we can avoid torch in favor of grain.

But the same problems as always remain.
Efficient dataloading of a large dataset.


# 07/29/2025

I think I will implement a very naive dataloader initially.
There's no point in writing a beautiful multiprocess'ed dataloader if the naive one is fast enough.

# 08/02/2025

I tried making a naive dataloader for the replogle data to recreate the STATE training.
I don't know how successful I was.

# 08/09/2025

```sh
gcloud compute tpus tpu-vm create tpu-2 --project trc-project-466816 --zone us-central1-f --accelerator-type v2-8 --version tpu-ubuntu2204-base
```

I picked out the various options here:

![Image of my google cloud account, showing that my project trc-project has ID trc-project-466816]()

![Image of my email from TRC, showing that I have access to `v2-8` TPU VMs in `us-central1-f`]()

```sh
gcloud compute tpus tpu-vm ssh tpu-2 --project trc-project-466816 --zone us-central1-f
```


# 08/10/2025

# What I got done

* Wrote unit + property tests for metrics (identity, control baseline, derangement, gene-order trap).
* Built a tiny deterministic Equinox model that mixes tokens via pooled context.
* Implemented streaming inference with a memmap to avoid 7 GB RAM spikes; generated `pred_raw.h5ad`.
* Set up deterministic control pooling and consistent log1p gene space for predictions.
* Wrote `tools/submit_vcc.py`.

# Challenges

* Uploads are slow: 1.1 MB/s aligns with a 10 Mb/s uplink; full 100k files take 1 hour.
* Unseen validation perturbations mean learned ID embeddings don’t help; training won’t move PDS/DE much without metadata features.
* Potential variance from changing control sets; fixed by seeding and reusing a canonical pool.

# What’s next

1. Submit a .vcc file.
2. Record scores; verify basic expectations (MAE magnitude, no schema errors).
3. Add a tiny local "fake val" harness from training to sanity-check metrics end-to-end.
4. Implement metrics to satisfy tests:
    * `compute_pds` (L1 distance, exclude target gene, normalized inverse rank, top-k)
    * `compute_de` (Wilcoxon rank-sum, BH FDR at 0.05, overlap; optional PR-AUC, Spearman)
5. Train the same model on H1 train:
    * control set → predicted cells
    * still use OOV=0 for unseen val IDs
    * compare to random init via your local metrics
6. Add a "fixed pseudobulk" mode: predict and then average per-perturbation to reduce variance; measure MAE vs per-cell.

Other stuff

-Port a minimal STATE-style mapper (JAX) that consumes control set + perturbation embedding; keep it small.
-Explore using metadata embeddings (target-gene features) so val IDs aren’t blind.
-Wire scPerturb (CRISPR-only) loaders and alignment to VCC genes (zero-fill missing genes), then fine-tune.
-SAE-on-residuals idea: cache ST residuals and train an SAE for interpretability; design a small eval (e.g., residual attribution to DE genes).


# 08/13/2025

Maciej is looking for some guidance on the Jax-based metrics.
He "doesn't even know where to start".
I think we need to provide some very high-level context. It should be simple, concise and intuitive.

# 08/15/2025

Great news! I think we have a complete process of training and evaluating a model!

TODOs

1. Add wandb logging of loss and performance (speed)
2. Add scperturb datasets


# 08/16/2025

1. Download datasets
2. Do EDA
3. Distributional loss
4. Run training. Observe speed issues.
   ^-- HAHHAHA like it's going to be this easy.

# 08/23/2025

1. Download datasets [done]
2. Do EDA
3. Distributional loss
4. Run training. Observe speed issues.
   ^-- HAHHAHA like it's going to be this easy.

What does it mean to do EDA?
I think I need to figure out and describe precisely what the training objective is, then I can write it, then I can force the data to fit it.

I also think it's critical to log normalize the data

So, my next set of tasks:

1. Log-normalize the data. Add checks to make sure this is always true?
2. Overfit on a single batch.
3. Include the scPerturb data.
4. Write a non-shit grain dataloader.
5. Read about queued resources? Maybe my training jobs need to be docker images that run?

Some tips from GPT:

- Log grad norms and param update norms each step. If updates ~0, LR too small or parameters frozen; if huge/NaN, LR too big or bad init.
- Check you’re training in the same space you evaluate: log-normalized in, log-normalized out. No hidden exp/log lingering.
- Verify data labels: each batch element’s (line, batch, perturbation) used for the same control in eval.
- Make your eval slice deterministic (fixed RNG, fixed HVG mask, fixed S cells). Randomness can hide real progress.

# 08/24/2025

I'm going to make a huge checklist of things to do.

- Log normalize the data.
- [done] Overfit with pseudobulk MSE
- Include the scPerturb data
- Write a non-shit grain dataloader.
- Learn about queued GCS resources
- Convert h5mu to h5ad
- Distributional loss (MDD2)
- Add a validation/holdout split.
- Log optimizer and loss as config options
- Log effect L1

My order is going to be

- [done] Los optimizer as config option
- [done] Log effect L1
- [done] Write a non-shit grain dataloader.
- [done] Include the scPerturb data
- Log normalize the data.
- Distributional loss (MDD2)
- Learn about queued GCS resources
- Convert h5mu to h5ad
- Add a validation/holdout split.
- Use a bigger model
- Log loss term as config option
- Metrics

# 08/27/2025

1. [done] Include scPerturb data.
2. Log normalization.
3. Use a slightly bigger model.

Including scPerturb checklist:

- Is the pert2id lookup consistent across datasets?
- Is the pert2id lookup consistent across reloads?
- Is the model's pert embedding table the right size
- Is the mask is being used?
- Is the mask correct?
- Are we stripping ensembl versions everywhere? -> what are ensembl versions?

Sequential

```sh
fio --name=net --filename=$FILENAME --rw=read --bs=1kb --direct=1 --iodepth=16 --runtime=30 --time_based
```

Random

```sh
fio --name=net --filename=$FILENAME --rw=randread --bs=1kb --direct=1 --iodepth=16 --runtime=30 --time_based
```

| Disk | Task | Result |
|---|---|---|
| Macbook SSD | Sequential Read | READ: bw=88.2MiB/s (92.4MB/s), 88.2MiB/s-88.2MiB/s (92.4MB/s-92.4MB/s), io=2645MiB (2773MB) |
| Macbook SSD | Random Read | READ: bw=31.5MiB/s (33.1MB/s), 31.5MiB/s-31.5MiB/s (33.1MB/s-33.1MB/s), io=946MiB (992MB) |
| External HDD | Sequential Read | READ: bw=700KiB/s (717kB/s), 700KiB/s-700KiB/s (717kB/s-717kB/s), io=20.5MiB (21.5MB) |
| External HDD | Random Read | READ: bw=4785B/s (4785B/s), 4785B/s-4785B/s (4785B/s-4785B/s), io=141KiB (144kB) |
| GCC PD | Sequential Read | READ: bw=2305KiB/s (2360kB/s), 2305KiB/s-2305KiB/s (2360kB/s-2360kB/s), io=67.5MiB (70.8MB) |
| GCC PD | Random Read | READ: bw=41.6KiB/s (42.6kB/s), 41.6KiB/s-41.6KiB/s (42.6kB/s-42.6kB/s), io=1248KiB (1278kB) |

# 08/30/2025

I am working on understanding how to leverage these six datasets for training:

- VCC training data
- Replogle x2, Nadig x2
- KOLF

So I think for each dataset source, we need to:

1. Map observation columns to ensembl IDs. If they have gene symbols but not ensembl IDs, then we need to look up the ensembld IDs using some web service
2. Measure the highly variable genes using scanpy for each dataset. Then we pick the top 2K across all datasets by (1) the intersection of HVGs across all dataset (2) fill the rest by which genes are HVGs in the most number of datasets
3. Record the keys to use for grouping by (batch, cell line, etc) for each dataset
4. Log-normalize each dataset CP10K + log1p

I think then we can sample a groupby key, select a perturbation, then sample S control RNA transcriptomes, sample S perturbed RNA transcriptomes, and use the 2K HVGs as input/output instead of the 18K. Then we will fill in the remaining 16K with mean counts, which should be okay since

1. The baseline is mean counts, so we aren't any worse
2. They're not highly variable, so they shouldn't change too much

# 08/31/2025

I am getting absolutely railed by the complexity of this project.
I almost don't know how to keep all the different pieces in my head.
There are more moving parts than I know what to do with.

Broadly, I think we can summarize it into:

1. Data preparation (HVGs, ensembl ID mapping, data loader, reformatting data for efficient row-reads, etc)
2. Cloud preparation (GCS buckets, persistent disks, spot instances, queued resources, init.sh, etc)
3. Misc (transformer architecture, wiring it all together, picking mean counts for non-HVG genes, testing)

I think I should work on them in this order too. The key is to do the bare minimum necessary so that I can keep making progress, recognizing that I can come back to the different stages.


# 09/03/2025

- Pick out HVGs from all datasets besides VCC
- Cross-reference those with that vary a lot from control to perturbation in just the VCC data
- We could have some highly variable with the 50 validation perturbations

# 09/22/2025

Well, I'm geting railed by the context switch again.
This job is fucking hard.
I implemented a transformer model with similar architecture to the ST model.
Now I need to update the code to use the common HVGs from all the datasets.
Don't worry about canonicalizing the genes.

# 09/23/2025

I think I wired the model up correctly.
GPT suggested a couple experiments.

1. Permute the inputs and then unpermute the model's outputs back to the original order and they should match.
2. Train with only control perturbations. We should quickly get to 0 loss.
3. Train with random perturbation IDs. We should not be able to meaningfully improve beyond a random prediction.
4. Training with random genes instead of the 2K HVGs should be worse than the HVGs.

I think I then need to train on VCC + Replogle + Nadig and see what we can do.
Finally, we can train with the Nourreddine 2025 as well.

# 09/25/2025

The Replogle-Nadig ST model seems to only have ~2.49M parameters.
Really, the transformer itself has 1.84M parameters, so about 650K, or 25% are embedding/MLP layers.
I think this is feasible to set up on my laptop.
With one layer and h=64, there's only 115K transformer parameters.

So I just need to pick out the HVGs for all four Replogle + Nadig datasets, then train a model on it.
I can use the VCC training data as a validation split.


1. Use the four datasets (2x Replogle, 2x Nadig)
2. Calculate the HVGs across these 4 datasets
3. Train an ST transformer from scratch with h=128, 4 layers, 8 attention heads, using the exact same setup as in Table 3
4. Measure zero-shot context generalization by evaluating the model on the VCC training split as a validation set
5. Using perturbation mean baseline prediction for non-HVGs


# 09/29/2025

To avoid setting the --project flag in every gcloud CLI command, use the gcloud config set command to set the project ID in your active configuration:

```sh
gcloud config set project project-id
```

This worked:

```sh
gcloud compute tpus tpu-vm create demo --zone=us-central1-a --accelerator-type=v5litepod-16 --version=v2-alpha-tpuv5-lite --metadata "experiment-script=$SCRIPT,experiment-args=$ARGS,wandb-api-key=$WANDB_KEY,wandb-project=$WANDB_PROJECT,wandb-entity=$WANDB_ENTITY"  --metadata-from-file startup-script=scripts/tpu-init.sh --spot
```
Now we do this:

```sh
gcloud compute tpus tpu-vm create TPUNAME1 --zone us-central1-a --accelerator-type v5litepod-16 --version v2-alpha-tpuv5-lite --spot --metadata git-commit=5d4204d,gcs-bucket=gs://sam-vcc-us-central1/bucket,exp-path=experiments/14_repro_st_rn.py,exp-args='--cfg configs/14-repro-st-rn.toml --vcc-root $ROOT',wandb-api-key=11b55b27cf1dab08762cd33c62e329ed291aa5ae,wandb-project=vcell,wandb-entity=samuelstevens --metadata-from-file startup-script=scripts/tpu-init.sh
```

Some lessons:

- For some reason, I can only have two TPU VMs at a time. So I need to create one, then delete the old one.
- I think I want to use rsync instead of cp for gcloud.
- I need to document all these variables. It's really important. Ideally there would be more structure than just natural language documentation.

# 09/30/2025

1. Can we train on the TPU? Just one slice/pod/etc.
2. Can we make changes to maciej's script instead of making a new script -> yes!
3. Set a baseline with 24 hours of training on one TPU.

```sh
gcloud compute tpus tpu-vm create TPUNAME1 --zone us-central1-a --accelerator-type v5litepod-16 --version v2-alpha-tpuv5-lite --spot --metadata git-commit=6976699,gcs-bucket=gs://sam-vcc-us-central1/bucket,exp-script=experiments/14_repro_st_rn.py,exp-args='--cfg configs/14-repro-st-rn.toml --vcc-root $DATA_ROOT',wandb-api-key=11b55b27cf1dab08762cd33c62e329ed291aa5ae,wandb-project=vcell,wandb-entity=samuelstevens --metadata-from-file startup-script=scripts/tpu-init.sh
```

Got it working with

```sh
gcloud compute tpus tpu-vm create TPUNAME1 --zone us-central1-a --accelerator-type v5litepod-16 --version v2-alpha-tpuv5-lite --spot --metadata git-commit=6976699,gcs-bucket=gs://sam-vcc-us-central1/bucket,exp-script=experiments/14_repro_st_rn.py,exp-args='--cfg configs/14-repro-st-rn.toml --vcc-root $DATA_ROOT',wandb-api-key=11b55b27cf1dab08762cd33c62e329ed291aa5ae,wandb-project=vcell,wandb-entity=samuelstevens --metadata-from-file startup-script=scripts/tpu-init.sh
```

I still need to:

1. Set up the code to use the TPUs. I can't get a single pod, so we need to use the slice. We can use data parallelism. Need to include which worker is logging, only log wandb on master, etc.
2. Set up a baseline with 24 hours on a v5e 16 TPU slice.

Resources for multi-TPU:

- https://cloud.google.com/tpu/docs/jax-pods
- https://docs.jax.dev/en/latest/multi_process.html
- https://docs.jax.dev/en/latest/the-training-cookbook.html
- https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
