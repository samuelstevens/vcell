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
- Log loss term as config option
- [done] Log effect L1
- [done] Write a non-shit grain dataloader.
- Log normalize the data.
- Include the scPerturb data
- Distributional loss (MDD2)
- Learn about queued GCS resources
- Convert h5mu to h5ad
- Add a validation/holdout split.
- Use a bigger model

# 08/27/2025

1. Include scPerturb data.
2. Log normalization.
3. Use a slightly bigger model.

Including scPerturb checklist:

- Is the pert2id lookup consistent across datasets?
- Is the pert2id lookup consistent across reloads?
- Is the model's pert embedding table the right size
- Is the mask is being used?
- Is the mask correct?
- Are we stripping ensembl versions everywhere? -> what are ensembl versions?

Disk speeds :)

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
