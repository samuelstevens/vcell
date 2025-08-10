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


# 08/10/2025

1. Finish `tools/submit_vcc.py`.
2. Run `experiments/04_validation.py` to create `pred_raw.h5ad` (memmap path works).
3. Generate `genes.txt` from `adata_Training.h5ad` and run `cell-eval prep` via `just submit`.
4. Upload the prepped file on the Evaluation page (<=100k cells, includes controls).
5. Implement metrics to satisfy tests:
    * `compute_pds` (L1 distance, exclude target gene, normalized inverse rank, top-k)
    * `compute_de` (Wilcoxon rank-sum, BH FDR at 0.05, overlap; optional PR-AUC, Spearman)
6. Train the same model on H1 train:
    * control set → predicted cells
    * still use OOV=0 for unseen val IDs
    * compare to random init via your local metrics
7. Add a "fixed pseudobulk" mode: predict and then average per-perturbation to reduce variance; measure MAE vs per-cell.


Other stuff

-Port a minimal STATE-style mapper (JAX) that consumes control set + perturbation embedding; keep it small.
-Explore using metadata embeddings (target-gene features) so val IDs aren’t blind.
-Wire scPerturb (CRISPR-only) loaders and alignment to VCC genes (zero-fill missing genes), then fine-tune.
-SAE-on-residuals idea: cache ST residuals and train an SAE for interpretability; design a small eval (e.g., residual attribution to DE genes).

