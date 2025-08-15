# experiments/01_mlp.py
# ruff: noqa
"""
Uses a simple embedding to predict a transcriptome from a perturbation.

The goal is to pre-train on the Replogle dataset for all genes in the virtual cell challenge, then fine-tune the model's MLP on the H1 cell data from the challenge, then evaluate on the unseen perturbations.

Fine-tuning is a TODO.

One epoch with bsz=1024 takes 6 minutes on my m1 Macbook.
"""

import collections.abc
import csv
import dataclasses
import json
import logging
import os
import pathlib
import time
import typing as tp

import anndata as ad
import beartype
import chex
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import vcell.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("01")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 42
    """Random seed."""

    # Model
    n_perts: int = 200
    """Number of total possible perturbations."""
    n_genes: int = 18080
    """Number of genes."""
    d_hidden: int = 1024 * 4
    """Hidden dimension."""

    # Data
    replogle: pathlib.Path = pathlib.Path(
        "data/inputs/replogle/ReplogleWeissman2022_K562_essential.h5ad"
    )
    genes_path: pathlib.Path = pathlib.Path("data/inputs/vcc/gene_names.csv")
    perts_path: pathlib.Path = pathlib.Path("data/inputs/vcc/perts.json")

    # Optimization
    learning_rate: float = 0.001
    """Peak learning rate."""
    batch_size: int = 1024
    """Batch size."""
    beta1: float = 0.9
    """Adam beta1."""
    beta2: float = 0.999
    """Adam beta2."""
    grad_clip: float = 1.0
    """Maximum gradient norm. `0` implies no clipping."""
    weight_decay: float = 0.0001
    """Weight decay applied to Optax's AdamW optimizer."""
    n_epochs: int = 10
    """Number of epochs to train for."""

    # Logging
    log_every: int = 10
    """how often to log metrics."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store model checkpoints."""


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    perts: eqx.nn.Embedding
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, n_perts: int, n_genes: int, d_hidden: int, *, key: chex.PRNGKey):
        key0, key1, key2 = jax.random.split(key, 3)
        self.perts = eqx.nn.Embedding(
            num_embeddings=n_perts + 1, embedding_size=d_hidden, key=key0
        )
        self.linear1 = eqx.nn.Linear(d_hidden, d_hidden, key=key1)
        self.linear2 = eqx.nn.Linear(d_hidden, n_genes, key=key2)

    def __call__(
        self, pert: Int[Array, ""], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, " n_genes"]:
        x_d = self.perts(pert)
        # Feed embedding through the MLP.
        x_h = self.linear1(x_d)
        x_h = jax.nn.gelu(x_h)
        x_d = self.linear2(x_h)

        return x_d


@beartype.beartype
def save(filename: str, cfg: Config, model):
    with open(filename, "wb") as fd:
        cfg_str = json.dumps(cfg)
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@jaxtyped(typechecker=beartype.beartype)
def compute_loss(
    model: eqx.Module,
    pert: Int[Array, " batch"],
    expr: Float[Array, "batch n_genes"],
) -> Float[Array, ""]:
    logits = jax.vmap(model)(pert)
    loss = jnp.mean((logits - expr) ** 2)

    return loss


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    pert: Int[Array, " batch"],
    expr: Float[Array, "batch n_genes"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, pert, expr)
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@beartype.beartype
class DataLoader:
    """
    This is not a long-term solution to dataloading.

    Some points:
    - Not every dataset has every gene in the transcriptome. Specifically, the Replogle dataset only has ~7K of the ~18K in vcc. We simply set them to 0 before log1p.
    -
    """

    def __init__(
        self,
        h5ad_path: pathlib.Path,
        *,
        vcc_perts: list[str],
        genes: list[str],
        batch_size: int,
        shuffle: bool = False,
        key: chex.PRNGKey | None = None,
    ):
        if shuffle:
            assert key is not None, "Need a key when shuffle is True."

        self.h5ad_path = h5ad_path
        self.vcc_perts = vcc_perts
        self.genes = genes

        self.batch_size = batch_size

        self.shuffle = shuffle
        self.key = key

        adata = ad.read_h5ad(self.h5ad_path, backed="r")

        # Map gene order once
        ad_genes = adata.var_names.str.upper().to_numpy()
        vcc_genes = np.asarray(self.genes)

        (self.vcc_cols,) = np.where(np.isin(vcc_genes, ad_genes))
        (self.ad_cols,) = np.where(np.isin(ad_genes, vcc_genes))

        keep_rows = np.isin(adata.obs["gene"].to_numpy(), self.vcc_perts)
        (self.row_pool,) = np.where(keep_rows)

        self.pert2id = {p: i for i, p in enumerate(self.vcc_perts)}

    @staticmethod
    def load_vcc_val_perts(path: pathlib.Path) -> list[str]:
        perts = []
        with open(path) as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                perts.append(row["target_gene"])

        return perts

    def __iter__(
        self,
    ) -> collections.abc.Iterable[
        tuple[Int[Array, " batch"], Float[Array, " batch n_genes"]]
    ]:
        adata = ad.read_h5ad(self.h5ad_path, backed="r")

        pert_ids = np.asarray([
            self.pert2id[p] for p in adata.obs["gene"].iloc[self.row_pool]
        ])

        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            row_pool = jax.random.permutation(subkey, self.row_pool)
        else:
            row_pool = self.row_pool

        for start, end in vcell.helpers.batched_idx(len(row_pool), self.batch_size):
            rows = np.sort(row_pool[start:end])

            # This line is slow as fuck because it's a ton of random reads.
            x_bg = adata.X[rows]

            # zero-pad to full 18 080-gene frame
            bsz, n_ad_genes = x_bg.shape
            x = np.zeros((bsz, len(self.genes)), dtype=x_bg.dtype)
            x[:, self.vcc_cols] = x_bg[:, self.ad_cols]

            yield pert_ids[start:end], np.log1p(jnp.asarray(x))


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    key, model_key = jax.random.split(key)

    # Model
    model_cfg = dict(n_perts=cfg.n_perts, n_genes=cfg.n_genes, d_hidden=cfg.d_hidden)
    model = Model(**model_cfg, key=model_key)
    logger.info("Initialized model.")

    # Data
    key, data_key = jax.random.split(key)
    genes = cfg.genes_path.read_text().strip().split("\n")
    perts = json.loads(cfg.perts_path.read_text().strip())
    dataloader = DataLoader(
        cfg.replogle,
        vcc_perts=perts["train"] + perts["val"],
        genes=genes,
        batch_size=cfg.batch_size,
        shuffle=True,
        key=data_key,
    )

    # Optimization
    optim = optax.adamw(
        learning_rate=cfg.learning_rate,
        b1=cfg.beta1,
        b2=cfg.beta2,
        weight_decay=cfg.weight_decay,
    )

    if cfg.grad_clip > 0:
        optim = optax.chain(optim, optax.clip_by_global_norm(cfg.grad_clip))

    state = optim.init(eqx.filter([model], eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    # 4. Multi-device training
    n_devices = len(jax.local_devices())
    logger.info("Training on %d devices.", n_devices)

    if n_devices > 1 and n_devices % 2 != 0:
        logger.warning(
            "There are %d devices, which is an odd number for multi-GPU training.",
            n_devices,
        )
    # This is kind of nasty. I don't really understand multi-device training on Jax just yet.
    mesh = mesh_utils.create_device_mesh((n_devices,))
    image_sharding = jax.sharding.PositionalSharding(mesh)
    label_sharding = jax.sharding.PositionalSharding(mesh)
    # We replicate() the sharding because we want an exact copy of the model and
    # optimizer state on each device.
    model, state = eqx.filter_shard((model, state), image_sharding.replicate())

    # 5. Logging and checkpointing
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    global_step = 0
    start_time = time.time()

    for epoch in range(cfg.n_epochs):
        for b, (perts, exprs) in enumerate(dataloader):
            key, *subkeys = jax.random.split(key, num=cfg.batch_size + 1)

            perts = eqx.filter_shard(jnp.asarray(perts), image_sharding)
            exprs = eqx.filter_shard(jnp.asarray(exprs), label_sharding)
            model, state, loss = step_model(model, optim, state, perts, exprs)
            global_step += 1

            if global_step % cfg.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                logger.info(
                    "epoch: %d, step: %d, loss: %.5f, step/sec: %.2f",
                    epoch,
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

    # After pre-training on the replogle data, we fine-tune the MLP on vcc train data.
    # TODO: implement. :)
    print("TODO: implement fine-tuning.")

    # After fine-tuning on the vcc train data, we make predictions on the vcc val data.
    val_perts = [dataloader.pert2id[p] for p in perts["val"]]
    assert len(val_perts) <= cfg.batch_size
    preds = jax.vmap(model)(jnp.array(val_perts))

    obs = pd.DataFrame({
        "target_gene": (["non-targeting"] * cfg.controls)
        + [
            g
            for g in adata.obs["target_gene"].unique()
            for _ in range(cfg.cells_per_pert)
        ]
    })


if __name__ == "__main__":
    main(tyro.cli(Config))
