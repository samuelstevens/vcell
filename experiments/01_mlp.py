# experiments/01_mlp.py
import collections.abc
import dataclasses
import json
import logging
import os
import pathlib
import time

import anndata as ad
import beartype
import chex
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

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
    hidden_d: int = 1024 * 4
    """Hidden dimension."""

    # Data
    replogle_essential: pathlib.Path = pathlib.Path(
        "data/inputs/replogle/ReplogleWeissman2022_K562_essential.h5ad"
    )
    replogle_gwps: pathlib.Path = pathlib.Path(
        "data/inputs/replogle/ReplogleWeissman2022_K562_gwps.h5ad"
    )
    gene_list: pathlib.Path = pathlib.Path("data/inputs/challenge/gene_names.csv")

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
    n_epochs: int = 90
    """Number of epochs to train for."""

    # Logging
    log_every: int = 10
    """how often to log metrics."""
    track: bool = True
    """whether to track with Aim."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store model checkpoints."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """any tags for this specific run."""


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    perts: eqx.nn.Embedding
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, n_perts: int, n_genes: int, hidden_d: int, *, key: chex.PRNGKey):
        key0, key1, key2 = jax.random.split(key, 3)
        self.perts = eqx.nn.Embedding(
            num_embeddings=n_perts + 1, embedding_size=n_genes, key=key0
        )
        self.linear1 = eqx.nn.Linear(n_genes, hidden_d, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_d, n_genes, key=key2)

    def __call__(
        self, pert: Int[Array, ""], *, key: chex.PRNGKey | None = None
    ) -> Float[Array, " n_genes"]:
        x = self.perts(pert)
        # Feed embedding through the MLP.
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.linear2)(x)

        return x


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
    state: optax.OptState,
    pert: Int[Array, " batch"],
    expr: Float[Array, "batch n_genes"],
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, pert, expr)
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@beartype.beartype
def stream_batches(
    file_path: pathlib.Path,
    gene_list: list[str],
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> collections.abc.Iterable[dict[str, Array]]:
    """Yield dicts with 'pert' (int id) and 'expr' (float32[G])"""
    adata = ad.read_h5ad(file_path, backed="r")
    # map gene order once
    gene_idx = np.where(np.isin(adata.var_names, gene_list))[0]

    # map perturb labels to int ids
    perts = np.asarray(adata.obs["target_gene"], dtype="U")
    pert2id = {g: i for i, g in enumerate(np.unique(perts), start=1)}
    pert_ids = np.vectorize(pert2id.get)(perts).astype(np.int32)  # 0 is unused

    idxs = np.arange(adata.n_obs)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)

    for start in range(0, adata.n_obs, batch_size):
        sel = idxs[start : start + batch_size]
        x = adata.X[sel, :][:, gene_idx]  # sparse sub-matrix
        if sp.issparse(x):
            x = x.A  # csr -> dense
        yield {
            "pert": jnp.array(pert_ids[sel]),
            "expr": jnp.log1p(jnp.array(x, dtype=jnp.float32)),
        }


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    key, model_key = jax.random.split(key)

    # Model
    model_cfg = dict(n_perts=cfg.n_perts, n_genes=cfg.n_genes, hidden_d=cfg.hidden_d)
    model = Model(**model_cfg, key=model_key)
    logger.info("Initialized model.")

    # Data
    gene_list = cfg.gene_list.read_text().strip().split("\n")

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
        for b, batch in enumerate(
            stream_batches(
                cfg.replogle_essential,
                gene_list,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            )
        ):
            key, *subkeys = jax.random.split(key, num=cfg.batch_size + 1)

            images = eqx.filter_shard(jnp.asarray(batch["image"]), image_sharding)
            labels = eqx.filter_shard(jnp.asarray(batch["label"]), label_sharding)

            model, state, loss = step_model(
                model, optim, state, images, labels, keys=subkeys
            )
            global_step += 1

            if global_step % cfg.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )


if __name__ == "__main__":
    main(tyro.cli(Config))
