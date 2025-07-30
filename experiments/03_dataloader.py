# experiments/03_dataloader.py
"""
A small experiment to try and get a better dataloader.

Dataloading:

These raw counts are depth-normalized and log-transformed using Scanpy(normalize_total -> log1p)
"""

import dataclasses
import json
import logging
import os
import pathlib
import typing as tp

import beartype
import chex
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import optax
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import vcell.data

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("03")


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
    data: vcell.data.PerturbationConfig = vcell.data.PerturbationConfig(
        h5ad_fpath=pathlib.Path("data/inputs/vcc/adata_Training.h5ad"),
        set_size=16,
        pert_col="target_gene",
        cell_line_col="guide_id",
    )
    vcc: pathlib.Path = pathlib.Path("data/inputs/vcc")

    # Optimization
    learning_rate: float = 0.001
    """Peak learning rate."""
    batch_size: int = 2
    """Batch size."""
    n_train: int = 1_000
    """Number of steps to train for."""
    n_val: int = 1_000
    """Number of steps to train for."""

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
            num_embeddings=n_perts + 1, embedding_size=n_genes, key=key0
        )
        self.linear1 = eqx.nn.Linear(n_genes, d_hidden, key=key1)
        self.linear2 = eqx.nn.Linear(d_hidden, n_genes, key=key2)

    def __call__(
        self,
        ctrls: Float[Array, "set_size n_genes"],
        pert: Int[Array, ""],
        *,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, " set_size n_genes"]:
        x_g = self.perts(pert)
        x_sg = jnp.expand_dims(x_g, axis=0)
        x_sg = x_sg + ctrls
        # breakpoint()

        x_sh = jax.vmap(self.linear1)(x_sg)
        x_sh = jax.nn.gelu(x_sh)
        x_sg = jax.vmap(self.linear2)(x_sh)

        return x_sg


class Batcher:
    def __init__(self, dataloader, batch_size: int):
        self.dataloader = dataloader
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = []
            it = iter(self.dataloader)
            while len(batch) < self.batch_size:
                batch.append(next(it))

            # Transform list of dicts into dict of arrays
            batch_dict = {}
            for key in batch[0].keys():
                batch_dict[key] = jnp.array([item[key] for item in batch])

            yield batch_dict


@beartype.beartype
def save(filename: str, cfg: Config, model):
    with open(filename, "wb") as fd:
        cfg_str = json.dumps(cfg)
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@jaxtyped(typechecker=beartype.beartype)
def compute_loss(
    model: eqx.Module,
    perts: Int[Array, " batch"],
    ctrls: Float[Array, "batch set_size n_genes"],
    tgts: Float[Array, "batch set_size n_genes"],
) -> Float[Array, ""]:
    logits = jax.vmap(model)(ctrls, perts)
    loss = jnp.mean((logits - tgts) ** 2)

    return loss


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    perts: Int[Array, " batch"],
    ctrls: Float[Array, "batch set_size n_genes"],
    tgts: Float[Array, "batch set_size n_genes"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, perts, ctrls, tgts)
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


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
    dataloader = Batcher(vcell.data.PerturbationDataloader(cfg.data), cfg.batch_size)
    logger.info("Initialized dataset.")

    optim = optax.sgd(learning_rate=cfg.learning_rate, nesterov=True)
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

    for b, batch in enumerate(dataloader):
        perts = eqx.filter_shard(jnp.asarray(batch["pert"]), image_sharding)
        ctrls = eqx.filter_shard(jnp.asarray(batch["control"]), label_sharding)
        tgts = eqx.filter_shard(jnp.asarray(batch["target"]), label_sharding)

        model, state, loss = step_model(model, optim, state, perts, ctrls, tgts)
        global_step += 1

        if global_step % cfg.log_every == 0:
            logger.info("step: %d, loss: %.5f", global_step, loss.item())

        if b > cfg.n_train:
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
