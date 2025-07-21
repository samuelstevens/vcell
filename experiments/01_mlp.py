# experiments/01_mlp.py

import dataclasses
import json
import logging
import os
import time

import beartype
import chex
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import optax
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("01")


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

    # Optimization
    learning_rate: float = 0.001
    """Peak learning rate."""
    beta1: float = 0.9
    """Adam beta1."""
    beta2: float = 0.999
    """Adam beta2."""
    grad_clip: float = 1.0
    """Maximum gradient norm. `0` implies no clipping."""
    grad_accum: int = 1
    """Number of steps to accumulate gradients for. `1` implies no accumulation."""
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


@beartype.beartype
def save(filename: str, cfg: Config, model):
    with open(filename, "wb") as fd:
        cfg_str = json.dumps(cfg)
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@jaxtyped(typechecker=beartype.beartype)
def compute_loss(
    model: eqx.Module,
    images: Float[Array, "batch 3 width height"],
    labels: Float[Array, "batch n_class"],
    *,
    keys: list[chex.PRNGKey],
):
    logits = jax.vmap(model, in_axes=(0, None, 0))(images, False, jnp.array(keys))
    loss = optax.safe_softmax_cross_entropy(logits, labels)

    return jnp.mean(loss)


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    images: Float[Array, "batch 3 width height"],
    labels: Float[Array, "batch n_class"],
    *,
    keys: list[chex.PRNGKey],
):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, images, labels, keys=keys
    )
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def evaluate(model: eqx.Module, dataloader, key: chex.PRNGKey) -> dict[str, object]:
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    @eqx.filter_jit(donate="all-except-first")
    def _compute_loss(
        model: eqx.Module,
        images: Float[Array, "b 3 w h"],
        keys,
        labels: Int[Array, " b"],
    ) -> tuple[Float[Array, ""], Float[Array, "b n_classes"]]:
        logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, jnp.array(subkeys))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    metrics = {"loss": []}

    for batch in dataloader:
        images = jnp.asarray(batch["image"])
        labels = jnp.asarray(batch["label"])
        key, *subkeys = jax.random.split(key, num=len(labels) + 1)
        loss, logits = _compute_loss(model, images, jnp.array(subkeys), labels)
        metrics["loss"].append(loss)

        _, indices = jax.lax.top_k(logits, k=5)

        for k in (1, 5):
            _, indices = jax.lax.top_k(logits, k=k)
            n_correct = jnp.any(indices == labels[:, None], axis=1).sum()

            name = f"acc{k}"
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(n_correct / len(labels))
    metrics = {key: jnp.mean(jnp.array(value)).item() for key, value in metrics.items()}
    return metrics


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    key, model_key = jax.random.split(key)

    # 1. Model
    model_cfg = dict(n_perts=cfg.n_perts, n_genes=cfg.n_genes, hidden_d=cfg.hidden_d)
    model = Model(**model_cfg, key=model_key)
    logger.info("Initialized model.")
    breakpoint()

    # 2. Dataset

    # 3. Train
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
    # Image batches have four dimensions: batch x channels x width x height. We want to
    # split the batch dimension up over all devices. The same applies to labels, but
    # they only have batch x classes
    image_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices, 1, 1, 1))
    )
    label_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices, 1))
    )
    # We replicate() the sharding because we want an exact copy of the model and
    # optimizer state on each device.
    model, state = eqx.filter_shard((model, state), image_sharding.replicate())

    # 5. Logging and checkpointing
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    flops_per_iter = 0
    flops_promised = 38.7e12  # 38.7 TFLOPS for fp16 on A6000
    global_step = 0
    start_time = time.time()

    t1 = time.time()
    for epoch in range(cfg.n_epochs):
        for b, batch in enumerate(train_dataloader):
            t0 = t1
            t1 = time.time()
            key, *subkeys = jax.random.split(key, num=cfg.batch_size + 1)

            images = eqx.filter_shard(jnp.asarray(batch["image"]), image_sharding)
            labels = eqx.filter_shard(jnp.asarray(batch["label"]), label_sharding)

            model, state, loss = step_model(
                model, optim, state, images, labels, keys=subkeys
            )
            global_step += 1

            if global_step % cfg.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                dt = t1 - t0
                metrics = {
                    "train/loss": loss.item(),
                    "perf/step_per_sec": step_per_sec,
                    "perf/mfu": flops_per_iter / dt / flops_promised,
                }
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

            if global_step == 10:
                # Calculate flops one time after a couple iterations.
                logger.info("Calculating FLOPs per forward/backward pass.")
                flops_per_iter = (
                    eqx.filter_jit(step_model)
                    .lower(model, optim, state, images, labels, keys=subkeys)
                    .compile()
                    .compiled.cost_analysis()[0]["flops"]
                )
                logger.info("Calculated FLOPs: %d.", flops_per_iter)

        # 4. Evaluate
        # We want to evaluate on the rest of the training set (minival) as well as (1) the true validation set (2) imagenet v2 and (3) imagenet real. Luckily this is the same 1K classes so we can simply do inference without any fitting.
        for name, dataloader in val_dataloaders.items():
            key, subkey = jax.random.split(key)
            logger.info("Evaluating %s.", name)
            metrics = evaluate(model, dataloader, subkey)
            metrics = {f"{name}/{key}": value for key, value in metrics.items()}
            run.log(metrics, step=global_step)
            logger.info(
                ", ".join(f"{key}: {value:.3f}" for key, value in metrics.items()),
            )
        # Record epoch at this step only once.
        run.log({"epoch": epoch}, step=global_step)

        # Checkpoint.
        save(os.path.join(args.ckpt_dir, f"{run.id}_ep{epoch}.eqx"), model_cfg, model)


if __name__ == "__main__":
    main(tyro.cli(Config))
