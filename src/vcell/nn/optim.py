import dataclasses
import typing as tp

import beartype
import optax


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    alg: tp.Literal["sgd", "adam", "adamw"] = "adam"
    """Optimizer algorithm."""
    learning_rate: float = 3e-4
    """Learning rate."""
    grad_clip: float = 1.0
    """Maximum gradient norm."""


@beartype.beartype
def make(cfg: Config):
    if cfg.alg == "sgd":
        optim = optax.sgd(learning_rate=cfg.learning_rate)
    elif cfg.alg == "adam":
        optim = optax.adam(learning_rate=cfg.learning_rate)
    elif cfg.alg == "adamw":
        optim = optax.adamw(learning_rate=cfg.learning_rate)
    else:
        tp.assert_never(cfg.alg)

    if cfg.grad_clip > 0:
        optim = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optim)

    return optim
