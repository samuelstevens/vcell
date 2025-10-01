# experiments/13_hvgs.py
"""
Train on VCC's highly variable genes.
"""

import dataclasses
import json
import logging
import os
import pathlib
import pprint
from random import random
import time
import tomllib
import typing as tp

import anndata as ad
import beartype
import chex
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import wandb
from vcell import helpers

@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    # Logging
    wandb_key: str = ""
    """Your W&B API key"""
    wandb_entity: str = "samuelstevens"
    """W&B entity (username or team)"""
    wandb_project: str = "vcell"
    """W&B project name"""

@beartype.beartype
def main(
    cfg: pathlib.Path | None = None,
    override: tp.Annotated[Config, tyro.conf.arg(name="")] = Config(),
):
    """Run the experiment.

    Args:
        cfg: Path to config file.
        override: CLI options to modify the config file.
    """

    if cfg is None:
        # Use override directly as the config
        cfg = override
    else:
        # Load config from file
        if not cfg.exists():
            raise FileNotFoundError(f"Config file not found: {cfg}")

        with open(cfg, "rb") as fd:
            cfg_dict = tomllib.load(fd)

        # Convert TOML dict to Config dataclass
        loaded_cfg = helpers.dict_to_dataclass(cfg_dict, Config)

        # Find non-default override values
        default_cfg = Config()
        non_default_overrides = helpers.get_non_default_values(override, default_cfg)

        # Merge non-default overrides into loaded config
        cfg = helpers.merge_configs(loaded_cfg, non_default_overrides)

    pprint.pprint(dataclasses.asdict(cfg))

    # Train
    wandb.login(key=cfg.wandb_key)
    run = wandb.init(
        entity=cfg.wandb_entity, project=cfg.wandb_project, config=dataclasses.asdict(cfg)
    )
    # log a few values
    for step in range(10):
        wandb.log({
            "loss": random.random(),
            "accuracy": step / 10,
        }, step=step)
        time.sleep(0.5)

if __name__ == "__main__":
    tyro.cli(main)
