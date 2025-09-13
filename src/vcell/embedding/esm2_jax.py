import dataclasses
import functools
import json
import math
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree, jaxtyped
from equinox.nn import LayerNorm as ESM1bLayerNorm

from vcell import helpers

@beartype.beartype
def gelu(x):
    pass

@beartype.beartype
def symmetrize(x):
    pass

@beartype.beartype
def apc(x):
    pass

@jaxtyped(typechecker=beartype.beartype)
class ESM1LayerNorm(eqx.Module):
    hidden_size: tuple
    eps: float = 1e-5
    affine: bool = True
    weight: eqx.nn.Parameter | None = None
    bias: eqx.nn.Parameter | None = None

    def __init__(self, hidden_size: int, eps: float = 1e-5, affine=True):
        """Construct layer norm in TF style (epsilon inside sqrt)."""
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)

        self.weight = eqx.nn.Parameter(jnp.ones(self.hidden_size)) if self.affine else None
        self.bias = eqx.nn.Parameter(jnp.zeros(self.hidden_size)) if self.affine else None

    def __call__(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = jnp.mean(x, axis=dims, keepdims=True)
        x_zeromean = x - means
        variances = jnp.mean(x_zeromean**2, axis=dims, keepdims=True)
        x = x_zeromean / jnp.sqrt(variances + self.eps)
        if self.affine:
            x = (x * self.weight) + self.bias
        return x

@jaxtyped(typechecker=beartype.beartype)
class TransformerLayer(eqx.Module):
    def __init__(self):
        pass

    def _init_submodules(cls):
        pass

    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class AxialTransformerLayer(eqx.Module):
    def __init__(self):
        pass

    def build_residual(self):  
        pass
    
    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class LearnedPositionalEmbedding(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class SinusoidalPositionalEmbedding(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, x):
        pass

    def make_positions(self, x):
        pass

    def get_embedding(self, num_embeddings):
        pass

@jaxtyped(typechecker=beartype.beartype)
class RobertaLMHead(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class ContactPredictionHead(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class NormalizedResidualBlock(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass

@jaxtyped(typechecker=beartype.beartype)
class FeedForwardNetwork(eqx.Module):
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass
