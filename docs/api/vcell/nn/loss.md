Module vcell.nn.loss
====================

Functions
---------

`mmd2_energy_kernel(pred: jaxtyping.Float[Array, 'set genes'], target: jaxtyping.Float[Array, 'set genes'], p: float = 1.0) ‑> jaxtyping.Float[Array, '']`
:   The squared MMD between the predicted and observed cell sets is computed in Eq. (19) of the STATE paper.