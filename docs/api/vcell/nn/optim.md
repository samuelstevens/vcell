Module vcell.nn.optim
=====================

Functions
---------

`make(cfg: vcell.nn.optim.Config)`
:   

Classes
-------

`Config(alg: Literal['sgd', 'adam', 'adamw'] = 'adam', learning_rate: float = 0.0003, grad_clip: float = 1.0)`
:   Config(alg: Literal['sgd', 'adam', 'adamw'] = 'adam', learning_rate: float = 0.0003, grad_clip: float = 1.0)

    ### Instance variables

    `alg: Literal['sgd', 'adam', 'adamw']`
    :   Optimizer algorithm.

    `grad_clip: float`
    :   Maximum gradient norm.

    `learning_rate: float`
    :   Learning rate.