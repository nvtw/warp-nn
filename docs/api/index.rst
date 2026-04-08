API Reference
=============

.. toctree::
    :hidden:

    activations
    initializers
    layers
    optimizers
    utils

This section contains the API reference for the |warp-nn| library.

Core modules
------------

.. currentmodule:: warp_nn.modules
.. autosummary::
    :nosignatures:

    ~module.Module
    ~parameter.Parameter

Activations
-----------

.. currentmodule:: warp_nn.modules.activations
.. autosummary::
    :nosignatures:

    ELU
    LeakyReLU
    ReLU
    SELU
    Sigmoid
    SoftPlus
    SoftSign
    Tanh

Initializers
------------

.. currentmodule:: warp_nn.initializers
.. autosummary::
    :nosignatures:

    constant
    kaiming_normal
    kaiming_uniform
    ones
    zeros

Layers
------

.. currentmodule:: warp_nn.modules.layers
.. autosummary::
    :nosignatures:

    Conv1D
    Conv2D
    GRUCell
    Linear
    LSTMCell
    RNNCell
    Sequential

Optimizers
----------

.. currentmodule:: warp_nn.optimizers
.. autosummary::
    :nosignatures:

    Adam
    SGD
