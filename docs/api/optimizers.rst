Optimizers
==========

.. toctree::
    :hidden:

    Adam <optimizers/adam>
    SGD <optimizers/sgd>

Optimizers are algorithms that update the parameters of a model.

The following table lists the available optimizers:

.. currentmodule:: warp_nn.optimizers
.. autosummary::
    :nosignatures:

    Adam
    SGD

Base class
----------

.. note::

    This is the base class for all the optimizers.
    **It is not intended to be used directly**.

API
^^^

.. autoclass:: warp_nn.optimizers.Optimizer
    :show-inheritance:
    :inherited-members:
    :members:
