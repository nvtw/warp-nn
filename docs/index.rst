|warp-nn|: CUDA Graphable Neural Networks for NVIDIA Warp
=========================================================

|warp-nn| is a Python library for building and training neural networks using |nvidia_warp|.
It enables end-to-end GPU-accelerated neural network implementation for Warp-based applications,
without the need for third-party libraries such as PyTorch or JAX.

.. note::

    |warp-nn| is designed with a focus on clean, simple, and readable neural network-related code.
    Although the library strives to offer high-performance implementations,
    they may not always outperform other highly optimized solutions for NVIDIA GPUs that use
    `cuBLAS <https://developer.nvidia.com/cublas>`_ / `cuDNN <https://developer.nvidia.com/cudnn>`_.

**Main features:**

* Clean, simple, and readable code.
* Configurable CUDA kernels.
* CUDA Graphable implementation.

Quickstart
----------

The easiest way to install |warp-nn| is from `PyPI <https://pypi.org/project/warp-nn>`_.
Refer to the :ref:`installation` section for more details.

.. code-block:: bash

    pip install warp-nn

Sections
--------

.. toctree::

    guide/index

.. toctree::

    api/index

.. toctree::

    examples/index
