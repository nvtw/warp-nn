|warp-nn|: CUDA Graphable Neural Networks for NVIDIA Warp
=========================================================

|warp-nn| is a Warp-native Python library for building and training neural networks for Physical AI workflows.
It is designed for compact neural network components that run directly within Warp-based simulation,
robotics, control, and differentiable computing pipelines.
It is not intended to be a general-purpose replacement for PyTorch, JAX, or other ML frameworks.

.. admonition:: Disclaimer

    |warp-nn| is not part of the ``warp-lang`` package, and it is not maintained by the |nvidia_warp| core team.
    It is a Warp ecosystem library maintained by `@Toni-SM <https://github.com/Toni-SM>`_ /
    `NVIDIA Isaac Sim <https://github.com/isaac-sim>`_. Issues, releases, roadmap,
    and support are managed by the maintainers of this repository.

.. note::

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
