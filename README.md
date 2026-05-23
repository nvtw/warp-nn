[![PyPI version](https://badge.fury.io/py/warp-nn.svg)](https://badge.fury.io/py/warp-nn)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Warp-NN: CUDA Graphable Neural Networks for NVIDIA Warp

**[Documentation](https://nvidia.github.io/warp-nn/latest)** | [Changelog](https://github.com/NVIDIA/warp-nn/blob/main/CHANGELOG.md)

Warp-NN is a Warp-native Python library for building and training neural networks for Physical AI workflows.
It is designed for compact neural network components that run directly within Warp-based simulation,
robotics, control, and differentiable computing pipelines.
It is not intended to be a general-purpose replacement for PyTorch, JAX, or other ML frameworks.

> **Disclaimer:**
> Warp-NN is not part of the `warp-lang` package, and it is not maintained by the [NVIDIA Warp](https://nvidia.github.io/warp) core team.
> It is a Warp ecosystem library maintained by [@Toni-SM](https://github.com/Toni-SM) / [NVIDIA Isaac Sim](https://github.com/isaac-sim).
> Issues, releases, roadmap, and support are managed by the maintainers of this repository.

## Installation

The easiest way to install Warp-NN is from [PyPI](https://pypi.org/project/warp-nn).
Refer to the *Installation* section in docs for more details.

```bash
pip install warp-nn
```

## Support

Questions and discussions can be opened on [GitHub Discussions](https://github.com/NVIDIA/warp-nn/discussions).

Problems, issues, and feature requests can be opened on [GitHub Issues](https://github.com/NVIDIA/warp-nn/issues).

## Contributing

Contributions and pull requests from the community are welcome.
Please see the [Contribution Guide](https://github.com/NVIDIA/warp-nn/blob/main/CONTRIBUTING.md)
for more information on contributing to the development of Warp-NN.

## License

Warp-NN is provided under the Apache License, Version 2.0.
Please see [LICENSE.md](https://github.com/NVIDIA/warp-nn/blob/main/LICENSE.md) for full license text.

This project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use.
