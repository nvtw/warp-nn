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

## Local coding agents

For Qwen with DFlash, a browser chat UI and local/LAN coding-client connections, run
`examples/qwen_dflash_server.py`; see the [server instructions](examples/openai_server.md).

For local Qwen/Muse chat with sandboxed file and script tools, see the
[coding-agent example](docs/coding-agent.md). The Qwen repetition diagnosis and
validation are documented in [the investigation notes](docs/qwen-loop-investigation.md).

## FLUX.2 Klein image generation

The [FLUX.2 Klein example](examples/flux2_klein.py) runs the Apache-2.0 4B
model with native Warp inference and four denoising steps. Download its
self-contained Diffusers bundle without the duplicate transformer checkpoint:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 \
  hf download black-forest-labs/FLUX.2-klein-4B \
  --exclude 'flux-2-klein-4b.safetensors' --exclude '*.jpg' \
  --local-dir ~/Models/warp-nn/black-forest-labs/FLUX.2-klein-4B
.venv/bin/python examples/flux2_klein.py \
  ~/Models/warp-nn/black-forest-labs/FLUX.2-klein-4B \
  --prompt "A red fox in a snowy forest, photographic" \
  --output fox.png
```

The example uses the bundled Qwen3 text encoder, FLUX transformer, and VAE;
PyTorch, Diffusers, and Transformers are not inference dependencies.

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
