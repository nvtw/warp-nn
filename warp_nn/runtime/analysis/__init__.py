# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline, dependency-free tools for understanding model checkpoints."""


def analyze_model(*args, **kwargs):
    """Lazily inspect a model checkpoint without loading tensor contents."""
    from .model_graph import analyze_model as analyze

    return analyze(*args, **kwargs)


def write_model_graph(*args, **kwargs):
    """Lazily write a standalone model-map HTML report."""
    from .model_graph import write_model_graph as write

    return write(*args, **kwargs)


__all__ = ["analyze_model", "write_model_graph"]
