# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from . import utilities


def test_check_arrays_applies_relative_tolerance():
    reference = torch.tensor([100.0])
    utilities.check_arrays(reference, torch.tensor([100.5]), rtol=0.01, atol=0.0)
    with pytest.raises(AssertionError, match="all-close"):
        utilities.check_arrays(reference, torch.tensor([102.0]), rtol=0.01, atol=0.0)
