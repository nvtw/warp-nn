# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility aliases for the shared autoregressive batching adapter."""

from warp_nn.runtime.services.autoregressive_batching import (
    AutoregressiveBatchExecutor as QwenBatchExecutor,
    AutoregressiveBatchPayload as QwenBatchPayload,
)

__all__ = ["QwenBatchExecutor", "QwenBatchPayload"]
