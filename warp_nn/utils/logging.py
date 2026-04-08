# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging


__all__ = ["logger"]


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {
        logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
        logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
        logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
        logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
        logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m",
    }

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)


_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("warp-nn")
logger.setLevel(logging.INFO)
logger.addHandler(_handler)
