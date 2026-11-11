# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tsingmicro common fused ops entry.

All fused operators live in the fully isolated per-chip packages ``tx81/``
and ``tx82/`` (ops, fused, utils, tune_configs.yaml,
heuristics_config_utils.py). This module simply dispatches to the package
matching the detected chip and re-exports its fused operators, so the
framework observes the exact same set of fused ops as before.
"""

import importlib
import inspect
import os
import sys

from .. import ARCH_MAP, arch

_arch_name = ARCH_MAP[arch]
_vendor_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _vendor_dir)
try:
    _arch_fused = importlib.import_module(f"{_arch_name}.fused")
finally:
    sys.path.remove(_vendor_dir)

for _name, _fn in inspect.getmembers(_arch_fused, inspect.isfunction):
    globals()[_name] = _fn

__all__ = list(getattr(_arch_fused, "__all__", []))
