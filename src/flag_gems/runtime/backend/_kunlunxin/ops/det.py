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

"""Kunlunxin backend override for ``aten::det``.

The generic ``flag_gems/ops/det.py`` routes to the generic LU in
``flag_gems/ops/linalg_det.py`` (register / blocked / panel kernels). On XPU3
that kernel does not lower at all -- ``n = 1`` already aborts in
``TritonXPULegalize`` with ``out of resource: uni_sram`` (``Required: 0,
Hardware limit: 0``), so shrinking tiles cannot help. Baseline
``tests/test_det.py -m det --ref cpu`` is 36 failed / 2 passed.

The backend already ships a working, XPU3-specific ``linalg_det`` (closed-form
4x4 kernel, single-launch elimination, launch-per-step and grouped paths).
``aten::det`` is numerically the same reduction, so this override only
validates the shape contract and delegates to that implementation instead of
re-deriving a second detector kernel.
"""

import importlib
import logging

# Resolve the *module* rather than `from . import linalg_det`: the package
# namespace already exports a function named `linalg_det`, which shadows the
# submodule attribute. Importing the helper function directly would also make
# it show up in `inspect.getmembers(module, inspect.isfunction)` and be
# re-registered as an extra global by SpecOpRegistrar.
_backend_linalg_det = importlib.import_module(".linalg_det", __package__)

logger = logging.getLogger(__name__)


def det(A):
    logger.debug("GEMS_KUNLUNXIN DET")
    # aten::det is numerically equivalent to aten::linalg_det; both only
    # support float32/float64 (LU decomposition). _linalg_det_impl performs the
    # same dtype/dim/square validation as the generic det path.
    return _backend_linalg_det._linalg_det_impl(A)
