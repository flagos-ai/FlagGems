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

import logging

import torch

logger = logging.getLogger(__name__)

# Redispatch key used to reach the vendor (torch_xmlir) `_linalg_eigvals`
# kernel.  The vendor registers its LAPACK-backed eigenvalue implementation
# under the XPU dispatch key, while the gems override that routed here occupies
# the CUDA key.  Redispatching with only the XPU key executes the vendor kernel
# directly on the Kunlunxin device: no CPU fallback and no recursion into this
# override (evidence: key-by-key redispatch enumeration — XPU returns a device
# tensor matching the CPU reference; CompositeExplicitAutograd /
# CompositeImplicitAutograd / AutogradCUDA raise NotImplementedError, and the
# CUDA key re-enters the gems override).
#
# Why not the generic implementation (`flag_gems/ops/_linalg_eigvals.py`):
#   * it routes through `torch.linalg.eigvals(inp.cpu()).to(inp.device)` — a
#     CPU ATen fallback round-trip that is forbidden as a production path for
#     this backend (n=300 fp32: ~2.07 s via the CPU round-trip vs ~0.05 s via
#     the vendor XPU key);
#   * its Triton "proxy" copy kernel cannot even handle complex inputs on this
#     backend (XPU pointwise codegen does not support complex pointers —
#     complex64 input raises `KeyError: 'complex64'`), while the vendor XPU
#     kernel handles float32/float64/complex64/complex128, batched inputs,
#     non-contiguous views and empty matrices correctly.
_VENDOR_XPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.XPU)


def _linalg_eigvals(inp):
    logger.debug("GEMS_KUNLUNXIN _LINALG_EIGVALS")
    return torch.ops.aten._linalg_eigvals.default.redispatch(
        _VENDOR_XPU_KEYSET,
        inp,
    )
