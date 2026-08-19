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

"""Hygon (ROCm/DCU) aten::cudnn_convolution fallback registration.

The hygon backend uses a ROCm build of PyTorch that reports ``device_name``
"cuda" but is **not** compiled with cuDNN. Consequently the native
``aten::cudnn_convolution`` op raises ``RuntimeError: cudnn_convolution: ATen
not compiled with cuDNN support`` the moment it is invoked.

This module registers a fallback implementation for ``aten::cudnn_convolution``
that delegates to the native ROCm MIOpen convolution via
``torch.nn.functional.conv{1,2,3}d``. The fallback is registered directly into
the aten dispatcher so it intercepts all call sites (including
``torch.ops.aten.cudnn_convolution.default`` used by benchmarks).

**Why not use FlagGems Triton kernels?**

The Triton convolution kernels, while correct, are 2-5x slower than ROCm's
highly-optimized MIOpen library (the ROCm equivalent of cuDNN). Following the
pattern of the ``_arm`` backend, we **blacklist** ``cudnn_convolution`` from
FlagGems dispatch (via ``op_black_list.yaml``) so the native vendor library
handles all convolution calls. This patch simply ensures the operator doesn't
crash when invoked, allowing tests and benchmarks to run successfully.

The result is:
- Tests pass (operator works via fallback)
- Benchmarks show speedup ≈ 1.0 (native vs native, no FlagGems override)
- No performance regression (optimal vendor path used everywhere)
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Redispatch at CompositeExplicitAutograd to bypass FlagGems Triton dispatch
# and reach the native MIOpen implementation directly.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _miopen_conv(input, weight, padding, stride, dilation, groups):
    """Call aten::convolution via MIOpen, bypassing FlagGems dispatch."""
    ndim = input.dim()
    output_padding = [0] * (ndim - 2)
    return torch.ops.aten.convolution.default.redispatch(
        _FALLBACK_KEYSET,
        input,
        weight,
        None,
        stride,
        padding,
        dilation,
        False,
        output_padding,
        groups,
    )


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """Hygon replacement for aten::cudnn_convolution.

    ROCm lacks cuDNN; routes to MIOpen via aten::convolution redispatch,
    bypassing any FlagGems Triton convolution kernel active under use_gems().
    The cuDNN flags (benchmark/deterministic/allow_tf32) are ignored.
    """
    logger.debug("GEMS_HYGON CUDNN_CONVOLUTION")
    ndim = input.dim()
    if ndim not in (3, 4, 5):
        raise ValueError(
            f"cudnn_convolution supports 1D/2D/3D (ndim 3/4/5), got {ndim}"
        )
    return _miopen_conv(input, weight, padding, stride, dilation, groups)


def _patch_aten_cudnn_convolution():
    """Register MIOpen fallback at the CUDA dispatch key.

    This ensures ALL call sites (including reference computations in tests that
    run outside use_gems()) work on hygon without crashing. Idempotent.
    """
    sentinel = "_flag_gems_hygon_cudnn_conv_patched"
    if getattr(torch, sentinel, False):
        return

    def _impl(inp, wt, pad, stride, dil, groups, bm, det, tf32):
        logger.debug("GEMS_HYGON CUDNN_CONVOLUTION")
        return _miopen_conv(inp, wt, pad, stride, dil, groups)

    torch.library.impl("aten::cudnn_convolution", "CUDA")(_impl)

    setattr(torch, sentinel, True)


# Side-effect: patch the global dispatcher so torch.cudnn_convolution /
# torch.ops.aten.cudnn_convolution.default work outside use_gems() contexts
# (e.g. in test reference computations).
_patch_aten_cudnn_convolution()
