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

"""GCU300 specialization of ``aten::_stack``.

Shares the int32-offset copy kernels of this backend's ``stack`` module; the
generic kernel widens the flat index to int64, which this target miscompiles
once a block covers 256 or more lanes (see ``stack.py``).
"""

import logging
from typing import List, Tuple, Union

import torch

from .stack import _launch

logger = logging.getLogger(__name__)


def _stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME _STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    # Validate ``dim`` against tensors[0] unconditionally: with a single input
    # the per-entry loop below never runs and an out-of-range dim would be
    # silently accepted instead of matching ATen.
    ndim0 = tensors[0].dim()
    if (dim < -ndim0 - 1) or (dim > ndim0):
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -ndim0 - 1, ndim0, dim
            )
        )
    for i, s in enumerate(inp_shapes[1:]):
        ndim = tensors[i + 1].dim()
        if (dim < -ndim - 1) or (dim > ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -ndim - 1, ndim, dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                "stack expects each tensor to be equal size, but got "
                f"{inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1

    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for dt in dtypes[1:]:
        dtype = torch.promote_types(dtype, dt)

    # Triton has no complex scalar types: stack the real view and reinterpret.
    is_complex = dtype.is_complex
    if is_complex:
        complex_dtype = dtype
        dtype = torch.view_as_real(torch.empty((), dtype=dtype, device="cpu")).dtype
        tensors = [torch.view_as_real(t.to(complex_dtype)) for t in tensors]
    tensors = [t.to(dtype) if t.dtype != dtype else t for t in tensors]
    device = tensors[0].device

    work_shape = list(tensors[0].shape)
    out_shape = work_shape[:dim] + [len(tensors)] + work_shape[dim:]
    contig_strides = []
    stride = 1
    for dim_sz in reversed(out_shape):
        contig_strides.append(stride)
        stride *= dim_sz
    contig_strides.reverse()
    out = torch.empty_strided(out_shape, contig_strides, dtype=dtype, device=device)

    dim_prod_post = 1
    for s in work_shape[dim:]:
        dim_prod_post *= s

    if out.numel() == 0:
        return torch.view_as_complex(out) if is_complex else out

    _launch(out, tensors, dim, dim_prod_post)

    if is_complex:
        out = torch.view_as_complex(out)
    return out
