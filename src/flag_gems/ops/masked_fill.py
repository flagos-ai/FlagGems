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
import struct

import torch
import triton
import triton.language as tl

from flag_gems.utils import broadcastable_to, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    inp = tl.where(expand_mask == 1, value, inp)
    return inp


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "NO_OPMATH")])
@triton.jit
def _masked_fill_complex_part(inp, mask, value):
    if inp.dtype == tl.float64:
        value = value.to(tl.int64).to(tl.float64, bitcast=True)
    return tl.where(mask, value, inp)


def _masked_fill_complex(inp, mask, value, *, inplace):
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise RuntimeError("masked_fill only supports a 0-dimensional value tensor")
        value = value.item()
    if mask.dtype != torch.bool:
        raise RuntimeError("masked_fill only supports boolean masks")
    mask = mask.expand(inp.shape)
    scalar = complex(value)
    source = inp
    if inplace:
        if inp.is_conj():
            source = torch.conj(inp)
            scalar = scalar.conjugate()
    else:
        source = inp.resolve_conj()
    parts = torch.view_as_real(source)
    output = parts if inplace else torch.empty_like(parts)
    for lane, value in enumerate((scalar.real, scalar.imag)):
        # Triton scalar arguments otherwise narrow Python floats to fp32.
        if parts.dtype == torch.float64:
            value = struct.unpack("=q", struct.pack("=d", value))[0]
        _masked_fill_complex_part(parts[..., lane], mask, value, out0=output[..., lane])
    return inp if inplace else torch.view_as_complex(output)


def masked_fill(inp, mask, value):
    logger.debug("GEMS MASKED_FILL")
    if inp.is_complex():
        return _masked_fill_complex(inp, mask, value, inplace=False)
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value)


def masked_fill_(inp, mask, value):
    logger.debug("GEMS MASKED_FILL")
    if inp.is_complex():
        return _masked_fill_complex(inp, mask, value, inplace=True)
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp

    expand_mask = mask.expand(inp.shape)
    return masked_fill_kernel(inp, expand_mask, value, out0=inp)
