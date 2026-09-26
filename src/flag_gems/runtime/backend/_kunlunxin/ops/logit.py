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
import triton

from flag_gems.ops.logit import _to_triton_dtype, logit_kernel
from flag_gems.runtime import torch_device_fn

from .copy import copy_ as _gems_copy_

logger = logging.getLogger("flag_gems.ops.logit")


def _logit_impl(input: torch.Tensor, eps=None, out: torch.Tensor = None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_floating_point():
        raise TypeError("logit expected a floating point tensor as input")
    if eps is not None:
        eps = float(eps)
        if not (0.0 <= eps <= 0.5):
            raise ValueError("eps must be in the range [0.0, 0.5].")

    in_contig = input.contiguous()
    in_supported = _to_triton_dtype(in_contig.dtype) is not None
    in_kernel = in_contig if in_supported else in_contig.to(torch.float32)

    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise TypeError("For logit.out, out.dtype must match input.dtype")
        out_supported = _to_triton_dtype(out.dtype) is not None
        need_copy_back = (not out.is_contiguous()) or (not out_supported)

        if need_copy_back:
            work_dtype = out.dtype if out_supported else torch.float32
            work_out = torch.empty_like(out, dtype=work_dtype)
        else:
            work_out = out

        n_elements = in_kernel.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        triton_dtype = _to_triton_dtype(work_out.dtype)
        with torch_device_fn.device(input.device):
            logit_kernel[grid](
                in_kernel,
                work_out,
                n_elements,
                eps if eps is not None else 0.0,
                HAS_EPS=(eps is not None),
                BLOCK_SIZE=BLOCK_SIZE,
                OUT_DTYPE=triton_dtype,
            )

        if need_copy_back:
            _gems_copy_(out, work_out)
        return out

    desired_dtype = input.dtype
    desired_supported = _to_triton_dtype(desired_dtype) is not None
    if desired_supported:
        result = torch.empty_like(input, dtype=desired_dtype)
        work_out = result
    else:
        work_out = torch.empty_like(input, dtype=torch.float32)

    n_elements = in_kernel.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    triton_dtype = _to_triton_dtype(work_out.dtype)
    with torch_device_fn.device(input.device):
        logit_kernel[grid](
            in_kernel,
            work_out,
            n_elements,
            eps if eps is not None else 0.0,
            HAS_EPS=(eps is not None),
            BLOCK_SIZE=BLOCK_SIZE,
            OUT_DTYPE=triton_dtype,
        )

    if desired_supported:
        return work_out
    else:
        return work_out.to(desired_dtype)


def logit(input, eps=None):
    logger.debug("GEMS_KUNLUNXIN LOGIT")
    return _logit_impl(input, eps=eps, out=None)


def logit_out(input, eps=None, out=None):
    logger.debug("GEMS_KUNLUNXIN LOGIT_OUT")
    if out is None:
        raise TypeError("logit_out requires an 'out' tensor.")
    return _logit_impl(input, eps=eps, out=out)
