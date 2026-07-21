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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _functional_sym_constrain_range_for_size_kernel(
    x_ptr,
    y_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + off, x, mask=mask)


def _functional_sym_constrain_range_for_size(*args, **kwargs):
    logger.debug("GEMS_ENFLAME FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE")
    tensor_arg = None
    for a in args:
        if isinstance(a, torch.Tensor):
            tensor_arg = a
            break
    if tensor_arg is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor_arg = v
                break

    if tensor_arg is not None:
        if tensor_arg.is_contiguous():
            N_total = tensor_arg.numel()
            if N_total > 0:
                BLOCK = 8192
                grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
                with torch_device_fn.device(tensor_arg.device):
                    _functional_sym_constrain_range_for_size_kernel[(grid_size,)](
                        tensor_arg, tensor_arg, N_total, BLOCK=BLOCK, num_warps=4
                    )
        return tensor_arg

    return args[0] if len(args) > 0 else None
