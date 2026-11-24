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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + offset, start + step * offset, mask=offset < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_TSINGMICRO ARANGE_START")
    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)
    size = int(size)

    BLOCK_SIZE = 128
    grid = triton.cdiv(size, BLOCK_SIZE)

    if dtype is None:
        # ATen infers int64 for integral arguments and the default float dtype
        # as soon as one of start/end/step is a float.
        integer_args = all(
            isinstance(v, (int, torch.SymInt)) for v in (start, end, step)
        )
        dtype = torch.int64 if integer_args else torch.get_default_dtype()

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = (
            runtime.device.name
        )  # Note(Zhengzekang): Torch default value is CPU, but triton is target to GPU.

    # txda runs ATen's arange through its CPU fallback (aten::arange.start_out is
    # registered on PrivateUse1 in torch_txda's fallback_cpu.cpp). That kernel
    # computes `start + step * i` in double precision and then rounds once to
    # the output dtype, which an fp32 Triton kernel cannot reproduce bit-exactly
    # (bf16 even diverges by whole ulps). Build floating-point ranges on the
    # host so the output stays bit-identical to ATen for any start/step.
    if dtype.is_floating_point:
        host = torch.arange(
            start,
            end,
            step,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )  # allow-torch-compute
        if torch.device(device).type == "cpu":
            return host
        return host.to(device)

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    arange_func[grid,](result, start, end, step, size, BLOCK_SIZE)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_TSINGMICRO ARANGE")
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
