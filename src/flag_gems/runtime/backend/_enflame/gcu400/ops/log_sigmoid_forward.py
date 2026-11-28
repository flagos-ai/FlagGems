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

# GCU caps grid.x at 65535 (CUDA allows 2^31-1). The original KernelGen body
# launched one 1024-wide block per program, i.e. grid.x = cdiv(n, 1024) =
# 524288 for the 512M-element benchmark shapes, which aborted with
# triton OutOfResources. Worse, this kernel is the de-facto native
# implementation of aten::log_sigmoid_forward on GCU: the PrivateUse1 rescue
# in log_sigmoid_backward.py routes every torch F.logsigmoid /
# torch.ops.aten.log_sigmoid_forward call here (P21: the torch_gcu native
# kernel leaves the backward buffer uninitialized). Tiny 1024-wide CTAs also
# schedule pathologically on GCU400 (0.34 TB/s at 512M fp16), so the launch
# follows the house flat grid-stride recipe (relu.py / sigmoid.py / abs.py):
# few fat CTAs (grid <= 48 SPC), a numel-based BLOCK/num_warps ladder, EVEN
# mask elision, and an int64-offset WIDE arm that keeps the flat addressing
# correct past the int32 domain (randn_like precedent).
_NUM_SPC = 48
_INT32_MAX = torch.iinfo(torch.int32).max
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def log_sigmoid_forward_kernel(
    x_ptr,
    buffer_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    SOFTPLUS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    for block_id in tl.range(pid, num_blocks, num_pids):
        # WIDE: past the int32 flat-offset domain (n >= 2^31), compute the
        # offsets in int64 so the stride loop cannot silently wrap around.
        if WIDE:
            offsets = block_id.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        else:
            offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if EVEN:
            x = tl.load(x_ptr + offsets)
        else:
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # log_sigmoid(x) = -log(1 + exp(-x)) = min(x, 0) - log(1 + exp(-|x|)).
        # Form choice, measured on GCU400 (512M, paired interleaved do_bench,
        # 2026-11-17): for 16-bit dtypes the plain softplus chain is 25%
        # faster than the where-guarded variant (guarding with tl.where
        # breaks the `log(1+exp(-x))` fusion into a HW softplus op: 1132 vs
        # 1518us) and 16% faster than the min/abs chain (1132 vs 1353us; the
        # halved traffic exposes the abs/min ALU ops). The unguarded form is
        # still domain-safe on GCU: the fused softplus saturates exactly for
        # x in [-1e4, -87] (probe-verified for all three dtypes, output == x
        # bit-exact for fp32), and the known-values test pins x = -100
        # permanently. For fp32 the 3-stream traffic already hides the
        # abs/min chain, where sharing the buffer's exp(-|x|) with the log
        # term is fastest instead (2039 vs 2076us at 512M). tl.log is the
        # cheapest log form: tl.math.log2*ln2 +45%, deg-6/5 Estrin or Horner
        # log1p polynomials +88-92% (GCU taxes every extra ALU op; a deg-3
        # poly is 8% faster but its 9.1e-4 max error breaks the fp16 gate at
        # x=0).
        #
        # Program order matters on GCU (-8.5% at 512M fp16 if violated): the
        # fusion region groups ops by source order, and the softplus chain
        # must come FIRST so the HW softplus fuses against the load; only
        # afterwards the abs/exp buffer chain. The min/abs branch consumes
        # buffer_val, so its forced order is the exception.
        x_f32 = x.to(tl.float32)
        if SOFTPLUS:
            log_sigmoid_val = -tl.log(1.0 + tl.exp(-x_f32))
            buffer_val = tl.exp(-tl.abs(x_f32))
        else:
            buffer_val = tl.exp(-tl.abs(x_f32))
            log_sigmoid_val = tl.minimum(x_f32, 0.0) - tl.log(1.0 + buffer_val)

        if EVEN:
            tl.store(output_ptr + offsets, log_sigmoid_val.to(x.dtype))
            tl.store(buffer_ptr + offsets, buffer_val.to(x.dtype))
        else:
            tl.store(output_ptr + offsets, log_sigmoid_val.to(x.dtype), mask=mask)
            tl.store(buffer_ptr + offsets, buffer_val.to(x.dtype), mask=mask)


def _forward_config(n_elements, itemsize):
    # (BLOCK_SIZE, num_warps) ladder for the 3-stream flat kernel. Locked by
    # interleaved do_bench(median) sweeps on GCU400 (2026-11-17, min of 3-4
    # paired rounds, log-chain-first order): 16-bit big wants 131072/nw8 and
    # mid 32768/nw4; fp32 measured (32768, nw2) best on BOTH the 16.7M mid
    # band (71.1us) and the 512M big band (2038.9us), so one entry covers the
    # whole >4M range. Each (dtype, BLOCK) pair maps to exactly one num_warps:
    # LibEntry's dispatch key omits num_warps, so a shared BLOCK must never
    # launch with two different warp counts. Grid stays capped at
    # _NUM_SPC=48: 24/96/192 measured worse on every band.
    if n_elements <= 2048:
        return max(256, triton.next_power_of_2(n_elements)), 2
    if n_elements <= 65536:
        return 2048, 2
    if n_elements <= 4194304:
        return 8192, 2
    if itemsize == 2:
        if n_elements <= 67108864:
            return 32768, 4
        return 131072, 8
    return 32768, 2


def log_sigmoid_forward(A):
    logger.debug("GEMS_ENFLAME LOG_SIGMOID_FORWARD")
    if not (A.is_contiguous() and A.dtype in _FLAT_DTYPES):
        # Non-contiguous / non-float inputs. The generic public
        # log_sigmoid_forward is the same flat KernelGen shape as the pre-fix
        # code (reshape(-1) of a preserve-format output detaches the kernel's
        # stores, and n > 64M still aborts on the direct grid), so compose the
        # stride-correct pointwise FlagGems ops instead: other FlagGems ops are
        # the only allowed host-side compute. exp(neg(abs(A))) reproduces the
        # aten backward-buffer contract exp(-|A|); log_sigmoid(A) is the value
        # the output carries. Lazy imports dodge the ops __init__ import cycle.
        from flag_gems import abs as _fg_abs
        from flag_gems import exp as _fg_exp
        from flag_gems import log_sigmoid as _fg_log_sigmoid
        from flag_gems import neg as _fg_neg

        output = _fg_log_sigmoid(A)
        buffer = _fg_exp(_fg_neg(_fg_abs(A)))
        return output, buffer

    output = torch.empty_like(A)
    buffer = torch.empty_like(A)

    n_elements = A.numel()
    if n_elements == 0:
        return output, buffer

    block_size, num_warps = _forward_config(n_elements, A.element_size())
    # Round the block count up first: if that would leave the int32 safe
    # domain, dispatch the int64-offset (WIDE) arm of the kernel.
    wide = triton.cdiv(n_elements, block_size) * block_size > _INT32_MAX
    grid = (min(triton.cdiv(n_elements, block_size), _NUM_SPC),)

    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (abs.py / neg.py /
    # square.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                log_sigmoid_forward_kernel[grid](
                    A,
                    buffer,
                    output,
                    n_elements,
                    block_size,
                    EVEN=(n_elements % block_size == 0),
                    WIDE=wide,
                    SOFTPLUS=(A.element_size() < 4),
                    num_warps=num_warps,
                )
            return output, buffer
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "log_sigmoid_forward kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
