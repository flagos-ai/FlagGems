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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _copy_kernel(in_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(in_ptr + off, mask=mask)
        tl.store(out_ptr + off, x, mask=mask)


# GCU400 retune (2026-11-16, abs/rsqrt 1R1W family recipe, interleaved
# do_bench(median) A/B; see _functional_sym_constrain_range.py precedent):
# the fixed single-tier B8192/nw4 launch left the 16-bit big bucket 1.36x
# off the ~3.18TB/s DRAM copy floor and the mid bucket on undersized
# blocks. Reuse the abs-family size/itemsize dispatch: tiny wants the exact
# next_pow2 tile, 16-bit big wants B131072/nw2, fp32 stays on B65536/nw8.
def _choose_launch(n_total, itemsize):
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 64), 4
    if n_total <= 65536:
        return 8192, 8
    if n_total <= 524288:
        return 32768, 8
    if itemsize == 4:
        return 65536, 8
    if n_total > (1 << 28):
        return 131072, 2
    return 65536, 2


def _launch_copy(src, dst):
    n_total = src.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, src.element_size())
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # neg.py / rad2deg.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                _copy_kernel[(grid_size,)](
                    src,
                    dst,
                    n_total,
                    BLOCK=block,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "lift_fresh_copy kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _find_input(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        return args[0]
    if "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        return kwargs["self"]
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor):
            return v
    raise ValueError("lift_fresh_copy expects a Tensor argument")


def lift_fresh_copy(*args, **kwargs):
    logger.debug("GEMS_ENFLAME LIFT_FRESH_COPY")
    x = _find_input(*args, **kwargs)
    if x.device.type != flag_gems.device:
        raise ValueError(
            f"lift_fresh_copy Triton kernel requires a {flag_gems.device} tensor"
        )

    x_contig = x.contiguous()
    if x_contig.numel() >= _INT32_SAFE_LIMIT:
        # int32-unsafe flat offsets: defer to the generic implementation
        from flag_gems.ops.lift_fresh_copy import lift_fresh_copy as _public

        return _public(x)
    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)
    _launch_copy(x_contig, out)
    return out.view_as(x_contig)


def lift_fresh_copy_out(x: torch.Tensor, out: torch.Tensor = None):
    logger.debug("GEMS_ENFLAME LIFT_FRESH_COPY_OUT")
    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("lift_fresh_copy_out expects 'x' to be a Tensor")
    if x.device.type != flag_gems.device:
        raise ValueError(
            f"lift_fresh_copy_out Triton kernel requires {flag_gems.device} tensors"
        )

    x_contig = x.contiguous()

    if out is None:
        out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(f"Output tensor 'out' must be on {flag_gems.device}")
        if out.dtype != x_contig.dtype:
            raise ValueError("Output tensor 'out' must have the same dtype as input")
        if out.numel() != x_contig.numel() or not out.is_contiguous():
            out.resize_(x_contig.shape)
            if not out.is_contiguous():
                out = out.contiguous()

    if x_contig.numel() >= _INT32_SAFE_LIMIT:
        # int32-unsafe flat offsets: defer to the generic implementation
        from flag_gems.ops.lift_fresh_copy import lift_fresh_copy_out as _public

        return _public(x, out)

    _launch_copy(x_contig, out)
    return out.view_as(x_contig)
