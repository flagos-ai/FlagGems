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

"""rsub_scalar candidate v1: hand-written 1D kernel tiered by shape.

Production (common + gcu400) uses pointwise_dynamic rsub_func_tensor_scalar:
    return y - x * alpha   (x=tensor A, y=scalar B, alpha=scalar)
ENFLAME max_grid_size=(12,1,1) leaves large shapes under-parallelized.
Reuse the tiered strategy validated by square/square_out/clamp_max_.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total", "B", "ALPHA"])
def rsub_scalar_flat_kernel(
    X,
    OUT,
    N_total,
    B,
    ALPHA,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(X + off, mask=mask).to(tl.float32)
        r = B - x * ALPHA
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


# abs-family six-band ladder (2026-11-16 bench-validated on this card for the
# identical 1R1W out-of-place streaming shape; see abs.py / lift_fresh_copy),
# transplanted without per-band edits (zero-change abs-table migration
# precedent). The rsqrt-style fp32 arm (B32768/nw2) won the isolated
# single-tensor microbench at 16.7M (47.9-49.6 vs 49.6-52.2us) but under the
# official full-sequence bench (1Gi allocation churn + native interleave) it
# turned bimodal (48.6-56.0us across rounds, sp down to 0.91) while the abs
# B65536/nw8 form stayed tight (50.0-50.5us) -- probe gain did not survive
# production-form re-validation, so the family form is kept. The EVEN
# no-mask arm measured WITHIN-NOISE on every band (-1.8%~+0.1%,
# conj_copy verdict) and is rejected to keep a single masked body.
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


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS_ENFLAME RSUB_SCALAR")
    if not (
        isinstance(B, (int, float))
        and isinstance(alpha, (int, float))
        and A.is_contiguous()
        and A.dtype in _FLAT_DTYPES
        and 0 < A.numel() < _INT32_SAFE_LIMIT
    ):
        from flag_gems.ops.rsub import rsub_scalar as _public

        return _public(A, B, alpha)
    out = torch.empty_like(A)
    n = A.numel()
    block, num_warps = _choose_launch(n, A.element_size())
    grid_size = min(triton.cdiv(n, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # rsqrt.py / square.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(A.device):
                rsub_scalar_flat_kernel[(grid_size,)](
                    A,
                    out,
                    n,
                    float(B),
                    float(alpha),
                    BLOCK=block,
                    num_warps=num_warps,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "rsub_scalar flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )
    return out
