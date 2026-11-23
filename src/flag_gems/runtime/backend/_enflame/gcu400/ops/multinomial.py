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

from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset, uniform

logger = logging.getLogger(__name__)

# Sampling kernel: one NBLOCK-wide sample block per program, 128 or 256 lanes
# selected per shape by the host (see below); 2 warps measured best everywhere.
_SAMPLE_NUM_WARPS = 2

# Gumbel-score kernel for the without-replacement / single-sample paths.
_SCORE_BLOCK = 1024
_SCORE_NUM_WARPS = 4

# Per-distribution fp32 prefix scan, produced by one of the two kernels below.
# The prefix stays unnormalized: the sampling kernel compares against
# rv * row_total, which is exactly equivalent to the normalized-cdf compare
# and skips a whole normalize pass.
_ROW_PREFIX_MAX_ROWS = 16
_ROW_PREFIX_MAX_K = 16384


@libentry()
@triton.jit(do_not_specialize=["K", "n_rows"])
def multinomial_prefix_row_kernel(src, dst, K, n_rows, BLOCK: tl.constexpr):
    # One distribution per program, single in-register tl.cumsum block scan.
    # Fastest prefix for a handful of rows (measured 7.3us for (1, 1000) vs
    # 24.4us for the dot scan): the dot kernel's serial chunk chain and its
    # M>=16 tile padding cannot amortize over so few rows.
    row = tl.program_id(0)
    if row < n_rows:
        k = tl.arange(0, BLOCK)
        x = tl.load(src + row * K + k, mask=k < K, other=0.0).to(tl.float32)
        y = tl.cumsum(x, 0)
        tl.store(dst + row * K + k, y, mask=k < K)


@libentry()
@triton.jit(do_not_specialize=["N", "rows", "num_blocks", "num_chunks"])
def multinomial_prefix_dot_kernel(
    src,
    dst,
    N,
    rows,
    num_blocks,
    num_chunks,
    R: tl.constexpr,
    V: tl.constexpr,
):
    # Triangular-dot prefix scan (the gcu400-cumsum trick): prefix = x @ tri is
    # the only fast scan primitive on this backend (tl.cumsum in a loop body
    # measures an order of magnitude slower per element at high row counts).
    # R rows per program amortize the dot's [R, V] tiles; fp16/bf16/fp32
    # inputs all accumulate exactly in fp32.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, V)
    ri = tl.arange(0, V)[:, None]
    ci = tl.arange(0, V)[None, :]
    tri = tl.where(ri <= ci, 1.0, 0.0).to(src.dtype.element_ty)
    for rb in tl.range(pid, num_blocks, nprog):
        r = rb * R + tl.arange(0, R)
        rv = r < rows
        base = r * N
        carry = tl.zeros([R], dtype=tl.float32)
        for c in tl.range(0, num_chunks):
            offs = c * V + lanes
            lv = offs < N
            x = tl.load(
                src + base[:, None] + offs[None, :],
                mask=rv[:, None] & lv[None, :],
                other=0.0,
            )
            s = tl.dot(x, tri, input_precision="ieee")
            res = s + carry[:, None]
            carry = carry + tl.sum(x.to(tl.float32), axis=1)
            tl.store(
                dst + base[:, None] + offs[None, :],
                res,
                mask=rv[:, None] & lv[None, :],
            )


@libentry()
@triton.jit(
    do_not_specialize=[
        "K",
        "N",
        "n_sample_blocks",
        "n_work",
        "philox_seed",
        "philox_offset",
    ]
)
def multinomial_with_replacement_kernel(
    cdf_ptr,
    out_ptr,
    K,
    N,
    n_sample_blocks,
    n_work,
    philox_seed,
    philox_offset,
    NBLOCK: tl.constexpr,
):
    # Inverse-CDF sampling: a per-lane binary search of the fp32 prefix row.
    # The returned index is the leftmost category whose cumulative mass is
    # strictly greater than rv * row_total, i.e. the textbook inverse-CDF
    # sample with the same strict semantics as the generic implementation
    # (zero-mass categories can never be drawn, including a zero-mass leading
    # category when rv == 0). The flattened 1D grid (dist_id, sample_block)
    # keeps grid.y clear of the gcu400 255 limit; a grid-stride work loop
    # keeps n_dist * n_sample_blocks within the 65535 grid.x cap.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for wid in tl.range(pid, n_work, nprog):
        dist_id = wid // n_sample_blocks
        sample_batch = wid % n_sample_blocks

        n = sample_batch * NBLOCK + tl.arange(0, NBLOCK)
        y_off = dist_id * N
        rv, _, _, _ = uniform(philox_seed, philox_offset, y_off + n)

        row = cdf_ptr + dist_id * K
        total = tl.load(row + K - 1)
        target = rv * total

        start = tl.zeros((NBLOCK,), dtype=tl.int32)
        end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
        steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
        for _ in range(steps):
            mid = start + (end - start) // 2
            x = tl.load(row + mid, mask=n < N)
            start = tl.where(x <= target, mid + 1, start)
            end = tl.where(x <= target, end, mid)

        # Clamp for rows whose fp32 prefix never reaches rv * total (rounding),
        # and for degenerate all-zero rows (total == 0).
        start = tl.where(start >= K, K - 1, start)
        tl.store(out_ptr + y_off + n, start.to(out_ptr.dtype.element_ty), mask=n < N)


@libentry()
@triton.jit(
    do_not_specialize=["n_elements", "n_blocks", "philox_seed", "philox_offset"]
)
def multinomial_gumbel_scores_kernel(
    prob_ptr,
    out_ptr,
    n_elements,
    n_blocks,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # scores = prob / q with q ~ Exp(1): the gumbel-max ordering used for
    # without-replacement sampling (taking the top-n scores of s = p / q
    # selects n categories without replacement). Computed in fp32 so
    # fp16/bf16 inputs cannot overflow to inf (matching the generic op).
    # Grid-stride so tensors larger than BLOCK * 65535 stay covered.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for blk in tl.range(pid, n_blocks, nprog):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        u, _, _, _ = uniform(philox_seed, philox_offset, offs)
        q = -tl.log(1.0 - u)
        q = tl.maximum(q, 1e-20)

        p = tl.load(prob_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.maximum(p, 1e-12)

        tl.store(out_ptr + offs, p / q, mask=mask)


def _multinomial_prefix(prob):
    """Unnormalized fp32 prefix along the last dim; prob is 1D or 2D."""
    n_rows = prob.numel() // prob.size(-1)
    K = prob.size(-1)
    out = torch.empty(prob.shape, dtype=torch.float32, device=prob.device)
    if n_rows <= _ROW_PREFIX_MAX_ROWS and K <= _ROW_PREFIX_MAX_K:
        block = triton.next_power_of_2(K)
        num_warps = 1 if block <= 1024 else (2 if block <= 4096 else 4)
        multinomial_prefix_row_kernel[(n_rows,)](
            prob, out, K, n_rows, BLOCK=block, num_warps=num_warps
        )
        return out
    # Dot-scan configs, measured on the core shapes (do_bench median) and kept
    # inside the gcu400 917KB per-program DSM budget for every input dtype:
    # V is capped at 128 because a fresh compile of the fp32 (V=256, R=32)
    # tile set already needs 1,099,776 bytes (tri 256KB + [R, V] load/dot/
    # store tiles + fp32 promotion copies, double-buffered) and the tuner's
    # block-size exploration doubles V on cold keys.
    if n_rows <= 512:
        V, R, num_warps = 128, 8, 2
    elif n_rows <= 4096:
        V, R, num_warps = 128, 32, 4
    else:
        V, R, num_warps = 128, 128, 2
    num_blocks = triton.cdiv(n_rows, R)
    multinomial_prefix_dot_kernel[(min(num_blocks, 65535),)](
        prob,
        out,
        K,
        n_rows,
        num_blocks,
        triton.cdiv(K, V),
        R=R,
        V=V,
        num_warps=num_warps,
    )
    return out


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    logger.debug("GEMS_ENFLAME MULTINOMIAL")
    assert prob.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert (
        with_replacement or n_samples <= n_categories
    ), "cannot sample n_samples > prob.size(-1) samples without replacement."

    # Sampling without replacement (and the n_samples == 1 shortcut): rank the
    # gumbel-perturbed scores and take the winners. All heavy lifting runs on
    # this backend's own kernels: one scores launch here plus the FlagGems
    # argmax / sort ops for the ranking.
    if (not with_replacement) or n_samples == 1:
        import flag_gems

        scores = torch.empty(prob.shape, dtype=torch.float32, device=prob.device)
        n_elements = prob.numel()
        # philox_backend_seed_offset rounds the increment up to a multiple of 4.
        philox_seed, philox_offset = philox_backend_seed_offset(
            n_elements, generator=gen
        )
        n_blocks = triton.cdiv(n_elements, _SCORE_BLOCK)
        multinomial_gumbel_scores_kernel[(min(n_blocks, 65535),)](
            prob,
            scores,
            n_elements,
            n_blocks,
            philox_seed,
            philox_offset,
            BLOCK=_SCORE_BLOCK,
            num_warps=_SCORE_NUM_WARPS,
        )
        if n_samples == 1:
            return flag_gems.argmax(scores, dim=-1, keepdim=True)
        _, indices = flag_gems.sort(scores, dim=-1, descending=True)
        return indices[..., :n_samples]

    # With replacement: fp32 prefix row per distribution (unnormalized - the
    # sampling kernel compares against rv * row_total), then one
    # binary-search sampling launch over the flattened (dist, sample_block)
    # work items.
    cum_prob = _multinomial_prefix(prob)

    if cum_prob.dim() == 1:
        n_dist = 1
        out = torch.empty((n_samples,), device=prob.device, dtype=torch.int64)
    else:
        n_dist = cum_prob.size(0)
        out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int64)
    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=gen)
    # Measured per-shape optimum: wide sample blocks (fewer programs) win once
    # the wide-block grid still holds enough programs to fill the device;
    # single small distributions keep the narrow block for its extra programs.
    wide_blocks = triton.cdiv(n_samples, 256)
    nblock = 256 if n_dist * wide_blocks >= 32 else 128
    n_sample_blocks = triton.cdiv(n_samples, nblock)
    n_work = n_sample_blocks * n_dist
    multinomial_with_replacement_kernel[(min(n_work, 65535),)](
        cum_prob,
        out,
        n_categories,
        n_samples,
        n_sample_blocks,
        n_work,
        philox_seed,
        philox_offset,
        NBLOCK=nblock,
        num_warps=_SAMPLE_NUM_WARPS,
    )
    return out


def _install_torch_multinomial_shim() -> None:
    """Make plain ``torch.multinomial`` calls survivable on gcu400.

    torch_gcu's native topsatenMultinomial only supports i32/u32 outputs while
    the aten contract mandates an int64 result, so every plain gcu call fails
    its argument check and terminates the whole process from C++
    (op_aten_multinomial.h:83 -> gcu_utils.cpp:74 Check failed: 0, SIGABRT).
    That abort kills benchmark baseline legs and any reference leg that calls
    torch.multinomial directly on gcu tensors.

    ENFLAME_PT_OP_DEBUG_CONFIG is parsed by torch_gcu at .so load, i.e. before
    this module is imported, so an env recipe cannot be applied from here -
    and fallback_cpu=all_ops additionally poisons do_bench (the L2-flush
    between iterations takes ~207ms, which caps n_repeat at 1 and turns the
    candidate medians into single noisy samples). Instead install a
    dispatch-context-aware python shim (same mechanism as cumsum.py /
    softmax.py in this directory):

    - while a FlagGems registration is active (use_gems holds the
      ``multinomial`` key) pass straight through to the native entry: the
      aten dispatcher routes to this backend's kernel, which is exactly what
      benchmark candidate legs must measure;
    - non-gcu tensors pass through to the native entry unchanged;
    - otherwise (plain caller on gcu) compute via the CPU reference
      implementation and move the result back - abort-proof and value-exact.
      The gcu generator argument cannot drive the CPU RNG and is dropped.
    """
    native = getattr(torch, "multinomial", None)
    if native is None or getattr(native, "_flaggems_gcu400_multinomial_shim", False):
        return

    def _multinomial_shim(
        input, num_samples, replacement=False, *, generator=None, out=None
    ):
        if torch.is_tensor(input) and input.device.type == "gcu":
            import flag_gems

            registrar = getattr(flag_gems, "current_work_registrar", None)
            keys = getattr(registrar, "all_keys", ()) if registrar is not None else ()
            if "multinomial" not in keys:
                result = native(input.detach().cpu(), num_samples, replacement)
                if out is not None:
                    out.copy_(result)
                    return out
                return result.to(input.device)
        return native(input, num_samples, replacement, generator=generator, out=out)

    _multinomial_shim._flaggems_gcu400_multinomial_shim = True
    torch.multinomial = _multinomial_shim


_install_torch_multinomial_shim()
