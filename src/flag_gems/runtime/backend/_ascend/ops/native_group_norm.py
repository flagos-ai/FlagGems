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
"""TLE (``triton.experimental.tle``) implementation of ``native_group_norm``.

Shape: pure vector work (no ``tl.dot``), so the kernel is vec-explicit: one
program owns whole ``(batch, group)`` pairs of the flattened ``(N, C, HxW)``
buffer and strides over the pairs by ``tl.num_programs(0)``.

Round-3 change: the per-pair cost, not the data volume, is the limiter.  The
measured round-2 latencies were flat across a 5.5x span of pair size on the mid
shapes (a pair of 256, 768 and 1408 elements all take ~10.6-11.2 us) and were
the same for float16 and float32 on the large shape even though float32 moves
2x the bytes, while the *reference* shows the same flat shape.  Fitting the
round-2 mid-shape measurements (32 pairs -> 5.002 us, 128 pairs -> 10.823 us)
gives an intercept of ~3.1 us and a slope of ~60 ns per pair; the reference's
slope over the same two points is ~18 ns per pair.  The whole mid-shape gap is
therefore the ~6 sequential masked tile memory operations the chunked three-pass
walk issues per pair (two per reduction pass, plus one load/store pair per
channel in the normalize pass), independent of ``M``.

So the kernel now has two paths, selected on the host from the pair footprint
and pruned at trace time by a ``tl.constexpr`` flag:

* ``SINGLE`` -- the whole padded pair ``[group_size, HW]`` fits one 2-D tile
  (``CH_BLOCK x HW_BLOCK``, both constexpr, derived from the runtime geometry).
  The pair is read with **one** masked 2-D load, both reductions run on the
  register tile (a 2-D ``tl.sum`` reduces every axis to the pair scalar that the
  reference computes), and the normalized tile is written with **one** masked
  store.  The weight/bias term is a ``CH_BLOCK``-wide vector load broadcast down
  the rows, so there is no per-lane channel indexing and no per-channel loop.
  This cuts the mid shapes from ~6 to 2 tile memory ops per pair.
* otherwise -- the streaming three-pass path for pairs whose footprint is far too
  large to be resident (the large-HW shape: 2 x 65536 elements).  It walks
  contiguous ``BLOCK_M`` pieces of the pair through ``tle.dsa.copy`` GM<->UB
  staging with a full-width fp32 vector accumulator and one ``tl.sum`` per pass,
  with a channel-major normalize pass so the weight/bias index stays a
  per-channel scalar.  It is the regression guard for the SINGLE path.

Round-4 change: ``BLOCK_M`` is raised from 2048 to 4096, and it only affects the
streaming path.  The round-3 measurements show that path is not byte-bound: on
the large shape float16 moves exactly half the bytes of float32 (a 4096-element
tile is 8 KB vs 16 KB) yet runs only ~4.5% faster (0.078272 ms vs 0.0818227 ms),
and float16 vs bfloat16, which move identical bytes, differ by ~3.7%.  A
bandwidth-limited walk would show a ~2x spread across the fp16/fp32 pair.  What
the three dtypes *do* share is the number of sequential staging operations per
pass (HW / BLOCK_M chunks per channel pass), so the cost tracks the serialized
GM<->UB copy + ``to_tensor`` + accumulate chain rather than the traffic.
Doubling ``BLOCK_M`` halves that chain at identical bytes: for the large shape
the per-pair walk drops from 192 to 96 staged chunks (32 + 32 reduction reads
plus 2 x 16 normalize read/write chunks), which should cut the streaming-path
latency materially if the per-chunk fixed cost dominates.  If the latency is
instead unchanged, the byte-bound reading is correct and the next lever is
merging the two reduction passes into a single read, not a larger tile.
``BLOCK_M`` is an existing source-level construct, so this is tile tuning, and
the masked remainder arithmetic (``nfull * BLOCK_M < Mv``, ``hfull * BLOCK_M <
HW``) is kept unchanged so the tail behaviour is identical for any geometry.

The SINGLE path uses masked ``tl.load`` / ``tl.store`` rather than
``tle.dsa.copy`` because that is a capability boundary, not a preference: the
padded rectangle has a *runtime* row stride (``HW``) plus a two-sided boundary
mask (rows past ``C``, columns past ``HW``), while ``dsa.copy`` takes a static
block shape and has no mask.  The streaming path, whose blocks are contiguous
``BLOCK_M`` runs lying fully inside the pair, keeps using ``tle.dsa.alloc`` /
``tle.dsa.copy``.

Numerics follow the pinned reference (``flag_gems/ops/groupnorm.py``) exactly:
``mean = sum(x) / M`` over the full ``M = group_size * HW`` (positions beyond
``C`` contribute 0), then ``var = sum((x - mean)^2) / M`` (biased), then
``rstd = rsqrt(var + eps)`` and ``out = (x - mean) * rstd * w[c] + b[c]``.
Both reductions accumulate in fp32, so the SINGLE path's 2-D reduction has the
same precision class as the streaming path's per-pass ``tl.sum``.  ``out`` keeps
the input dtype; ``mean``/``rstd`` are stored in the *input* dtype as well,
because the FlagGems assertion helper checks ``res.dtype == dtype`` before
comparing values.
"""

import logging

import torch
import torch_npu  # noqa: F401  -- must be imported before triton on Ascend
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

# Staging width for the streaming (non-SINGLE) path.  Doubled from 2048 to 4096
# in round 4: the measured streaming-path latency is nearly insensitive to the
# element dtype (fp16 is 4.5% faster than fp32 across half the bytes), so the
# limiter is the serialized per-chunk copy/convert/accumulate chain, and halving
# the chunk count halves that chain at identical traffic.  It does not affect the
# SINGLE path, which is sized by CH_BLOCK x HW_BLOCK.
_BLOCK_M = 4096
_SINGLE_MAX_LANES = 4096
_CACHED_CORE_NUM = None


def _vector_core_num():
    """Number of vector engines, queried from the device (never hard-coded).

    The submission target (Ascend910B2) and the development/evaluation device
    (Ascend910B4) do not report the same core count, so the value is read at
    run time and the grid is capped by it.
    """
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        num = 0
        try:
            from triton.runtime import driver as _driver

            props = _driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
            try:
                num = int(props["num_vectorcore"])
            except (TypeError, KeyError, IndexError):
                num = int(getattr(props, "num_vectorcore", 0) or 0)
        except Exception:  # noqa: BLE001
            num = 0
        _CACHED_CORE_NUM = num if num > 0 else 0
    return _CACHED_CORE_NUM


@triton.jit
def _native_group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    total_pairs,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    SINGLE: tl.constexpr,
    CH_BLOCK: tl.constexpr,
    HW_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    if SINGLE:
        # Whole padded pair resident: one masked 2-D load, two reductions over
        # the register tile, one masked 2-D store.
        ro = tl.arange(0, CH_BLOCK)
        co = tl.arange(0, HW_BLOCK)
        aoff = ro[:, None] * HW + co[None, :]
        for gp in range(pid, total_pairs, nprog):
            g = gp % num_groups
            bs = gp // num_groups
            chan0 = g * group_size
            base = (bs * C + chan0) * HW
            rmask = ro < tl.minimum(group_size, C - chan0)
            amask = rmask[:, None] & (co[None, :] < HW)
            Mn = group_size * HW

            x = tl.load(X + base + aoff, mask=amask, other=0.0).to(tl.float32)
            mean = tl.sum(x) / Mn
            d = tl.where(amask, x - mean, 0.0)
            rstd = tl.rsqrt(tl.sum(d * d) / Mn + eps)

            y = d * rstd
            if HAS_W:
                w = tl.load(W + chan0 + ro, mask=rmask, other=1.0).to(tl.float32)
                y = y * w[:, None]
            if HAS_B:
                b = tl.load(B + chan0 + ro, mask=rmask, other=0.0).to(tl.float32)
                y = y + b[:, None]

            tl.store(Y + base + aoff, y.to(Y.dtype.element_ty), mask=amask)
            tl.store(Mean + gp, mean.to(Mean.dtype.element_ty))
            tl.store(Rstd + gp, rstd.to(Rstd.dtype.element_ty))
    else:
        # Streaming path: contiguous BLOCK_M runs fully inside the pair, so
        # GM<->UB bulk movement uses tle.dsa.copy both ways.
        offs = tl.arange(0, BLOCK_M)
        ub_x = tle.dsa.alloc(
            [BLOCK_M], dtype=X.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
        )
        for gp in range(pid, total_pairs, nprog):
            g = gp % num_groups
            bs = gp // num_groups
            chan0 = g * group_size
            base = (bs * C + chan0) * HW
            nchan = tl.minimum(group_size, C - chan0)
            Mv = nchan * HW
            Mn = group_size * HW

            # pass 1: sum -> mean (one vector accumulator, one reduction)
            nfull = Mv // BLOCK_M
            acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i in range(nfull):
                o = i * BLOCK_M + offs
                tle.dsa.copy(X + base + o, ub_x, [BLOCK_M])
                acc += tle.dsa.to_tensor(ub_x).to(tl.float32)
            if nfull * BLOCK_M < Mv:
                o = nfull * BLOCK_M + offs
                x = tl.load(X + base + o, mask=o < Mv, other=0.0).to(tl.float32)
                acc = acc + x
            mean = tl.sum(acc) / Mn

            # pass 2: centered sum of squares -> rstd (biased variance)
            acc2 = tl.zeros([BLOCK_M], dtype=tl.float32)
            for i in range(nfull):
                o = i * BLOCK_M + offs
                tle.dsa.copy(X + base + o, ub_x, [BLOCK_M])
                d = tle.dsa.to_tensor(ub_x).to(tl.float32) - mean
                acc2 += d * d
            if nfull * BLOCK_M < Mv:
                o = nfull * BLOCK_M + offs
                m = o < Mv
                x = tl.load(X + base + o, mask=m, other=0.0).to(tl.float32)
                d = tl.where(m, x - mean, 0.0)
                acc2 = acc2 + d * d
            rstd = tl.rsqrt(tl.sum(acc2) / Mn + eps)

            # pass 3: normalize and store, channel-major so the weight/bias
            # index stays a per-channel scalar (no per-lane channel gather).
            hfull = HW // BLOCK_M
            for ci in range(nchan):
                c = chan0 + ci
                ch = base + ci * HW
                w = tl.load(W + c).to(tl.float32) if HAS_W else 1.0
                b = tl.load(B + c).to(tl.float32) if HAS_B else 0.0
                for i in range(hfull):
                    o = i * BLOCK_M + offs
                    tle.dsa.copy(X + ch + o, ub_x, [BLOCK_M])
                    y = (tle.dsa.to_tensor(ub_x).to(tl.float32) - mean) * rstd
                    if HAS_W:
                        y = y * w
                    if HAS_B:
                        y = y + b
                    yb = tle.dsa.to_buffer(
                        y.to(Y.dtype.element_ty), space=tle.dsa.ascend.UB
                    )
                    tle.dsa.copy(yb, Y + ch + o, [BLOCK_M])
                if hfull * BLOCK_M < HW:
                    o = hfull * BLOCK_M + offs
                    m = o < HW
                    y = (
                        tl.load(X + ch + o, mask=m, other=0.0).to(tl.float32) - mean
                    ) * rstd
                    if HAS_W:
                        y = y * w
                    if HAS_B:
                        y = y + b
                    tl.store(Y + ch + o, y.to(Y.dtype.element_ty), mask=m)

            tl.store(Mean + gp, mean.to(Mean.dtype.element_ty))
            tl.store(Rstd + gp, rstd.to(Rstd.dtype.element_ty))


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def run(input, weight, bias, N, C, HxW, group, eps=1e-5):
    if input.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.native_group_norm import native_group_norm as _generic

        return _generic(input, weight, bias, N, C, HxW, group, eps)

    logger.debug("GEMS_ASCEND NATIVE_GROUP_NORM")

    if not input.is_contiguous():
        input = input.contiguous()
    if weight is not None and not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    N = int(N)
    C = int(C)
    HxW = int(HxW)
    group = int(group)
    eps = float(eps)

    group_size = triton.cdiv(C, group)
    total_pairs = N * group

    out = torch.empty_like(input)
    mean = torch.empty((N, group), dtype=input.dtype, device=input.device)
    rstd = torch.empty((N, group), dtype=input.dtype, device=input.device)

    if total_pairs <= 0:
        return out, mean, rstd

    # Trace-time path selection: either the padded pair rectangle fits one tile
    # (2 tile memory ops per pair) or the pair must be streamed (3 passes).
    ch_block = triton.next_power_of_2(group_size)
    hw_block = triton.next_power_of_2(HxW)
    single = ch_block * hw_block <= _SINGLE_MAX_LANES

    grid_n = total_pairs
    ncore = _vector_core_num()
    if ncore > 0:
        grid_n = min(grid_n, ncore)

    has_w = weight is not None
    has_b = bias is not None
    w_arg = weight if has_w else out
    b_arg = bias if has_b else out

    _native_group_norm_kernel[(grid_n,)](
        input,
        out,
        w_arg,
        b_arg,
        mean,
        rstd,
        total_pairs,
        group_size,
        C,
        HxW,
        group,
        eps,
        HAS_W=has_w,
        HAS_B=has_b,
        SINGLE=single,
        CH_BLOCK=ch_block,
        HW_BLOCK=hw_block,
        BLOCK_M=_BLOCK_M,
        multibuffer=True,
    )
    return out, mean, rstd


native_group_norm = run
