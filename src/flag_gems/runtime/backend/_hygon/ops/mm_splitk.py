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

"""gfx936 small-output MM with tunable global and CTA-local K partitions."""

import copy
import os

import torch

import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

try:
    from triton.backends.hcu.compiler import HIPOptions
except ImportError:
    _SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED = False
else:
    _SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED = (
        "wave_private_sync" in HIPOptions.__dataclass_fields__
    )

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_hygon_splitk_expand.yaml")
)


@libentry()
@libtuner(
    configs=[triton.Config(dict(BLOCK=256), num_warps=4, num_stages=1)],
    key=["MN", "SPLITS"],
    strategy=["default"] * 2,
    warmup=5,
    rep=10,
    benchmark_mode="replay",
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_hygon_splitk_post",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def _mm_splitk_post(W, C, MN: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(W + split * MN + i, i < MN, other=0.)
    tl.store(C + i, acc, i < MN)


def _splitk_post_hook(args, exception):
    # Triton invokes this inside its timed kernel_call. The final launch and
    # LibEntry cache hits are completed by the public wrapper below.
    if exception is None and args["KIND"] == 0 and args["SPLITS"] > 1:
        mn = args["M"] * args["N"]
        _mm_splitk_post[lambda meta: (triton.cdiv(mn, meta["BLOCK"]),)](
            args["W"], args["C"], mn, args["SPLITS"],
        )


def _splitk_default_configs():
    # Retain the two existing synchronous global-split families for Default.
    # Expanded replaces this list with the complete declared YAML space.
    return [
        triton.Config(
            dict(BM=32, BN=bn, BK=bk, SPLITS=splits, KIND=0,
                 LOAD_MODE=0, optimize_epilogue=False,
                 schedule_hint="local-prefetch"),
            num_warps=2, num_stages=2,
        )
        for bn, bk, splits in [(16, 64, 32), (32, 32, 16)]
    ]


def _splitk_prune(configs, named_args, **kwargs):
    mask = int(kwargs.get("ASYNC_LOAD_MASK", named_args.get("ASYNC_LOAD_MASK", 0)))
    # Tail tiles and shallow per-split loops remain eligible. Resource limits
    # are checked by the compiler/launcher, not an estimated performance model.
    supports_private = bool(kwargs.get(
        "WAVE_PRIVATE_SYNC_SUPPORTED",
        named_args.get("WAVE_PRIVATE_SYNC_SUPPORTED", False),
    ))
    selected = []
    for config in configs:
        if config.kwargs["LOAD_MODE"] & ~mask:
            continue
        if not supports_private and config.kwargs.get("wave_private_sync", False):
            continue
        if not supports_private and "wave_private_sync" in config.kwargs:
            # Older HCU backends must not receive an unknown backend option.
            config = copy.deepcopy(config)
            config.kwargs.pop("wave_private_sync")
        selected.append(config)
    return selected


@libentry()
@libtuner(
    configs=_splitk_default_configs(),
    key=["M", "N", "K", "stride_bk", "stride_bn", "B_K_CONTIGUOUS",
         "ASYNC_LOAD_MASK", "WAVE_PRIVATE_SYNC_SUPPORTED", "M_ALIGNMENT", "N_ALIGNMENT"],
    strategy=["default"] * 10,
    warmup=5,
    rep=10,
    benchmark_mode="replay",
    post_hook=_splitk_post_hook,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_hygon_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    prune_configs_by={"early_config_prune": _splitk_prune},
)
@triton.jit
def _mm_splitk_partial(A, B, W, C, M, N, K, stride_bk, stride_bn,
            B_K_CONTIGUOUS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            SPLITS: tl.constexpr, KIND: tl.constexpr,
            LOAD_MODE: tl.constexpr, ASYNC_LOAD_MASK: tl.constexpr,
            WAVE_PRIVATE_SYNC_SUPPORTED: tl.constexpr,
            M_ALIGNMENT: tl.constexpr = 1, N_ALIGNMENT: tl.constexpr = 1):
    tl.static_assert(KIND == 0 or KIND == 1)
    tl.static_assert(LOAD_MODE >= 0 and LOAD_MODE <= 3)
    tl.static_assert((LOAD_MODE & ~ASYNC_LOAD_MASK) == 0)
    # Host launch contract: exact ceil-grid, positive M/N, facts divide M/N.
    tl.static_assert(M_ALIGNMENT > 0 and (M_ALIGNMENT & (M_ALIGNMENT - 1)) == 0)
    tl.static_assert(N_ALIGNMENT > 0 and (N_ALIGNMENT & (N_ALIGNMENT - 1)) == 0)
    FULL_M: tl.constexpr = M_ALIGNMENT % BM == 0
    FULL_N: tl.constexpr = N_ALIGNMENT % BN == 0
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    if KIND == 0:
        split = tl.program_id(2)
        # Match the existing production split-K interleaving exactly. Keep
        # both loads as block pointers to avoid the known DTK masked-pointer
        # byte-offset problem; every candidate is still numerically checked.
        ap = tl.make_block_ptr(A, (M, K), (K, 1),
                              (tl.program_id(0) * BM, split * BK),
                              (BM, BK), (1, 0))
        if B_K_CONTIGUOUS:
            bp = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn),
                                  (split * BK, tl.program_id(1) * BN),
                                  (BK, BN), (0, 1))
        else:
            bp = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn),
                                  (split * BK, tl.program_id(1) * BN),
                                  (BK, BN), (1, 0))
        acc = tl.zeros((BM, BN), tl.float32)
        for q in range(tl.cdiv(K, BK * SPLITS)):
            if FULL_M:
                a = tle.load(ap, boundary_check=(1,), padding_option='zero',
                             is_async=(LOAD_MODE & 1) != 0)
            else:
                a = tle.load(ap, boundary_check=(0, 1), padding_option='zero',
                             is_async=(LOAD_MODE & 1) != 0)
            if FULL_N:
                b = tle.load(bp, boundary_check=(0,), padding_option='zero',
                             is_async=(LOAD_MODE & 2) != 0)
            else:
                b = tle.load(bp, boundary_check=(0, 1), padding_option='zero',
                             is_async=(LOAD_MODE & 2) != 0)
            acc = tl.dot(a, b, acc)
            ap = tl.advance(ap, (0, BK * SPLITS))
            bp = tl.advance(bp, (BK * SPLITS, 0))
        if SPLITS == 1:
            out = C + rm[:, None] * N + rn[None, :]
        else:
            out = W + split * M * N + rm[:, None] * N + rn[None, :]
        if FULL_M and FULL_N:
            tl.store(out, acc)
        elif FULL_M:
            tl.store(out, acc, rn[None, :] < N)
        elif FULL_N:
            tl.store(out, acc, rm[:, None] < M)
        else:
            tl.store(out, acc, (rm[:, None] < M) & (rn[None, :] < N))
    else:
        # Express CTA-internal K partitioning with existing batched dot and
        # reduction primitives. Inspect actual warp distribution before
        # calling this the same parallel algorithm as Torch.
        sp = tl.arange(0, SPLITS)
        acc = tl.zeros((SPLITS, BM, BN), tl.float32)
        for q in range(tl.cdiv(K, BK * SPLITS)):
            kk = (sp[:, None] + q * SPLITS) * BK + kr[None, :]
            if FULL_M:
                a = tle.load(A + rm[None, :, None] * K + kk[:, None, :],
                             kk[:, None, :] < K,
                             other=0., is_async=(LOAD_MODE & 1) != 0)
            else:
                a = tle.load(A + rm[None, :, None] * K + kk[:, None, :],
                             (rm[None, :, None] < M) & (kk[:, None, :] < K),
                             other=0., is_async=(LOAD_MODE & 1) != 0)
            if FULL_N:
                b = tle.load(B + kk[:, :, None] * stride_bk
                             + rn[None, None, :] * stride_bn,
                             kk[:, :, None] < K,
                             other=0., is_async=(LOAD_MODE & 2) != 0)
            else:
                b = tle.load(B + kk[:, :, None] * stride_bk
                             + rn[None, None, :] * stride_bn,
                             (kk[:, :, None] < K) & (rn[None, None, :] < N),
                             other=0., is_async=(LOAD_MODE & 2) != 0)
            acc = tl.dot(a, b, acc)
        result = tl.sum(acc, 0)
        if FULL_M and FULL_N:
            tl.store(C + rm[:, None] * N + rn[None, :], result)
        elif FULL_M:
            tl.store(C + rm[:, None] * N + rn[None, :], result, rn[None, :] < N)
        elif FULL_N:
            tl.store(C + rm[:, None] * N + rn[None, :], result, rm[:, None] < M)
        else:
            tl.store(C + rm[:, None] * N + rn[None, :], result,
                     (rm[:, None] < M) & (rn[None, :] < N))


def _splitk_parameters(a, b, c):
    if tle is None:
        return None
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        return None
    M, K = a.shape
    N = b.shape[1]
    if not (
        M >= 2 and N >= 2
        # K partitions supply parallel work for small output grids. Limit
        # the grid to 64 tiles and the tuning workspace to 16MiB.
        and triton.cdiv(M, 32) * triton.cdiv(N, 32) <= 64
        and K >= 2048
        and b.shape[0] == K
        and c.shape == (M, N)
        and a.dtype == b.dtype == c.dtype
        and a.dtype in (torch.bfloat16, torch.float16)
        and a.device == b.device == c.device
        and a.device.type == "cuda"
        and not c.requires_grad
        and (not c.is_inference() or torch.is_inference_mode_enabled())
        and c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        and a.is_contiguous() and c.is_contiguous()
        and (b.is_contiguous() or (
            b.stride(0) == 1 and b.stride(1) == K and b.t().is_contiguous()
        ))
        and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
        and all(t.untyped_storage().nbytes() <= 2**31 - 1 for t in (a, b, c))
    ):
        return None
    if N < 16:
        if K < 8192:
            return None
        block_n, block_k, splits = 16, 64, 32
    else:
        block_n, block_k, splits = 32, 32, 16
    from triton import knobs

    if not (
        knobs.hcu.use_buffer_ops
        and not knobs.hcu.use_async_copy
        and not knobs.hcu.use_block_pingpong
        and not knobs.hcu.use_in_thread_transpose
    ):
        return None
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        return None
    return 32, block_n, block_k, splits


def _splitk_alignment_facts(M, N, enabled=True):
    """Divisibility facts for the exact ceil-grid used by try_mm_splitk."""
    if type(enabled) is not bool:
        raise TypeError("use_alignment_facts must be boolean")
    if not enabled or any(type(x) is not int or not 0 < x <= 2**31 - 1 for x in (M, N)):
        return 1, 1
    return M & -M, N & -N


def try_mm_splitk(a, b, c, *, use_alignment_facts=True, compare_packet=False):
    """Write c using the small-output family; return False outside its domain."""
    if type(use_alignment_facts) is not bool:
        raise TypeError("use_alignment_facts must be boolean")
    if type(compare_packet) is not bool:
        raise TypeError("compare_packet must be boolean")
    if _splitk_parameters(a, b, c) is None:
        return False
    M, K = a.shape
    N = b.shape[1]
    if compare_packet and use_alignment_facts and b.stride() == (1, K):
        _mm_splitk_partial._apply_flagtune()
        tuner = _mm_splitk_partial.fn
        if tuner._run_mode == "normal":
            from .mm_packet import try_mm_packet

            key = (
                "splitK-baseline", tuner.config_table_name,
                id(tuner.configs), tuple(tuner._benchmark_protocol),
            )
            if try_mm_packet(
                a, b, c, lambda: try_mm_splitk(a, b, c), tuning_key=key,
                ordinary_with_output=lambda output: try_mm_splitk(a, b, output),
                variants=("splitu_nt_k256_16", "splitu_nt_k256_peeled_16"),
            ):
                return True
    ma, na = _splitk_alignment_facts(M, N, use_alignment_facts)
    # Facts below are valid only for this exact ceil-grid and its original
    # pid axes. Arbitrary caller grids must retain the original checks.
    # All Expanded configurations share one workspace. CTA/GSU1 leave it
    # unused; every global split writes its complete disjoint FP32 partial.
    workspace = torch.empty((64, M, N), device=a.device, dtype=torch.float32)
    with torch_device_fn.device(a.device):
        from .mm import _hygon_mm_async_load_mask

        mask = _hygon_mm_async_load_mask(a, b, c, M, N, K)
        grid = lambda meta: (
            triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]),
            meta["SPLITS"] if meta["KIND"] == 0 else 1,
        )
        _, meta = _mm_splitk_partial[grid](
            a, b, workspace, c, M, N, K, *b.stride(),
            B_K_CONTIGUOUS=b.stride(0) == 1, ASYNC_LOAD_MASK=mask,
            WAVE_PRIVATE_SYNC_SUPPORTED=_SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED,
            M_ALIGNMENT=ma, N_ALIGNMENT=na,
        )
        # LibEntry returns the metadata for this exact launch, including on
        # cache hits; do not read a mutable global tuner's best_config.
        _splitk_post_hook(dict(meta, W=workspace, C=c, M=M, N=N, K=K), None)
    return True
