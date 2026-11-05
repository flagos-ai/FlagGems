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

"""PT2 contracts for the existing FlagGems MHC Triton kernels.

Each contract launches the *original* ``@triton.autotune``-wrapped kernel
object through ``torch.library.triton_op`` + ``torch.library.wrap_triton``.
Dynamo explicitly supports exactly one autotuner layer on a wrapped kernel
(``torch._higher_order_ops.triton_kernel_wrap``) and Inductor re-autotunes
among the same declared ``configs`` at compile time
(``define_user_defined_triton_kernel(kernel, configs, ...))``.  In eager
execution ``triton_op``'s backend runs the op body with ``wrap_triton``
disabled, i.e. the call falls through to the original ``Autotuner.run`` —
bit-identical to the uncompiled FlagGems launcher, including its
benchmarking cache.  No kernel is copied or rewritten, and no launch
parameter is decided by Python inside the traced body: running host-side
cache bookkeeping under symbolic shapes during the fake/meta trace was the
root cause of the old ``TypeError('unhashable type: non-nested SymInt')``
failures in a pinned-config design (autotuner-cache keys embed shapes).

Grids: ``mhc_pre`` / ``hc_head_fused`` launch with the original
``(num_tokens,)`` tuple grid, which does not depend on the selected config.
``mhc_post`` needs ``cdiv(H, BLOCK_H)`` while ``BLOCK_H`` is only chosen at
compile time under Inductor autotuning, so it uses the documented META
closure grid pattern (``lambda META: (N, cdiv(H, META["BLOCK_H"]))``), which
``wrap_triton`` supports for both eager (wrap disabled → plain
``Autotuner.run(grid=callable)``) and compile.
"""

# ``import a.b.c as name`` resolves to the attribute ``c`` of package ``a.b``
# when a.b.__init__ eagerly re-exports a same-named symbol (flag_gems.fused.mhc
# does exactly that), so bind the submodules through importlib instead.
import importlib as _importlib

import torch

import flag_gems.fused.mhc as _mhc_pkg  # noqa: F401  (ensures submodules loaded)
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec

_hc_head_mod = _importlib.import_module("flag_gems.fused.mhc.hc_head_fused_kernel")
_mhc_post_mod = _importlib.import_module("flag_gems.fused.mhc.mhc_post")
_mhc_pre_mod = _importlib.import_module("flag_gems.fused.mhc.mhc_pre")

# The original autotune-wrapped kernel objects: ``*.fn`` chains end at the
# same JITFunction objects the eager launchers invoke.
MHC_PRE_FUSED_AUTOTUNER = _mhc_pre_mod.mhc_pre_fused_kernel_hc_mult_4
MHC_PRE_GENERIC_AUTOTUNER = _mhc_pre_mod.mhc_pre_generic_kernel
MHC_POST_HCMULT4_AUTOTUNER = _mhc_post_mod.mhc_post_kernel_hc_mult_4
MHC_POST_GENERIC_AUTOTUNER = _mhc_post_mod.mhc_post_kernel_generic
HC_HEAD_FUSED_AUTOTUNER = _hc_head_mod._hc_head_fused_kernel

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)


def supports_pt2_triton() -> bool:
    """Return whether this Torch build exposes the required PT2 Triton APIs."""

    return _HAS_TRITON_OP


def _num_tokens_bucket(num_tokens: int) -> int:
    """Mirror of the bucket table in ``flag_gems.fused.mhc.mhc_pre.mhc_pre``.

    It is passed to the kernel as the ``num_tokens_bucket`` autotune-key
    argument, so the traced path must produce the identical value for every
    token count; keeping it in sync is part of the contract.
    """

    if num_tokens <= 512:
        return 1
    if num_tokens <= 1024:
        return 2
    if num_tokens <= 2048:
        return 3
    if num_tokens <= 4096:
        return 4
    return 5


if _HAS_TRITON_OP:

    # The fn->bf16 cast of the GEMM must not re-execute on every compiled
    # call.  ``torch.compile`` cannot hoist a tensor->tensor cast out of a
    # graph (it depends on the runtime value of ``fn``), and the original
    # WeakKeyDictionary/_version cache cannot be traced.  The split below
    # moves the cast+GEMM behind a plain custom op whose body runs *outside*
    # the graph; Inductor treats it as an opaque extern call, so the
    # data_ptr+_version cache is consulted once per call and the cast fires
    # exactly once per (weight tensor, version).
    _FN_BF16_CACHE: dict = {}
    _FN_BF16_CAST_CALLS = {"n": 0}  # cast-count observability for benchmarks

    def _get_fn_bf16_cached(fn: torch.Tensor) -> torch.Tensor:
        key = fn.data_ptr()
        entry = _FN_BF16_CACHE.get(key)
        if entry is not None and entry[0] == fn._version:
            return entry[1]
        _FN_BF16_CAST_CALLS["n"] += 1
        out = fn.to(dtype=torch.bfloat16)
        # strong ref: guards against data_ptr reuse after the weight is freed
        _FN_BF16_CACHE[key] = (fn._version, out, fn)
        return out

    @torch.library.custom_op("flag_gems_pt2::mhc_pre_gemm", mutates_args=())
    def _mhc_pre_gemm_op(
        residual_flat: torch.Tensor,  # (num_tokens, hc_mult, hidden_size), bf16
        fn: torch.Tensor,  # (hc_mult3, hc_hidden_size), fp32
    ) -> torch.Tensor:
        """fn->bf16 cached cast + cuBLAS GEMM, opaque to the compiled graph."""
        num_tokens = residual_flat.shape[0]
        hc_hidden_size = fn.shape[1]
        x_flat = residual_flat.reshape(num_tokens, hc_hidden_size)
        fn_bf16 = _get_fn_bf16_cached(fn)
        return torch.mm(x_flat, fn_bf16.t()).float()

    @_mhc_pre_gemm_op.register_fake
    def _(residual_flat, fn):
        return residual_flat.new_empty(
            residual_flat.shape[0], fn.shape[0], dtype=torch.float32
        )

    @torch.library.triton_op(
        "flag_gems_pt2::mhc_pre",
        mutates_args={"post_mix", "comb_mix", "layer_input"},
    )
    def _mhc_pre_op(
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
    ) -> None:
        """Opaque cached GEMM + original fused MHC-pre Triton kernel."""

        num_tokens = residual.shape[0]
        hc_mult = residual.shape[1]
        hidden_size = residual.shape[2]
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        hc_hidden_size = hc_mult * hidden_size
        num_tokens_bucket = _num_tokens_bucket(num_tokens)

        # Step 1: cached bf16 cast + GEMM behind a custom op (outside graph).
        gemm_out = torch.ops.flag_gems_pt2.mhc_pre_gemm(residual, fn)

        # Step 2: the original autotune-wrapped kernel, same call signature
        # as the eager launcher.  Eager (wrap_triton disabled by triton_op):
        # falls through to Autotuner.run.  Compile: Dynamo traces the single
        # autotuner layer and Inductor re-autotunes among the same configs.
        if hc_mult == 4:
            torch.library.wrap_triton(_mhc_pre_mod.mhc_pre_fused_kernel_hc_mult_4)[
                (num_tokens,)
            ](
                gemm_out,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                num_tokens_bucket,
                residual.stride(0),
                residual.stride(1),
                residual.stride(2),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC_MULT3=hc_mult3,
            )
        else:
            torch.library.wrap_triton(_mhc_pre_mod.mhc_pre_generic_kernel)[
                (num_tokens,)
            ](
                gemm_out,
                hc_scale,
                hc_base,
                residual,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                num_tokens_bucket,
                residual.stride(0),
                residual.stride(1),
                residual.stride(2),
                layer_input.stride(0),
                layer_input.stride(1),
                hidden_size,
                hc_hidden_size,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                HC=hc_mult,
            )

    @torch.library.triton_op("flag_gems_pt2::mhc_post", mutates_args={"out"})
    def _mhc_post_op(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Original MHC-post Triton kernel with the documented META grid."""

        N, hc, H = residual.shape
        a = comb.contiguous()
        b = residual.contiguous()
        c = post.squeeze(-1).contiguous()
        d = x.contiguous()

        if hc == 4:
            # BLOCK_H is chosen by the (Inductor- or eager-) autotuner, so the
            # grid must be computed from the config: documented META-closure
            # grid, equivalent to the eager launcher's ``grid_specialized``.
            torch.library.wrap_triton(_mhc_post_mod.mhc_post_kernel_hc_mult_4)[
                lambda META: (N, (H + META["BLOCK_H"] - 1) // META["BLOCK_H"])
            ](
                a,
                b,
                c,
                d,
                out,
                H=H,
            )
        else:
            torch.library.wrap_triton(_mhc_post_mod.mhc_post_kernel_generic)[
                lambda META: (N, hc, (H + META["BLOCK_H"] - 1) // META["BLOCK_H"])
            ](
                a,
                b,
                c,
                d,
                out,
                H=H,
                HC=hc,
            )

    @torch.library.triton_op("flag_gems_pt2::hc_head_fused", mutates_args={"out"})
    def _hc_head_fused_op(
        hs_flat: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out: torch.Tensor,
        hidden_size: int,
        rms_eps: float,
        hc_eps: float,
        hc_mult: int,
    ) -> None:
        """Original HC-head fused Triton kernel; mutates ``out`` in place."""

        num_tokens = hs_flat.shape[0]
        residual_c = hs_flat.contiguous()
        fn_c = fn.contiguous()
        # ``out_c is out`` is the traceable form of the eager launcher's
        # ``out.data_ptr() == out_c.data_ptr()`` check (FakeTensors have no
        # data_ptr).  Same condition, same behavior; ``out`` arrives
        # contiguous from the public wrapper (``empty_like``) in practice.
        out_c = out if out.is_contiguous() else torch.empty_like(out)

        torch.library.wrap_triton(_hc_head_mod._hc_head_fused_kernel)[(num_tokens,)](
            residual_c,
            fn_c,
            hc_scale,
            hc_base,
            out_c,
            num_tokens,
            hidden_size,
            rms_eps,
            hc_eps,
            residual_c.stride(0),
            fn_c.stride(0),
            out_c.stride(0),
            HC=hc_mult,
        )

        if out_c is not out:
            out.copy_(out_c)

else:
    _mhc_pre_op = None
    _mhc_post_op = None
    _hc_head_fused_op = None
    _FN_BF16_CACHE = {}
    _FN_BF16_CAST_CALLS = {"n": 0}


def fn_bf16_cast_count() -> int:
    """Number of fn->bf16 casts actually executed (hot-path observability)."""
    return _FN_BF16_CAST_CALLS["n"]


MHC_PRE_GEMM_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::mhc_pre_gemm",
        kind=CompileKind.OPAQUE_CUSTOM,
        source_kernel=(
            "opaque: data_ptr+_version-keyed fn->bf16 cache + torch.mm "
            "(body runs outside the compiled graph; not Inductor-traceable "
            "by design)"
        ),
        mutates_args=(),
        dynamic_dims=("num_tokens",),
        requires=("torch.library.custom_op",),
    )
)


MHC_PRE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::mhc_pre",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.mhc.mhc_pre.mhc_pre_fused_kernel_hc_mult_4 "
            "(original @triton.autotune object wrapped by wrap_triton; "
            "generic fallback: mhc_pre_generic_kernel)"
        ),
        mutates_args=("post_mix", "comb_mix", "layer_input"),
        dynamic_dims=("num_tokens",),
        requires=(
            "torch.library.triton_op",
            "torch.library.wrap_triton",
            "torch.library.custom_op",
            "flag_gems_pt2::mhc_pre_gemm (OPAQUE_CUSTOM)",
        ),
    )
)

MHC_POST_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::mhc_post",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.mhc.mhc_post.mhc_post_kernel_hc_mult_4 "
            "(original @triton.autotune object wrapped by wrap_triton; "
            "generic fallback: mhc_post_kernel_generic)"
        ),
        mutates_args=("out",),
        dynamic_dims=("num_tokens",),
        requires=("torch.library.triton_op", "torch.library.wrap_triton"),
    )
)

HC_HEAD_FUSED_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::hc_head_fused",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.mhc.hc_head_fused_kernel._hc_head_fused_kernel "
            "(original @triton.autotune object wrapped by wrap_triton)"
        ),
        mutates_args=("out",),
        dynamic_dims=("num_tokens",),
        requires=("torch.library.triton_op", "torch.library.wrap_triton"),
    )
)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiler-visible mhc_pre calling the original FlagGems kernel.

    Supported PT2 builds call the registered ``triton_op`` in both eager and
    compiled execution, so the same JITFunction body runs in both modes.  The
    ``n_splits`` argument is accepted for interface parity with the dispatch
    contract; the original FlagGems implementation reserves it and only
    supports ``n_splits == 1``.
    """

    if n_splits != 1:
        raise NotImplementedError(
            "FlagGems mhc_pre reserves n_splits for a future split implementation; "
            "only n_splits == 1 is supported"
        )
    if _mhc_pre_op is None:
        raise RuntimeError(
            "This Torch build does not provide torch.library.triton_op and "
            "torch.library.wrap_triton; FlagGems mhc_pre cannot enter a PT2 graph"
        )

    outer_shape = residual.shape[:-2]
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]

    residual_flat = residual.reshape(-1, hc_mult, hidden_size).contiguous()
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    _mhc_pre_op(
        residual_flat,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Compiler-visible mhc_post calling the original FlagGems kernel."""

    if _mhc_post_op is None:
        raise RuntimeError(
            "This Torch build does not provide torch.library.triton_op and "
            "torch.library.wrap_triton; FlagGems mhc_post cannot enter a PT2 graph"
        )

    out = torch.empty_like(residual)
    _mhc_post_op(x, residual, post, comb, out)
    return out


def hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Compiler-visible hc_head_fused_kernel; mutates ``out`` in place."""

    if _hc_head_fused_op is None:
        raise RuntimeError(
            "This Torch build does not provide torch.library.triton_op and "
            "torch.library.wrap_triton; FlagGems hc_head_fused_kernel cannot "
            "enter a PT2 graph"
        )

    _hc_head_fused_op(
        hs_flat, fn, hc_scale, hc_base, out, hidden_size, rms_eps, hc_eps, hc_mult
    )


__all__ = [
    "HC_HEAD_FUSED_AUTOTUNER",
    "HC_HEAD_FUSED_SPEC",
    "MHC_POST_GENERIC_AUTOTUNER",
    "MHC_POST_HCMULT4_AUTOTUNER",
    "MHC_POST_SPEC",
    "MHC_PRE_FUSED_AUTOTUNER",
    "MHC_PRE_GENERIC_AUTOTUNER",
    "MHC_PRE_GEMM_SPEC",
    "MHC_PRE_SPEC",
    "fn_bf16_cast_count",
    "hc_head_fused_kernel",
    "mhc_post",
    "mhc_pre",
    "supports_pt2_triton",
]
