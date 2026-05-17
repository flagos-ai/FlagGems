#!/usr/bin/env python3
"""Standalone smoke for flag_gems.chunk_gated_delta_rule.

Run from the repository root after installing the package, for example:

    PYTHONPATH=src python3 chunk_gated_delta_rule_smoke.py

The probe verifies direct and chunk/FLA routing, q/k L2 normalization,
initial_state, and packed varlen cu_seqlens.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton


@dataclass(frozen=True)
class Case:
    name: str
    b: int
    t: int
    hg: int
    h: int
    k: int
    v: int
    dtype: torch.dtype
    head_first: bool = False
    initial_state: bool = False
    output_final_state: bool = False
    cu_seqlens: tuple[int, ...] | None = None
    qk_norm: bool = False
    expected_path: str = "chunk"


def install_triton_allocator(device: str) -> None:
    if (
        device != "cuda"
        or not torch.cuda.is_available()
        or not hasattr(triton, "set_allocator")
    ):
        return

    def alloc(size: int, _alignment: int, _stream: int | None):
        return torch.empty((size,), dtype=torch.uint8, device=device)

    triton.set_allocator(alloc)


def stable_decay(
    shape: tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    decay = (
        torch.empty(shape, device=device, dtype=torch.float32)
        .uniform_(-4.605170185988091, -3.506557897319982)
        .exp()
    )
    return torch.log1p(-decay).to(dtype)


def stable_beta(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    return (
        torch.empty(shape, device=device, dtype=torch.float32)
        .uniform_(-2.0, 2.0)
        .sigmoid()
        .to(dtype)
    )


def public_layout(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2) if head_first else x


def seq_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2) if head_first else x


def make_inputs(case: Case, device: str):
    q = torch.randn(
        case.b, case.t, case.hg, case.k, device=device, dtype=torch.float32
    ).to(case.dtype)
    k = F.normalize(
        torch.randn(
            case.b, case.t, case.hg, case.k, device=device, dtype=torch.float32
        ),
        p=2.0,
        dim=-1,
        eps=1e-6,
    ).to(case.dtype)
    v = (
        0.125
        * torch.randn(
            case.b, case.t, case.h, case.v, device=device, dtype=torch.float32
        )
    ).to(case.dtype)
    beta = stable_beta((case.b, case.t, case.h), case.dtype, device)
    g = stable_decay((case.b, case.t, case.h), case.dtype, device)
    return (
        public_layout(q, case.head_first),
        public_layout(k, case.head_first),
        public_layout(v, case.head_first),
        public_layout(beta, case.head_first),
        public_layout(g, case.head_first),
    )


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    head_first: bool,
    scale: float | None,
    qk_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q = seq_first(q, head_first).float()
    k = seq_first(k, head_first).float()
    v_seq = seq_first(v, head_first)
    v_float = v_seq.float()
    beta = seq_first(beta, head_first).float()
    g = seq_first(g, head_first).float()
    if qk_norm:
        q = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2.0, dim=-1, eps=1e-6)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    bsz, t_total, hg, kk = q.shape
    heads, vv = v_seq.shape[2], v_seq.shape[3]
    heads_per_group = heads // hg
    out = torch.empty_like(v_float)
    state_count = bsz if cu_seqlens is None else cu_seqlens.numel() - 1
    final_state = (
        torch.empty(state_count, heads, kk, vv, device=v.device, dtype=torch.float32)
        if output_final_state
        else None
    )

    if cu_seqlens is None:
        spans = [(b, b, 0, t_total) for b in range(bsz)]
    else:
        cu_cpu = cu_seqlens.detach().cpu().tolist()
        spans = [(0, n, cu_cpu[n], cu_cpu[n + 1]) for n in range(len(cu_cpu) - 1)]

    for batch_idx, state_idx, start, end in spans:
        if initial_state is None:
            state = torch.zeros(heads, kk, vv, device=v.device, dtype=torch.float32)
        else:
            state = initial_state[state_idx].float().clone()
        for pos in range(start, end):
            for hv in range(heads):
                hg_idx = hv // heads_per_group
                state[hv] *= torch.exp(g[batch_idx, pos, hv])
                key = k[batch_idx, pos, hg_idx]
                residual = v_float[batch_idx, pos, hv] - torch.matmul(key, state[hv])
                update = residual * beta[batch_idx, pos, hv]
                state[hv] += key[:, None] * update[None, :]
                out[batch_idx, pos, hv] = torch.matmul(
                    q[batch_idx, pos, hg_idx] * scale, state[hv]
                )
        if output_final_state:
            final_state[state_idx] = state

    if head_first:
        out = out.transpose(1, 2)
    return out.to(v.dtype), final_state


def assert_close(
    name: str, actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> None:
    if dtype == torch.float32:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        check_dtype=False,
        msg=lambda msg: f"{name} mismatch\n{msg}",
    )


def wrap_paths(module) -> tuple[dict[str, int], Callable[[], None]]:
    counts = {"direct": 0, "chunk": 0}
    original_direct = module.chunk_gated_delta_rule_direct_fwd
    original_chunk = module.chunk_gated_delta_rule_fwd

    def direct_wrapper(*args, **kwargs):
        counts["direct"] += 1
        return original_direct(*args, **kwargs)

    def chunk_wrapper(*args, **kwargs):
        counts["chunk"] += 1
        return original_chunk(*args, **kwargs)

    module.chunk_gated_delta_rule_direct_fwd = direct_wrapper
    module.chunk_gated_delta_rule_fwd = chunk_wrapper

    def restore() -> None:
        module.chunk_gated_delta_rule_direct_fwd = original_direct
        module.chunk_gated_delta_rule_fwd = original_chunk

    return counts, restore


def run_case(case: Case, flag_gems, counts: dict[str, int], device: str) -> None:
    before = dict(counts)
    q, k, v, beta, g = make_inputs(case, device)
    cu_seqlens = (
        torch.tensor(case.cu_seqlens, device=device, dtype=torch.long)
        if case.cu_seqlens is not None
        else None
    )
    state_count = case.b if cu_seqlens is None else cu_seqlens.numel() - 1
    initial_state = (
        (
            0.125
            * torch.randn(
                state_count,
                case.h,
                case.k,
                case.v,
                device=device,
                dtype=torch.float32,
            )
        ).to(case.dtype)
        if case.initial_state
        else None
    )
    scale = 1.0 / math.sqrt(case.k)

    actual, actual_state = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        BT=64,
        initial_state=initial_state,
        output_final_state=case.output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=case.head_first,
        scale=scale,
        use_qk_l2norm_in_kernel=case.qk_norm,
    )
    expected, expected_state = reference(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=case.output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=case.head_first,
        scale=scale,
        qk_norm=case.qk_norm,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    assert_close(case.name, actual, expected, case.dtype)
    if case.output_final_state:
        if actual_state is None or expected_state is None:
            raise AssertionError(f"{case.name}: expected final_state")
        assert_close(
            f"{case.name}.final_state", actual_state, expected_state, torch.float32
        )
    elif actual_state is not None:
        raise AssertionError(f"{case.name}: unexpected final_state")

    delta = {path: counts[path] - before[path] for path in counts}
    if delta[case.expected_path] != 1:
        raise AssertionError(
            f"{case.name}: expected {case.expected_path} path once, saw delta={delta}"
        )
    other = "chunk" if case.expected_path == "direct" else "direct"
    if delta[other] != 0:
        raise AssertionError(f"{case.name}: unexpected {other} path, saw delta={delta}")
    shape = (case.b, case.t, case.hg, case.h, case.k, case.v)
    print(f"PASS {case.name}: path={case.expected_path} shape={shape} dtype={case.dtype}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    import flag_gems

    device = args.device or flag_gems.device
    if device == "cuda" and not torch.cuda.is_available():
        print("SKIP: CUDA is not available", file=sys.stderr)
        return 77
    install_triton_allocator(device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    module = importlib.import_module("flag_gems.fused.chunk_gated_delta_rule")
    counts, restore = wrap_paths(module)
    print(
        "ENV "
        f"device={device} vendor={getattr(flag_gems, 'vendor_name', '<unknown>')} "
        f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}"
    )
    if device == "cuda":
        print(f"ENV gpu={torch.cuda.get_device_name(0)}")

    cases = [
        Case(
            "direct_basic",
            1,
            32,
            2,
            4,
            32,
            16,
            torch.float32,
            True,
            expected_path="direct",
        ),
        Case(
            "direct_qk_norm",
            1,
            32,
            2,
            4,
            32,
            16,
            torch.float32,
            True,
            qk_norm=True,
            expected_path="direct",
        ),
        Case(
            "chunk_large_t",
            1,
            129,
            2,
            4,
            32,
            16,
            torch.float32,
            False,
            expected_path="chunk",
        ),
        Case(
            "chunk_initial_state",
            2,
            33,
            2,
            4,
            32,
            16,
            torch.float32,
            False,
            True,
            True,
            expected_path="chunk",
        ),
        Case(
            "chunk_qk_norm_initial_state",
            1,
            33,
            2,
            4,
            32,
            16,
            torch.float32,
            True,
            True,
            True,
            qk_norm=True,
            expected_path="chunk",
        ),
        Case(
            "chunk_cu_seqlens",
            1,
            80,
            2,
            4,
            32,
            16,
            torch.float16,
            False,
            False,
            True,
            (0, 17, 80),
            expected_path="chunk",
        ),
    ]

    try:
        for case in cases:
            run_case(case, flag_gems, counts, device)
    finally:
        restore()

    print(f"SUMMARY PASS counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
