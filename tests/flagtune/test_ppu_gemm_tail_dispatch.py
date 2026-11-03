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

"""Dense GEMM tails must stay on the masked PPU kernels, in NN and NT."""

import importlib
from contextlib import ExitStack

import pytest
import torch
import triton

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead", reason="requires the PPU GEMM backend"
)

SHAPES = [
    (1, 33, 71),
    (2, 129, 129),
    (15, 1025, 144),
    (31, 65, 143),
    (65, 31, 255),
    (129, 257, 272),
    (495, 533, 71),
    (128, 128, 4304),
    (640, 64, 8208),
    (160, 4, 8209),
    (128, 128, 128),
]
OP_DTYPES = [
    pytest.param(op, dtype, id=f"{op}-{dtype}", marks=getattr(pytest.mark, op))
    for op in ("mm", "addmm", "bmm", "baddbmm")
    for dtype in (torch.bfloat16, torch.float16)
] + [
    pytest.param(
        "router_gemm",
        torch.bfloat16,
        id="router_gemm-bf16",
        marks=pytest.mark.router_gemm,
    )
]


@pytest.fixture(scope="module", autouse=True)
def short_tuning_protocol():
    """Exercise natural candidate selection without a full performance run."""
    lib = importlib.import_module("flag_gems.utils.libentry")
    with pytest.MonkeyPatch.context() as patch, ExitStack() as stack:
        patch.setenv("USE_FLAGTUNE", "0")
        entries = {
            id(value): value
            for public in (flag_gems.mm, flag_gems.bmm)
            for value in public.__globals__.values()
            if isinstance(value, lib.LibEntry)
        }
        tuners = {}
        for entry in entries.values():
            entry._apply_flagtune()
            fn = entry.fn
            while fn is not None:
                if isinstance(fn, lib.LibTuner):
                    tuners[id(fn)] = fn
                fn = getattr(fn, "fn", None)
        for tuner in tuners.values():
            original_prune = tuner.prune_configs

            def representative_configs(kwargs, prune=original_prune):
                configs = prune(kwargs)

                # Cover each legal load mode and split layout. Kernel-tail
                # correctness does not require a full performance autotune.
                def tile_size(config):
                    meta = config.kwargs
                    return (
                        int(meta.get("BLOCK_M", 1))
                        * int(meta.get("BLOCK_N", 1))
                        * int(meta.get("BLOCK_K", meta.get("BLOCK", 1)))
                    )

                representatives = {}
                for config in sorted(configs, key=tile_size):
                    key = (
                        config.kwargs.get("LOAD_MODE"),
                        config.kwargs.get("INTERLEAVED"),
                    )
                    representatives.setdefault(key, config)
                return list(representatives.values())

            patch.setattr(tuner, "prune_configs", representative_configs)
            stack.enter_context(
                tuner.use_benchmark_protocol(
                    "replay", warmup=0, rep=3, benchmark_retries=3
                )
            )
        yield
        for entry in entries.values():
            lib.clear_libentry_dispatch_cache(entry)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("nt", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize("op,dtype", OP_DTYPES)
def test_dense_gemm_tails_use_ppu(monkeypatch, shape, nt, op, dtype):
    m, n, k = shape
    batched = op in ("bmm", "baddbmm")
    prefix = (2,) if batched else ()
    a = torch.randn((*prefix, m, k), device=flag_gems.device, dtype=dtype)
    b = torch.randn(
        (*prefix, n, k) if nt else (*prefix, k, n),
        device=flag_gems.device,
        dtype=dtype,
    )
    if nt:
        b = b.transpose(-1, -2)
    public = getattr(flag_gems, op)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail(f"dense {op} {shape} {'NT' if nt else 'NN'} used generic fallback")

    monkeypatch.setitem(public.__globals__, f"_generic_{op}", unexpected_fallback)
    if op in ("addmm", "baddbmm"):
        bias = torch.randn((*prefix, m, n), device=flag_gems.device, dtype=dtype)
        reference = getattr(torch, op)(bias, a, b, alpha=0.5, beta=-0.25)
        result = public(bias, a, b, alpha=0.5, beta=-0.25)
    elif op == "router_gemm":
        reference = a.float() @ b.float()
        result = public(a, b.t())
    else:
        reference = getattr(torch, op)(a, b)
        result = public(a, b)
    torch.testing.assert_close(result.float(), reference.float(), rtol=0.02, atol=0.05)


@pytest.mark.mm
@pytest.mark.bmm
@pytest.mark.parametrize("k", [1, 7, 31, 32, 33, 71, 127, 128, 129, 144, 4304])
def test_nt_eligibility_does_not_require_aligned_k(k):
    mm = flag_gems.mm.__globals__
    bmm = flag_gems.bmm.__globals__
    a = torch.empty((31, k), device="meta", dtype=torch.bfloat16)
    b = torch.empty((65, k), device="meta", dtype=torch.bfloat16).t()
    out = torch.empty((31, 65), device="meta", dtype=torch.bfloat16)
    assert mm["_can_use_ppu_mm"](a, b, out)
    a = torch.empty((2, 31, k), device="meta", dtype=torch.bfloat16)
    b = torch.empty((2, 65, k), device="meta", dtype=torch.bfloat16).transpose(1, 2)
    assert bmm["_can_use_ppu_bmm_inputs"](a, b)


@pytest.mark.mm
@pytest.mark.bmm
def test_layout_guards_still_reject_general_strides():
    mm = flag_gems.mm.__globals__
    bmm = flag_gems.bmm.__globals__
    a = torch.empty((31, 71), device="meta", dtype=torch.bfloat16)
    b = torch.empty((71, 130), device="meta", dtype=torch.bfloat16)[:, ::2]
    out = torch.empty((31, 65), device="meta", dtype=torch.bfloat16)
    assert not mm["_can_use_ppu_mm"](a, b, out)
    a = torch.empty((2, 31, 71), device="meta", dtype=torch.bfloat16)
    b = torch.empty((2, 71, 130), device="meta", dtype=torch.bfloat16)[:, :, ::2]
    assert not bmm["_can_use_ppu_bmm_inputs"](a, b)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=getattr(pytest.mark, op))
        for op in ("mm", "addmm", "bmm", "baddbmm")
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("alias", [False, True], ids=["separate-out", "aliased-out"])
def test_nt_tail_out_and_aliasing(monkeypatch, op, dtype, alias):
    prefix = (2,) if op in ("bmm", "baddbmm") else ()
    a = torch.randn((*prefix, 31, 71), device=flag_gems.device, dtype=dtype)
    b = torch.randn((*prefix, 71, 71), device=flag_gems.device, dtype=dtype)
    b = b.transpose(-1, -2)
    fused = op in ("addmm", "baddbmm")
    bias = torch.randn_like(a) if fused else None
    out = (bias if fused else a) if alias else torch.empty_like(a)
    public = getattr(flag_gems, op + "_out")

    def unexpected_fallback(*args, **kwargs):
        pytest.fail(f"dense {op}_out NT K-tail used generic fallback")

    monkeypatch.setitem(public.__globals__, f"_generic_{op}_out", unexpected_fallback)
    monkeypatch.setitem(
        getattr(flag_gems, op).__globals__, f"_generic_{op}", unexpected_fallback
    )
    if fused:
        reference = getattr(torch, op)(bias, a, b)
        result = public(bias, a, b, out=out)
    else:
        reference = getattr(torch, op)(a, b)
        result = public(a, b, out=out)
    assert result is out
    torch.testing.assert_close(result.float(), reference.float(), rtol=0.02, atol=0.05)


@pytest.mark.mm
def test_unaligned_rows_prune_aiu_modes_instead_of_the_operator():
    helpers = flag_gems.mm.__globals__["_aiu_load_mask"].__globals__
    a = torch.empty((31, 31), dtype=torch.bfloat16)
    b = torch.empty((31, 64), dtype=torch.bfloat16)
    mask = helpers["_aiu_load_mask"](a, b)
    assert mask == 2  # Only B's row starts are 32-byte aligned.
    configs = [
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "LOAD_MODE": mode,
                "PIPE_STAGES": 3,
            },
            num_warps=4,
        )
        for mode in range(4)
    ]
    legal = helpers["_prune_gemm_configs"](
        configs, {"M": 31, "N": 64, "K": 31, "aiu_load_mask": mask}
    )
    assert {config.kwargs["LOAD_MODE"] for config in legal} == {2, 3}


@pytest.mark.mm
def test_split_k_pruning_keeps_partial_slices_interleaved():
    helpers = flag_gems.mm.__globals__["_prune_split_k_configs"].__globals__
    configs = [
        triton.Config(
            {"SPLIT_K": 4, "BLOCK_K": 32, "LOAD_MODE": 3, "INTERLEAVED": interleaved}
        )
        for interleaved in (False, True)
    ]
    for k in (144, 145):
        legal = helpers["_prune_split_k_configs"](configs, {"M": 31, "N": 64, "K": k})
        assert len(legal) == 1 and legal[0].kwargs["INTERLEAVED"]
