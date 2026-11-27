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
from tests import accuracy_utils as utils

mm = importlib.import_module(flag_gems.mm.__module__)
bmm = importlib.import_module(flag_gems.bmm.__module__)

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead" or not getattr(mm, "HAS_PPU_TLE", False),
    reason="requires the T-Head PPU GEMM kernels",
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
OPS = [
    pytest.param(op, marks=getattr(pytest.mark, op))
    for op in ("mm", "addmm", "bmm", "baddbmm")
]
DTYPE = torch.float16


@pytest.fixture(scope="module")
def select_legal_config():
    """Run the production pruner and choose a legal config without tuning."""
    from flag_gems.utils.libentry import LibEntry, LibTuner

    def select(_bench, configs, _args, _kwargs):
        # Exercise AIU loads when alignment allows them, and the pointer path
        # on unaligned NT inputs.
        return min(configs, key=lambda config: config.kwargs.get("LOAD_MODE", 0)), {}

    with pytest.MonkeyPatch.context() as patch, ExitStack() as stack:
        patch.setenv("USE_FLAGTUNE", "0")
        for entry in {
            value
            for module in (mm, bmm)
            for value in vars(module).values()
            if isinstance(value, LibEntry)
        }:
            patch.setattr(entry, "kernel_cache", tuple({} for _ in entry.kernel_cache))
            tuner = entry.fn
            while tuner is not None:
                if isinstance(tuner, LibTuner):
                    patch.setattr(tuner, "policy", select)
                    stack.enter_context(tuner.use_run_mode("force_policy"))
                tuner = getattr(tuner, "fn", None)
        yield


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("nt", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize("op", OPS)
def test_dense_gemm_tails_use_ppu(select_legal_config, monkeypatch, shape, nt, op):
    m, n, k = shape
    batched = op in ("bmm", "baddbmm")
    prefix = (2,) if batched else ()
    a = torch.randn((*prefix, m, k), device=flag_gems.device, dtype=DTYPE)
    b = torch.randn(
        (*prefix, n, k) if nt else (*prefix, k, n),
        device=flag_gems.device,
        dtype=DTYPE,
    )
    if nt:
        b = b.transpose(-1, -2)
    public = getattr(flag_gems, op)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail(f"dense {op} {shape} {'NT' if nt else 'NN'} used generic fallback")

    monkeypatch.setitem(public.__globals__, f"_generic_{op}", unexpected_fallback)
    ref_a, ref_b = utils.to_reference(a), utils.to_reference(b)
    if op in ("addmm", "baddbmm"):
        bias = torch.randn((*prefix, m, n), device=flag_gems.device, dtype=DTYPE)
        reference = getattr(torch, op)(
            utils.to_reference(bias), ref_a, ref_b, alpha=0.5, beta=-0.25
        )
        result = public(bias, a, b, alpha=0.5, beta=-0.25)
    else:
        reference = getattr(torch, op)(ref_a, ref_b)
        result = public(a, b)
    torch.testing.assert_close(
        utils.to_cpu(result, reference), reference, rtol=0.02, atol=0.05
    )


@pytest.mark.mm
@pytest.mark.bmm
@pytest.mark.parametrize("k", [1, 7, 31, 32, 33, 71, 127, 128, 129, 144, 4304])
def test_nt_eligibility_does_not_require_aligned_k(k):
    a = torch.empty((31, k), device="meta", dtype=DTYPE)
    b = torch.empty((65, k), device="meta", dtype=DTYPE).t()
    out = torch.empty((31, 65), device="meta", dtype=DTYPE)
    assert mm._can_use_ppu_mm(a, b, out)
    a = torch.empty((2, 31, k), device="meta", dtype=DTYPE)
    b = torch.empty((2, 65, k), device="meta", dtype=DTYPE).transpose(1, 2)
    assert bmm._can_use_ppu_bmm_inputs(a, b)


@pytest.mark.mm
@pytest.mark.bmm
def test_layout_guards_still_reject_general_strides():
    a = torch.empty((31, 71), device="meta", dtype=DTYPE)
    b = torch.empty((71, 130), device="meta", dtype=DTYPE)[:, ::2]
    out = torch.empty((31, 65), device="meta", dtype=DTYPE)
    assert not mm._can_use_ppu_mm(a, b, out)
    a = torch.empty((2, 31, 71), device="meta", dtype=DTYPE)
    b = torch.empty((2, 71, 130), device="meta", dtype=DTYPE)[:, :, ::2]
    assert not bmm._can_use_ppu_bmm_inputs(a, b)


@pytest.mark.router_gemm
def test_router_gemm_nt_tail_uses_ppu(select_legal_config, monkeypatch):
    # The specialized PPU router takes BF16 inputs and produces FP32 logits.
    a = torch.randn((15, 144), device=flag_gems.device, dtype=torch.bfloat16)
    weight = torch.randn((1025, 144), device=flag_gems.device, dtype=torch.bfloat16)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("dense router_gemm NT K-tail used generic fallback")

    public = flag_gems.router_gemm
    monkeypatch.setitem(public.__globals__, "_generic_router_gemm", unexpected_fallback)
    result = public(a, weight)
    assert result.dtype == torch.float32
    reference = utils.to_reference(a).float() @ utils.to_reference(weight).float().t()
    torch.testing.assert_close(
        utils.to_cpu(result, reference), reference, rtol=0.02, atol=0.05
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("alias", [False, True], ids=["separate-out", "aliased-out"])
def test_nt_tail_out_and_aliasing(select_legal_config, monkeypatch, op, alias):
    prefix = (2,) if op in ("bmm", "baddbmm") else ()
    a = torch.randn((*prefix, 31, 71), device=flag_gems.device, dtype=DTYPE)
    b = torch.randn((*prefix, 71, 71), device=flag_gems.device, dtype=DTYPE)
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
    ref_a, ref_b = utils.to_reference(a), utils.to_reference(b)
    if fused:
        reference = getattr(torch, op)(utils.to_reference(bias), ref_a, ref_b)
        result = public(bias, a, b, out=out)
    else:
        reference = getattr(torch, op)(ref_a, ref_b)
        result = public(a, b, out=out)
    assert result is out
    torch.testing.assert_close(
        utils.to_cpu(result, reference), reference, rtol=0.02, atol=0.05
    )


@pytest.mark.mm
def test_unaligned_rows_prune_aiu_modes_instead_of_the_operator():
    a = torch.empty((31, 31), dtype=DTYPE)
    b = torch.empty((64, 31), dtype=DTYPE).t()
    mask = mm._aiu_load_mask(a, b)
    assert mask == 0  # Neither NT operand has 32-byte-aligned rows.
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
    legal = mm._prune_gemm_configs(
        configs, {"M": 31, "N": 64, "K": 31, "aiu_load_mask": mask}
    )
    assert {config.kwargs["LOAD_MODE"] for config in legal} == {3}


@pytest.mark.mm
def test_split_k_pruning_keeps_partial_slices_interleaved():
    configs = [
        triton.Config(
            {"SPLIT_K": 4, "BLOCK_K": 32, "LOAD_MODE": 3, "INTERLEAVED": interleaved}
        )
        for interleaved in (False, True)
    ]
    for k in (144, 145):
        legal = mm._prune_split_k_configs(configs, {"M": 31, "N": 64, "K": k})
        assert len(legal) == 1 and legal[0].kwargs["INTERLEAVED"]
