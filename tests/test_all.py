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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, 1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KIND_KEEPDIM_DIMS_SHAPE = list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            utils.REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )


@pytest.mark.all
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_all(shape, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp)
    res_out = flag_gems.all(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("kind", ["normal", "allTrue"])
def test_all_dim(shape, dtype, keepdim, dim, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.all_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dims
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_all_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allTrue":
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.all(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.all_dims(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dim
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon vendor override")
class TestHygonAllDim:
    @staticmethod
    def _module():
        import importlib

        return importlib.import_module(flag_gems.all_dim.__module__)

    @staticmethod
    def _check(inp, dim, keepdim, op=None):
        cpu = inp.cpu()
        before = cpu.clone()
        expected = torch.all(cpu, dim=dim, keepdim=keepdim)
        result = (op or flag_gems.all_dim)(inp, dim=dim, keepdim=keepdim)
        assert result.device == inp.device
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert not result.requires_grad
        assert not torch._C._is_alias_of(result, inp)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(inp.cpu(), before, rtol=0, atol=0, equal_nan=True)
        return result

    @pytest.mark.parametrize(
        "shape",
        [
            (128,),
            (5, 129),
            (17, 255),
            (256, 256),
            (8, 33, 513),
            (1024, 1024),
            (65536, 128),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("positive_dim", [False, True])
    def test_fast_accuracy(self, shape, keepdim, positive_dim):
        generator = torch.Generator().manual_seed(20261104)
        cpu = torch.randint(1, 9, shape, generator=generator).float()
        cpu.reshape(-1, shape[-1])[::4, -1] = 0
        inp = cpu.to(flag_gems.device)
        dim = inp.ndim - 1 if positive_dim else -1
        assert self._module()._can_use_all_dim_fast_path(inp, dim, keepdim)
        self._check(inp, dim, keepdim)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_truth_and_padding(self, keepdim):
        cpu = torch.ones((11, 129), dtype=torch.float32)
        # All-true rows, signed zeros, NaNs, infinities and the smallest signed
        # FP32 subnormals; the last group has three real rows and one padding row.
        cpu[0].fill_(float("nan"))
        cpu[1].fill_(float("inf"))
        cpu[2].fill_(-float("inf"))
        cpu[3].fill_(0.0)
        cpu[4, -1] = -0.0
        tiny = torch.tensor([1, -2147483647], dtype=torch.int32).view(torch.float32)
        cpu[5].fill_(tiny[0])
        cpu[6].fill_(tiny[1])
        cpu[7, -1] = float("nan")
        cpu[8, 0] = 0
        cpu[9, -1] = 0
        self._check(cpu.to(flag_gems.device), -1, keepdim)

    @pytest.mark.parametrize(
        "shape,dtype,dim,layout",
        [
            ((7, 127), torch.float32, -1, "contiguous"),
            ((7, 1025), torch.float32, -1, "contiguous"),
            ((7, 128), torch.float32, 0, "contiguous"),
            ((7, 128), torch.float16, -1, "contiguous"),
            ((7, 128), torch.bfloat16, -1, "contiguous"),
            ((7, 128), torch.bool, -1, "contiguous"),
            ((7, 256), torch.float32, -1, "stride2"),
            ((128, 7), torch.float32, -1, "transpose"),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_fallback_accuracy(self, shape, dtype, dim, layout, keepdim):
        cpu = torch.ones(shape, dtype=dtype)
        cpu.reshape(-1)[-1] = 0
        inp = cpu.to(flag_gems.device)
        if layout == "stride2":
            inp = inp[:, ::2]
        elif layout == "transpose":
            inp = inp.T
        assert not self._module()._can_use_all_dim_fast_path(inp, dim, keepdim)
        self._check(inp, dim, keepdim)

    @pytest.mark.parametrize("dim", [None, -1, 1])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_offset_alias_and_dispatch(self, dim, keepdim):
        # The data pointer already includes the nonzero storage offset.
        backing = torch.ones(7 + 5 * 129, device=flag_gems.device)
        inp = backing[7:].view(5, 129)
        inp[2, -1] = 0
        self._check(inp, dim, keepdim)
        with flag_gems.use_gems():
            actual = torch.all(inp, dim=dim, keepdim=keepdim)
        torch.testing.assert_close(
            actual.cpu(), torch.all(inp.cpu(), dim=dim, keepdim=keepdim), rtol=0, atol=0
        )
        # Positive/negative dimension aliases must agree, without output aliasing.
        if dim is not None:
            self._check(inp, -1, keepdim)
            self._check(inp, 1, keepdim)

    def test_formal_export_and_full_reduction(self):
        from flag_gems.ops.all import all as generic_all
        from flag_gems.ops.all import all_dims as generic_all_dims

        assert flag_gems.all_dim is self._module().all_dim
        assert flag_gems.all is generic_all
        assert flag_gems.all_dims is generic_all_dims
        inp = torch.ones((5, 129), device=flag_gems.device)
        torch.testing.assert_close(
            flag_gems.all(inp).cpu(), torch.all(inp.cpu()), rtol=0, atol=0
        )
        for dim in ([1], (-1,), [], [0, 1]):
            self._check(inp, dim, True, op=flag_gems.all_dims)

    @pytest.mark.parametrize("dim", [-3, 2])
    def test_invalid_dim_falls_back(self, dim):
        inp = torch.ones((5, 128), device=flag_gems.device)
        assert not self._module()._can_use_all_dim_fast_path(inp, dim, False)
        with pytest.raises((AssertionError, IndexError)):
            flag_gems.all_dim(inp, dim=dim)

    def test_unresolved_negative_view(self):
        inp = torch.ones((5, 129), device=flag_gems.device)._neg_view()
        assert not self._module()._can_use_all_dim_fast_path(inp, -1, False)
        self._check(inp, -1, False)

    @pytest.mark.parametrize(
        "shape,dim,keepdim,dtype",
        [
            ((3, 0), -1, False, torch.float32),
            ((0, 128), -1, True, torch.float32),
            ((), 0, False, torch.float32),
            ((5, 128), -1, False, torch.uint8),
            ((5, 128), None, True, torch.float32),
        ],
    )
    def test_excluded_semantics_delegate(self, monkeypatch, shape, dim, keepdim, dtype):
        # Even known baseline defects must reach the captured generic function.
        module = self._module()
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
        sentinel = object()
        calls = []

        def fallback(value, dim=None, keepdim=False):
            calls.append((value, dim, keepdim))
            return sentinel

        monkeypatch.setattr(module, "generic_all_dim", fallback)
        assert not module._can_use_all_dim_fast_path(inp, dim, keepdim)
        assert flag_gems.all_dim(inp, dim=dim, keepdim=keepdim) is sentinel
        assert len(calls) == 1
        assert calls[0][0] is inp
        assert calls[0][1:] == (dim, keepdim)

    @pytest.mark.parametrize(
        "shape,dim,dtype",
        [
            pytest.param(
                (3, 0),
                -1,
                torch.float32,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic divides by N=0; upstream #8412 is separate",
                ),
            ),
            pytest.param(
                (),
                0,
                torch.float32,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic rejects scalar dim=0; separate semantics",
                ),
            ),
            pytest.param(
                (5, 128),
                -1,
                torch.uint8,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Frozen generic returns bool instead of uint8; separate semantics",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_known_baseline_semantics(self, shape, dim, dtype, keepdim):
        # Keep CPU correctness assertions visible; --runxfail exposes defects.
        inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
        self._check(inp, dim, keepdim)

    def test_empty_full_reduction(self):
        inp = torch.empty((3, 0), device=flag_gems.device)
        self._check(inp, None, False)
        self._check(inp, None, True)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_empty_spectator(self, keepdim):
        inp = torch.empty((0, 128), device=flag_gems.device)
        assert not self._module()._can_use_all_dim_fast_path(inp, -1, keepdim)
        self._check(inp, -1, keepdim)

    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((5, 127), False),
            ((5, 128), True),
            ((5, 1024), True),
            ((5, 1025), False),
            ((65536, 128), True),
            ((65537, 128), False),
            ((1, 1, 1, 1, 1, 1, 1, 128), True),
            ((1, 1, 1, 1, 1, 1, 1, 1, 128), False),
        ],
    )
    def test_guard_boundaries(self, shape, expected):
        # Metadata-only tensors avoid allocating the excluded upper-bound case.
        inp = torch.empty(shape, dtype=torch.float32, device="meta")
        assert self._module()._can_use_all_dim_fast_path(inp, -1, False) == expected

    @pytest.mark.xfail(
        strict=True,
        reason="Frozen generic full all ignores strides; upstream #7158 is separate",
    )
    def test_known_full_strided_semantics(self):
        backing = torch.zeros((2, 60), device=flag_gems.device)
        backing[:, -1] = 1
        inp = backing[:, -1]
        expected = torch.all(inp.cpu())
        result = flag_gems.all(inp)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.xfail(
        strict=True,
        reason="Frozen generic full all returns bool for uint8; separate semantics",
    )
    def test_known_full_uint8_semantics(self):
        inp = torch.ones((5, 128), dtype=torch.uint8, device=flag_gems.device)
        expected = torch.all(inp.cpu())
        result = flag_gems.all(inp)
        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_fast_launch_is_selected(self, monkeypatch):
        module = self._module()
        real_kernel = module._all_f32_rows
        grids = []

        class RecordingKernel:
            def __getitem__(self, grid):
                grids.append(grid)
                return real_kernel[grid]

        monkeypatch.setattr(module, "_all_f32_rows", RecordingKernel())
        inp = torch.ones((5, 129), device=flag_gems.device)
        self._check(inp, -1, True)
        assert grids == [(2,)]
