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
    with flag_gems.use_gems():
        res_out = torch.all(inp)

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
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.all_dim
@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin",
    reason="regression test for the kunlunxin all_dim bool fallback",
)
@pytest.mark.parametrize("shape", [(100, 33), (100, 5), (64, 257)])
def test_all_dim_bool_byte_fallback(shape, monkeypatch):
    # bool must not reach the pointer kernels' float branch: it mis-loads
    # 1-byte elements as 0 on XPU (an all-True row came back "not all"), and
    # the (64, 257) shape additionally trips the tle compile -- so the
    # fallback is what runs on the default config there.  Force the fallback
    # so the test does not pay that compile.  See all.py::_per_row_all.
    import importlib

    all_ops = importlib.import_module(flag_gems.all_dim.__module__)
    monkeypatch.setattr(all_ops, "_TLE_MIN_AVAILABLE", False)
    inp = torch.ones(shape, dtype=torch.bool, device=flag_gems.device)
    # The official runner drives pytest with `--ref cpu` (TO_CPU=True) and
    # `accuracy_utils.to_cpu` asserts the reference is on cpu -- build it
    # through `utils.to_reference` like the tests above.
    ref = torch.all(utils.to_reference(inp), dim=1)
    with flag_gems.use_gems():
        res = torch.all(inp, dim=1)
    utils.gems_assert_equal(res, ref)


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
    with flag_gems.use_gems():
        res_out = torch.all(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
