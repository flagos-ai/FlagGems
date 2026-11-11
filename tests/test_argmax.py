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
    DIM_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
else:
    DIM_LIST = [0, 1]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

EMPTY_SHAPES = [(0, 5), (3, 0, 4), (2, 5, 0), (0,)]


# TODO: There are some bugs in argmax with large size.
@pytest.mark.argmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES + EMPTY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax(shape, dim, keepdim, dtype):
    rank = len(shape)
    is_empty_tensor = any(d == 0 for d in shape)

    if dim is not None:
        if rank == 0 or dim >= rank or dim < -rank:
            # Skip invalid input combination - dimension out of bound for shape
            return

    if is_empty_tensor:
        if dim is None:
            # The dim parameter must be specified for empty tensor for PyTorch
            return

        dim_index = dim % rank
        if shape[dim_index] == 0:
            # Zero-sized dimension is invalid input for PyTorch
            return

    if is_empty_tensor:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    if flag_gems.vendor_name == "ascend" and dim is None and keepdim:
        # torch_npu 2.10 flattens the retained output dimensions on this path.
        ref_out = ref_out.reshape([1] * inp.ndim)
    res_out = flag_gems.argmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.argmax
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend regression")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "large",
        "second_stage",
        "transpose",
        "broadcast",
        "nan",
        "ties",
        "minimum",
        "grid_limit",
        "global_nan",
        "global_split_nan",
        "global_split_ties",
        "global_split_minimum",
        "scalar",
        "strided_nan",
        "strided_ties",
        "strided_minimum",
        "long_row_nan",
        "long_strided_nan",
        "long_row_ties",
        "long_strided_ties",
    ],
)
def test_argmax_ascend_regression(dtype, keepdim, case):
    dim = None
    if case == "large":
        cpu = torch.randn((200, 2560, 3), dtype=dtype)
    elif case == "second_stage":
        cpu = torch.zeros(1024 * 1024 + 17, dtype=dtype)
        cpu[-1] = 5
    elif case == "grid_limit":
        cpu = torch.randn((65, 2, 65536), dtype=dtype)
        dim = 1
    elif case.startswith("global_split_"):
        cpu = torch.full((1024 * 1024 + 17,), float("-inf"), dtype=dtype)
        if case != "global_split_minimum":
            value = float("nan") if case == "global_split_nan" else 5
            cpu[1027] = value
            cpu[12307] = value
            cpu[-1] = value
    elif case == "global_nan":
        cpu = torch.zeros(1024 * 1024 + 17, dtype=dtype)
        cpu[1024 * 1024 + 3] = float("nan")
        cpu[-1] = float("nan")
    elif case in ("long_row_nan", "long_row_ties"):
        cpu = torch.zeros((3, 8195), dtype=dtype)
        value = float("nan") if case == "long_row_nan" else 5
        cpu[:, 4101] = value
        cpu[:, 8194] = value
        dim = 1
    elif case in ("long_strided_nan", "long_strided_ties"):
        cpu = torch.zeros((3, 1027, 37), dtype=dtype)
        value = float("nan") if case == "long_strided_nan" else 5
        cpu[:, 517, :] = value
        cpu[:, 1026, :] = value
        dim = 1
    elif case.startswith("strided_"):
        cpu = torch.full((3, 257, 37), float("-inf"), dtype=dtype)
        dim = 1
        if case == "strided_nan":
            cpu[:, 129, :] = float("nan")
            cpu[:, 256, :] = float("nan")
        elif case == "strided_ties":
            cpu[:, 127, :] = 5
            cpu[:, 256, :] = 5
    elif case == "transpose":
        cpu = torch.arange(65 * 131, dtype=torch.float32).reshape(65, 131).to(dtype)
    elif case == "broadcast":
        cpu = torch.tensor([[1, 3, 2]], dtype=dtype)
    elif case in ("nan", "ties", "minimum"):
        cpu = torch.full((3, 1027), float("-inf"), dtype=dtype)
        dim = 1
        if case == "nan":
            cpu[:, 517] = float("nan")
            cpu[:, 1026] = float("nan")
        elif case == "ties":
            cpu[:, 511] = 5
            cpu[:, 1026] = 5
    else:
        cpu = torch.tensor(4, dtype=dtype)
        dim = -1
    inp = cpu.to(flag_gems.device)
    if case == "transpose":
        cpu, inp = cpu.t(), inp.t()
    elif case == "broadcast":
        cpu, inp = cpu.expand(7, 3), inp.expand(7, 3)
    expected = torch.argmax(cpu, dim=dim, keepdim=keepdim)
    actual = flag_gems.argmax(inp, dim=dim, keepdim=keepdim)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.argmax
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend regression")
@pytest.mark.parametrize("dim", [None, 1])
def test_argmax_ascend_empty_reduction(dim):
    inp = torch.empty((2, 0), device=flag_gems.device)
    with pytest.raises(IndexError):
        flag_gems.argmax(inp, dim=dim)


@pytest.mark.argmax
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend regression")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(65, 16385), (129, 65539), (65, 1, 257)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_argmax_ascend_long_rows(dtype, shape, keepdim):
    cpu = torch.full(shape, float("-inf"), dtype=dtype)
    if shape[1] > 1:
        # Cover lane scans, non-adjacent ties/NaNs, tails, and empty maxima.
        cpu[0, 17] = cpu[0, -1] = float("nan")
        cpu[1, 4113] = cpu[1, -1] = 5
        cpu[2, 17] = -0.0
        cpu[2, -1] = 0.0
        cpu[3, -1] = float("inf")
    actual = flag_gems.argmax(cpu.to(flag_gems.device), dim=1, keepdim=keepdim)
    expected = torch.argmax(cpu, dim=1, keepdim=keepdim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
