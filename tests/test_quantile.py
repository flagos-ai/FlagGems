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
    DIM_LIST = [0]
    KEEPDIM = [True]
    QUANTILE_Q = [(0.2, 0.5, 0.8)]
    QUANTILE_INTERPOLATION = ["linear"]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]
    QUANTILE_Q = [
        (0.4),
        (0.0, 0.2, 0.5, 0.8, 1.0),
        (0.662, 0.8, 0.104, 0.99, 0.347, 0.255),
    ]
    QUANTILE_INTERPOLATION = ["linear", "lower", "higher", "nearest", "midpoint"]

QUANTILE_SHAPES = utils.REDUCTION_SMALL_SHAPES + [(10, 64, 196), (65535, 1)]
QUANTILE_FLOAT_DTYPES = [torch.float32]


@pytest.mark.quantile
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"), reason="Triton has to be version 3.0+."
)
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_quantile(shape, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = utils.to_reference(q)

    ref_out = torch.quantile(ref_inp, ref_q, interpolation=interpolation)
    res_out = flag_gems.quantile(inp, q, interpolation=interpolation)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.quantile
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"), reason="Triton has to be 3.0+."
)
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_quantile_dim(shape, dim, keepdim, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = utils.to_reference(q)

    ref_out = torch.quantile(
        ref_inp, ref_q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    res_out = flag_gems.quantile(
        inp, q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.quantile
@pytest.mark.skipif(not utils.fp64_is_supported, reason="requires float64")
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
def test_quantile_fp64_rank(interpolation, caplog):
    inp = torch.full((1024,), 1e12, device=flag_gems.device, dtype=torch.float64)
    inp[:512] = -1e12
    # Straddle the jump in the sorted data at integer and half-integer ranks.
    q = inp.new_tensor(
        [
            (511 + 1e-6) / 1023,
            0.5 - 2**-28,
            0.5 + 2**-53,
            0.5 + 2**-28,
            (512 - 1e-6) / 1023,
        ]
    )
    ref = torch.quantile(
        utils.to_reference(inp), utils.to_reference(q), interpolation=interpolation
    )
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.quantile)):
        result = flag_gems.quantile(inp, q, interpolation=interpolation)
    assert f"{utils.gems_log_prefix(flag_gems.quantile)} QUANTILE" in caplog.text
    utils.gems_assert_close(result, ref, torch.float64, atol=1e-12)


@pytest.mark.quantile
@pytest.mark.skipif(not utils.fp64_is_supported, reason="requires float64")
def test_quantile_fp64_bitonic_padding(caplog):
    inp = torch.arange(1, 32, dtype=torch.float64, device=flag_gems.device)
    q = inp.new_tensor([0.0, 0.5, 1.0])
    ref = torch.quantile(utils.to_reference(inp), utils.to_reference(q))
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.quantile)):
        result = flag_gems.quantile(inp, q)
    assert f"{utils.gems_log_prefix(flag_gems.quantile)} QUANTILE" in caplog.text
    utils.gems_assert_equal(result, ref)
