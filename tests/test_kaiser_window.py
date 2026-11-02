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

# The upper bound is 4096 on purpose. torch does not compute the same
# float16/bfloat16 Kaiser window on CPU and on CUDA once the window gets long:
# the element index stops being representable in the output dtype (bfloat16 can
# only hold integers up to 256 exactly, float16 up to 2048), and the two
# backends round the intermediate results differently. From 8192 on, torch's own
# CPU and CUDA kernels disagree by roughly 5x the CI tolerance for bfloat16, so
# no implementation can satisfy both the default and the `--ref cpu` CI mode
# there. Staying at or below 4096 keeps every case meaningful in both modes.
WINDOW_LENGTHS = [
    0,
    1,
    2,
    3,
    7,
    8,
    16,
    17,
    33,
    64,
    100,
    128,
    129,
    200,
    256,
    333,
    512,
    1000,
    1024,
    2048,
    4096,
]


@pytest.mark.kaiser_window
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_kaiser_window(window_length, dtype):
    # periodic defaults to True and beta to 12.0; exercises the plain
    # aten::kaiser_window overload.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.kaiser_window(window_length, dtype=dtype, device=device)
    res_out = flag_gems.kaiser_window(
        window_length, dtype=dtype, device=flag_gems.device
    )

    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    if window_length == 0:
        return
    utils.gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.kaiser_window_periodic
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_kaiser_window_periodic(window_length, periodic, dtype):
    # An explicit periodic flag selects the aten::kaiser_window.periodic
    # overload; N is window_length for the periodic window and window_length - 1
    # for the symmetric one.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.kaiser_window(
        window_length, periodic=periodic, dtype=dtype, device=device
    )
    res_out = flag_gems.kaiser_window(
        window_length, periodic=periodic, dtype=dtype, device=flag_gems.device
    )

    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    if window_length == 0:
        return
    utils.gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.kaiser_window_beta
@pytest.mark.parametrize("window_length", [1, 2, 3, 16, 33, 256, 1000])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.5, 3.0, 8.6, 12.0, 16.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_kaiser_window_beta(window_length, periodic, beta, dtype):
    # An explicit beta selects the aten::kaiser_window.beta overload. The lengths
    # stop at 1000 here: a flat window (small beta) is the most sensitive to the
    # CPU/CUDA index-rounding difference described above, and 4096 would sit
    # within 2.4x of the float16 tolerance in `--ref cpu` mode. The long-window
    # coverage lives in the two tests above instead.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.kaiser_window(
        window_length, periodic=periodic, beta=beta, dtype=dtype, device=device
    )
    res_out = flag_gems.kaiser_window(
        window_length,
        periodic=periodic,
        beta=beta,
        dtype=dtype,
        device=flag_gems.device,
    )

    assert res_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    if window_length == 0:
        return
    utils.gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.kaiser_window
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_kaiser_window_degenerate_lengths(dtype):
    # window_length == 1 is documented to return a single 1.0, and 0 an empty
    # tensor, regardless of periodic / beta.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    for periodic in (True, False):
        ref_one = torch.kaiser_window(1, periodic=periodic, dtype=dtype, device=device)
        res_one = flag_gems.kaiser_window(
            1, periodic=periodic, dtype=dtype, device=flag_gems.device
        )
        assert res_one.shape == (1,)
        assert res_one.dtype == dtype
        utils.gems_assert_close(res_one, ref_one, dtype=dtype)

        ref_empty = torch.kaiser_window(
            0, periodic=periodic, dtype=dtype, device=device
        )
        res_empty = flag_gems.kaiser_window(
            0, periodic=periodic, dtype=dtype, device=flag_gems.device
        )
        assert res_empty.shape == (0,)
        assert res_empty.dtype == dtype
        assert res_empty.numel() == ref_empty.numel()


@pytest.mark.kaiser_window
@pytest.mark.parametrize("window_length", [16, 1000])
# The default-device case only needs one representative pair of dtypes; the full
# dtype matrix is already covered by test_kaiser_window above.
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kaiser_window_default_device(window_length, dtype):
    # dtype=None resolves to the global default dtype, and the window is written
    # onto the default flag_gems device.
    device = "cpu" if cfg.TO_CPU else flag_gems.device
    ref_out = torch.kaiser_window(window_length, dtype=dtype, device=device)
    res_out = flag_gems.kaiser_window(window_length, dtype=dtype)

    assert res_out.device.type == torch.device(flag_gems.device).type
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype=dtype)


@pytest.mark.kaiser_window
def test_kaiser_window_only_strided_layout():
    with pytest.raises(ValueError, match="strided layout"):
        flag_gems.kaiser_window(16, layout=torch.sparse_coo)


@pytest.mark.kaiser_window
def test_kaiser_window_negative_length():
    with pytest.raises(AssertionError):
        flag_gems.kaiser_window(-1)


@pytest.mark.kaiser_window
def test_kaiser_window_requires_float_dtype():
    with pytest.raises(AssertionError):
        flag_gems.kaiser_window(16, dtype=torch.int32)
