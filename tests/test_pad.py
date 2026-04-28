import random
import time

import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_equal, to_reference

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.pad
@pytest.mark.parametrize(
    "shape",
    [[1024, 1024], [64, 64, 64, 64], [1, 64, 112, 112], [4, 64, 128]],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@pytest.mark.parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    rank = len(shape)
    if pad_mode != "constant" and rank < 3:
        pytest.skip("PyTorch non-constant padding requires 3D+ input tensors")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    if not contiguous:
        if flag_gems.vendor_name == "kunlunxin":
            x = x.cpu()[::2, ::2].to(flag_gems.device)
        else:
            x = x[::2, ::2]

    ref_x = to_reference(x)
    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

    rank = x.ndim
    if pad_mode == "constant":
        num_pad = rank * 2
    else:
        # Non-constant modes only pad last (rank-1) dims, up to 3 dims max.
        # For 2D: pad last 1 dim (2 values); 3D: pad last 2 dims (4 values);
        # 4D+: pad last 3 dims (6 values).
        num_pad = min(rank - 1, 3) * 2
    pad_params = torch.randint(0, 10, (num_pad,), dtype=torch.int32, device="cpu")
    pad_value = float(torch.randint(0, 1024, (1,), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        # Clamp each pad value to be valid for reflect (< dim) / circular (<= dim).
        for i in range(num_pad // 2):
            dim_size = x.shape[rank - 1 - i]
            max_pad = dim_size - 1 if pad_mode == "reflect" else dim_size
            pad_params[2 * i] = int(pad_params[2 * i]) % max(max_pad, 1)
            pad_params[2 * i + 1] = int(pad_params[2 * i + 1]) % max(max_pad, 1)
        pad_value = None

    # Convert pad_params to list of Python ints for torch.nn.functional.pad
    pad_params_list = [int(pad_params[i]) for i in range(pad_params.shape[0])]

    ref_out = torch.nn.functional.pad(ref_x, pad_params_list, pad_mode, pad_value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pad(x, pad_params_list, pad_mode, pad_value)

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    gems_assert_equal(res_out, ref_out)
