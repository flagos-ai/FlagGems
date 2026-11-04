import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


MAXPOOL1D_CONFIGS = [
    # (shape, kernel_size, stride, padding, dilation, ceil_mode)
    # Classic case: stride defaults to kernel_size
    ((4, 8, 64), 2, None, 0, 1, False),
    # Explicit stride, padding
    ((2, 16, 100), 3, 2, 1, 1, False),
    # ceil_mode
    ((2, 4, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 32), 2, 1, 0, 2, False),
    # Larger channel/length
    ((8, 32, 224), 3, 2, 1, 1, False),
    # No padding, stride 1
    ((3, 8, 50), 4, 1, 0, 1, False),
    # 2D input (C, L)
    ((16, 128), 3, 2, 1, 1, False),
    # kernel_size passed as list
    ((2, 8, 40), [3], [2], [1], [1], False),
]


@pytest.mark.max_pool1d_with_indices
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL1D_CONFIGS
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_pool1d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    ref_out, ref_indices = torch.nn.functional.max_pool1d_with_indices(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    res_out, res_indices = flag_gems.max_pool1d_with_indices(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)


# CPU references are intentional: some vendor native kernels have different NaN
# or padding behavior. Include both the packed and general-load paths.
MAXPOOL1D_EDGE_CONFIGS = [
    pytest.param([float("-inf")] * 8, 3, 2, 1, 1, False, id="wide-all-inf"),
    pytest.param([float("-inf")] * 7, 3, 2, 1, 1, True, id="flat-all-inf-ceil"),
    pytest.param([float("-inf")] * 8, 2, 1, 1, 3, False, id="dilated-all-inf"),
    pytest.param([float("-inf")], 2, 1, 1, 2, False, id="no-valid-taps"),
    pytest.param(
        [float("nan"), 1, float("-nan"), -2, float("nan"), 3, 4, 5],
        3,
        2,
        1,
        1,
        False,
        id="wide-nan",
    ),
    pytest.param(
        [float("nan"), 1, float("-nan"), -2, float("nan"), 3, 4],
        3,
        1,
        1,
        2,
        True,
        id="dilated-nan",
    ),
    pytest.param([float("nan")] * 8, 4, 2, 2, 1, True, id="all-nan"),
    pytest.param(
        [-0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0],
        2,
        2,
        0,
        1,
        False,
        id="signed-zero",
    ),
    pytest.param(
        [4, 4, -3, -3, 4, 4, -3, -3], 3, 2, 1, 1, False, id="first-equal-maximum"
    ),
    pytest.param(
        [float("inf"), float("inf"), -1, -2, -3, -4, -5, -6],
        3,
        2,
        1,
        1,
        False,
        id="positive-inf",
    ),
    pytest.param(
        [-8, -7, -6, -5, -4, -3, -2, -1],
        3,
        3,
        1,
        1,
        True,
        id="negative-values-odd-stride",
    ),
    pytest.param(
        [0, 2**-24, -(2**-24), 2**-20, -(2**-20), 0, -0.0, 1],
        3,
        2,
        1,
        1,
        False,
        id="small-values",
    ),
]


@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin",
    reason="exact NaN/signed-zero matching is kunlunxin-only",
)
@pytest.mark.max_pool1d_with_indices
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "offset", "strided", "unbatched"])
@pytest.mark.parametrize(
    "values, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL1D_EDGE_CONFIGS,
)
def test_max_pool1d_with_indices_special_values(
    dtype, layout, values, kernel_size, stride, padding, dilation, ceil_mode
):
    cpu_input = torch.tensor(values, dtype=dtype).repeat(2, 3, 1)
    if layout == "unbatched":
        cpu_input = cpu_input[0]
    if layout == "offset":
        # .contiguous() preserves this odd storage offset. Packed loads must
        # check pointer alignment, not just the logical shape/contiguity.
        storage = torch.empty(
            cpu_input.numel() + 1, device=flag_gems.device, dtype=dtype
        )
        inp = storage[1:].view(cpu_input.shape)
        inp.copy_(cpu_input)
    elif layout == "strided":
        storage = torch.empty(
            (*cpu_input.shape[:-1], cpu_input.shape[-1] * 2),
            device=flag_gems.device,
            dtype=dtype,
        )
        inp = storage[..., ::2]
        inp.copy_(cpu_input)
    else:
        inp = cpu_input.to(device=flag_gems.device)

    args = (kernel_size, stride, padding, dilation, ceil_mode)
    ref_out, ref_indices = torch.nn.functional.max_pool1d_with_indices(cpu_input, *args)
    res_out, res_indices = flag_gems.max_pool1d_with_indices(inp, *args)
    actual = res_out.cpu()
    torch.testing.assert_close(actual, ref_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(res_indices.cpu(), ref_indices, rtol=0, atol=0)
    assert res_indices.dtype == torch.int64
    zeros = ref_out == 0
    torch.testing.assert_close(
        torch.signbit(actual[zeros]), torch.signbit(ref_out[zeros])
    )


@pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin",
    reason="exact NaN/signed-zero/bit-pattern matching is kunlunxin-only",
)
@pytest.mark.max_pool1d_with_indices
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("length", [255, 256])
def test_max_pool1d_with_indices_bit_patterns(dtype, length):
    if dtype == torch.float32:
        generator = torch.Generator().manual_seed(2026)
        raw = torch.randint(
            -(2**31), 2**31, (65536,), generator=generator, dtype=torch.int64
        ).to(torch.int32)
    else:
        # Every fp16/bf16 encoding, including subnormals, signed NaNs and both
        # zeros. Adjacent encodings exercise ordering around exponent changes.
        raw = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    cpu_input = raw.view(dtype).reshape(1, 256, 256)[..., :length].contiguous()
    ref_out, ref_indices = torch.nn.functional.max_pool1d_with_indices(
        cpu_input, 3, 2, 1
    )
    res_out, res_indices = flag_gems.max_pool1d_with_indices(
        cpu_input.to(flag_gems.device), 3, 2, 1
    )
    actual = res_out.cpu()
    torch.testing.assert_close(actual, ref_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(res_indices.cpu(), ref_indices, rtol=0, atol=0)
    # Selection should preserve source bits, including NaN signs/payloads.
    # Gather integer encodings: CPU floating gather can quiet signaling NaNs.
    selected = torch.gather(cpu_input.view(raw.dtype), -1, ref_indices)
    torch.testing.assert_close(actual.view(raw.dtype), selected)
    zeros = ref_out == 0
    torch.testing.assert_close(
        torch.signbit(actual[zeros]), torch.signbit(ref_out[zeros])
    )
