# Copyright 2026, The FlagOS Contributors.
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

# ``quantize_per_tensor_dynamic`` only accepts float32 tensors on the quantized
# CUDA backend (Half/BFloat16/Double raise "Quantize only works on Float
# Tensor"), so there is no FLOAT_DTYPES parametrization. Only qint8 / quint8
# are supported ("dtype QInt32 not supported"). The dynamic scale and
# zero_point are derived from the input's amax on both sides, so the qparams
# must be bit-exact and the int_repr compared exactly (round-half-to-even).
QUANT_DTYPES = [torch.qint8, torch.quint8]
REDUCE_RANGES = [False, True]

# Measured ATen contract: the [min, max] range is reduced on device, extended
# to include zero, and mapped onto the qmin/qmax grid (halved under
# reduce_range=True) with the small-scale cutoff at fp32(6.1e-5) and a 0.1
# fallback for degenerate ranges. The tests cover the regular regime and the
# degenerate ones (all zeros, tiny amax) that switch those branches on.
SHAPES = (
    [(8, 16), (16, 32, 4)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (1024, 1024),
        (16, 128, 64),
        (16, 7, 57, 32),
    ]
)


def _make_input(shape, device=None):
    device = device or flag_gems.device
    return torch.randn(shape, dtype=torch.float32, device=device) * 3.0


def _reference(x, dtype, reduce_range):
    return torch.quantize_per_tensor_dynamic(utils.to_reference(x), dtype, reduce_range)


def _assert_quantized_identical(res, ref):
    """Compare two int representations.

    On the accelerator the kernel is bit-exact against aten. Under
    ``--ref=cpu`` a quotient whose fp64 value sits a hair below a half-integer
    but rounds to the tie in fp32 (CPU computes the fp32 product, CUDA the
    fp64 quotient -- the same disagreement ``test_quantize_per_tensor``
    documents) may be off by one quantization level.
    """
    if utils.TO_CPU:
        # The two operands live on different devices here, so compare on CPU.
        res_int = res.int_repr().long().cpu()
        ref_int = ref.int_repr().long().cpu()
        diff = (res_int - ref_int).abs()
        assert diff.max().item() <= 1, (
            f"int_repr differs by more than one quantization level: "
            f"max |diff| = {diff.max().item()}"
        )
    else:
        utils.gems_assert_equal(res.int_repr(), ref.int_repr())


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGES)
def test_quantize_per_tensor_dynamic(shape, dtype, reduce_range):
    x = _make_input(shape)

    ref_out = _reference(x, dtype, reduce_range)
    res_out = flag_gems.quantize_per_tensor_dynamic(x, dtype, reduce_range)

    _assert_quantized_identical(res_out, ref_out)
    # dequantize exercises the (scale, zero_point) pair end to end. Under
    # --ref cpu the one-level int_repr disagreement documented in
    # _assert_quantized_identical propagates to the dequantized floats, so the
    # end-to-end check only runs against the CUDA reference.
    if not utils.TO_CPU:
        utils.gems_assert_close(
            torch.dequantize(res_out),
            utils.to_reference(torch.dequantize(ref_out)),
            dtype=torch.float32,
        )


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_zeros(dtype):
    """An all-zero input takes the degenerate-range branch (scale=0.1)."""
    x = torch.zeros(16, 32, dtype=torch.float32, device=flag_gems.device)

    ref_out = _reference(x, dtype, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(x, dtype, False)

    assert res_out.q_scale() == 0.1
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_tiny_amax(dtype):
    """An amax far below the small-scale threshold lifts scale to fp32(6.1e-5)."""
    x = torch.full((16, 32), 1e-10, dtype=torch.float32, device=flag_gems.device)

    ref_out = _reference(x, dtype, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(x, dtype, False)

    assert res_out.q_scale() == 6.0999998822808266e-05
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_scalar(dtype):
    """A 0-dim input keeps its shape through the quantization."""
    x = torch.tensor(2.5, dtype=torch.float32, device=flag_gems.device)

    ref_out = _reference(x, dtype, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(x, dtype, False)

    assert res_out.shape == ()
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_non_contiguous(dtype):
    """A strided payload is quantized with the same dynamic scale."""
    pool = torch.zeros(32, 64, dtype=torch.float32, device=flag_gems.device)
    x = pool[::2]
    x.copy_(_make_input((16, 64)))
    assert not x.is_contiguous()

    ref_out = _reference(x, dtype, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(x, dtype, False)

    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    assert res_out.int_repr().is_contiguous()
    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize(
    "memory_format,shape",
    [
        (torch.channels_last, (2, 3, 4, 5)),
        (torch.channels_last_3d, (2, 3, 4, 5, 6)),
    ],
    ids=["channels_last", "channels_last_3d"],
)
def test_quantize_per_tensor_dynamic_memory_format(memory_format, shape):
    """ATen keeps a channels-last input's layout through the quantizer."""
    utils.init_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).contiguous(
        memory_format=memory_format
    )

    ref_out = torch.quantize_per_tensor_dynamic(
        utils.to_reference(x), torch.qint8, False
    )
    res_out = flag_gems.quantize_per_tensor_dynamic(x, torch.qint8, False)

    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    assert res_out.int_repr().stride() == ref_out.int_repr().stride()
    utils.gems_assert_equal(res_out.int_repr(), utils.to_reference(ref_out.int_repr()))


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_quantize_per_tensor_dynamic_rejects_non_float32(bad_dtype):
    # ATen raises "Quantize only works on Float Tensor" rather than upcasting.
    x = torch.randn(32, dtype=bad_dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="Float Tensor"):
        flag_gems.quantize_per_tensor_dynamic(x, torch.qint8, False)


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic_rejects_qint32():
    x = _make_input((8, 8))
    with pytest.raises(RuntimeError, match="not supported"):
        flag_gems.quantize_per_tensor_dynamic(x, torch.qint32, False)


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic_empty():
    """An empty input raises on the min() reduction, exactly like ATen."""
    x = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.quantize_per_tensor_dynamic(x, torch.qint8, False)


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out():
    """The .out overload writes through quantized copy_ and adopts the
    dynamically computed quantizer.

    A single 2-D shape exercises the .out contract (same qdtype destination,
    quantizer adoption); the dynamic scale/zp paths are covered by the shape
    and edge-case tests of the functional overload above.
    """
    # A single small 2-D shape keeps the .out contract test focused on the
    # quantizer adoption; the shape sweep lives in the functional cases above.
    shape = (16, 32)  # small fixed .out target shape
    x = _make_input(shape)

    ref_out = _reference(x, torch.qint8, False)

    out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        0.9,
        3,
        torch.qint8,
    )
    res_out = flag_gems.quantize_per_tensor_dynamic_out(x, torch.qint8, False, out=out)
    assert res_out is out
    _assert_quantized_identical(utils.to_reference(out), utils.to_reference(ref_out))


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out_dtype_mismatch():
    x = _make_input((8, 8))
    # The destination must be a quantized tensor of the *requested* dtype; a
    # different quantized dtype is rejected with ATen's message.
    out = torch.quantize_per_tensor(
        torch.zeros((8, 8), dtype=torch.float32, device=flag_gems.device),
        0.9,
        3,
        torch.quint8,
    )
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.quantize_per_tensor_dynamic_out(x, torch.qint8, False, out=out)


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out_shape_mismatch():
    x = _make_input((8, 8))
    # The destination is a quantized tensor of the right dtype but the wrong
    # shape: a shape mismatch is a resize in ATen, ``aten::resize_`` has no
    # QuantizedCUDA kernel, so the native op fails here too.
    out = torch.quantize_per_tensor(
        torch.zeros((2, 2), dtype=torch.float32, device=flag_gems.device),
        0.9,
        3,
        torch.qint8,
    )
    with pytest.raises(NotImplementedError):
        flag_gems.quantize_per_tensor_dynamic_out(x, torch.qint8, False, out=out)
