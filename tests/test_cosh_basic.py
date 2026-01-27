# tests/test_cosh_basic.py
import pytest
import torch
import flag_gems

pytestmark = pytest.mark.cosh


def _tol(dtype):
    if dtype == torch.float16:
        return 1e-4, 1e-3
    if dtype == torch.bfloat16:
        return 1e-4, 1.6e-2
    if dtype == torch.float32:
        return 1e-4, 1.3e-6
    raise RuntimeError(f"unsupported dtype: {dtype}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1,), (8,), (257,), (64, 64), (256, 256)])
def test_cosh_basic(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(*shape, device="cuda", dtype=dtype)

    ref = torch.cosh(x)
    with flag_gems.use_gems():
        out = torch.cosh(x)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_cosh_special_values(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.tensor(
        [0.0, 1.0, -1.0, 10.0, -10.0, float("inf"), float("-inf"), float("nan")],
        device="cuda",
        dtype=dtype,
    )

    ref = torch.cosh(x)
    with flag_gems.use_gems():
        out = torch.cosh(x)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_cosh_empty(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.empty((0,), device="cuda", dtype=dtype)

    ref = torch.cosh(x)
    with flag_gems.use_gems():
        out = torch.cosh(x)

    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True)

# @pytest.mark.inplace
# @pytest.mark.cosh_
# @pytest.mark.parametrize("shape", POINTWISE_SHAPES)
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
# def test_accuracy_cosh_(shape, dtype):
#     # 如果算子有定义域限制（如 log 必须 > 0），这里需要调整输入分布
#     inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
#     ref_inp = to_reference(inp.clone(), True)

#     ref_out = torch.cosh_(ref_inp)
#     with flag_gems.use_gems():
#         res_out = torch.cosh_(inp)

#     gems_assert_close(res_out, ref_out, dtype)