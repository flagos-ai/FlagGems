import pytest
import torch
import triton

from flag_gems.utils.pointwise_dynamic import (
    CodeGenConfig,
    FunctionSchema,
    pointwise_dynamic,
)
from flag_gems.utils.tensor_wrapper import StridedBuffer

USE_BLOCK_POINTER = [True, False]
triton_version_less_than3 = int(triton.__version__[0]) < 3


def test_function_schema_with_non_tensor_input():
    schema = FunctionSchema(
        is_tensor=[True, False, True],
        dtypes=[None, float, None],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    assert schema.num_input_tensors() == 2
    assert schema.num_output_tensors() == 1
    assert schema.num_inputs() == 3
    assert schema.num_non_tensor_args() == 1
    assert schema.input_index(0) == 0  # the first input is the first input tensor
    assert schema.input_index(1) == 0  # the second input is the first non tensor input
    assert schema.input_index(2) == 1  # the third input is the second input tensor


def test_function_schema_mismatch_input_num1():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[True, False, True],
            dtypes=[None],
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_mismatch_input_num2():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[True, False, True],
            num_inputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_mismatch_input_num3():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            num_inputs=2,
            dtypes=[None, None, None],
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_missing_output_dtype_promotion_rules():
    with pytest.raises(ValueError):
        schema = FunctionSchema(
            num_inputs=2,
            dtypes=[None, None, None],
        )
        _ = schema


def test_function_schema_mismatch_output_num():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            num_inputs=1,
            num_outputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_missing_input_info():
    with pytest.raises(ValueError):
        schema = FunctionSchema(
            num_outputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_no_tensor_inputs1():
    # no tensor input is okay with FunctionSchema
    schema = FunctionSchema(
        is_tensor=[False, False, False],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    _ = schema


def test_function_schema_no_tensor_inputs2():
    schema = FunctionSchema(
        num_inputs=3,
        is_tensor=[False, False, False],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    _ = schema


def test_function_schema_no_outputs1():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[False, False, False],
            promotion_methods=[],
        )
        _ = schema


def test_function_schema_no_outputs2():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[False, False, False],
            num_outputs=0,
            promotion_methods=[],
        )
        _ = schema


def test_function_schema_illegal_dtypes():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(dtypes=[0, False, "a"])
        _ = schema


def test_function_schema_multiple_outputs():
    schema = FunctionSchema(
        num_inputs=3,
        num_outputs=2,
        promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, "ALWAYS_BOOL")],
    )
    _ = schema


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_without_non_tensor_args(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device="cuda")
        y = torch.randn_like(x)
        out = add(x, y)
        torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_non_tensor_args(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device="cuda")
        y = torch.randn_like(x)
        alpha = 2.0
        out = axpy(x, y, alpha)
        torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_multiple_outputs(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        num_outputs=2,
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def multiple_out(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device="cuda")
        y = torch.randn_like(x)
        alpha = 2.0
        out0, out1 = multiple_out(x, y, alpha)
        torch.testing.assert_close(out0, alpha * x + y)
        torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_broadcasting(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=True,  # [misaligned address]
    )

    # NOTE: [misaligned address]
    # triton 2.2 may cause Misaligned address when using >=3d tiles in some
    # cases with some zero strides
    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, 1, SIZE], device="cuda")
    y = torch.randn([1, SIZE, 1], device="cuda")
    alpha = 2.0
    out = axpy(x, y, alpha)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_broadcasting2(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=True,  # [misaligned address]
    )

    # NOTE: See note [misaligned address]
    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, 1, SIZE], device="cuda")
    y = torch.randn([], device="cuda")
    alpha = 2.0
    out = axpy(x, y, alpha)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_predefined_out(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([], device="cuda")
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    out = axpy(x, y, alpha, out0=o)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_some_predefined_out1(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([], device="cuda")
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_some_predefined_out2(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([], device="cuda")
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out1=o)
    assert out1 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_bool_input_and_output(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=1,
        is_tensor=[True],
        promotion_methods=[(0, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def invert(x):
        return ~x

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda") > 0
    notx = invert(x)

    torch.testing.assert_close(notx, ~x)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=1,
        is_tensor=[True],
        promotion_methods=[(0, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def invert(x):
        return ~x

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda") > 0
    o = torch.empty_like(x)
    # manually instantiated overload does not handle output allocation
    # since it is kind of low level
    notx = invert.instantiate(3)(x, out0=o)
    torch.testing.assert_close(notx, ~x)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_nd_buffer(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, K], device="cuda")[::2, ::2, ::2]
    y = torch.randn([N // 2, K // 2, M // 2], device="cuda").permute(2, 0, 1)
    alpha = 2.0
    o = torch.empty([M // 2, N // 2, K // 2], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_different_stride_order(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, K], device="cuda")
    y = torch.randn([N, K, M], device="cuda").permute(2, 0, 1)
    alpha = 2.0
    o = torch.empty([M, N, K], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_mixing_strided_buffer_and_tensor(
    use_block_pointer,
):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device="cuda"))
    out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)

    assert isinstance(out0, torch.Tensor)
    assert isinstance(out1, StridedBuffer)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_support_broadcasting1(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([1, SIZE], device="cuda")
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device="cuda"))

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_support_broadcasting2(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([SIZE, 1, SIZE], device="cuda")
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device="cuda")
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device="cuda"))

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_allocate_output(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device="cuda")
    y = torch.randn([SIZE, 1, SIZE], device="cuda")
    alpha = 2.0

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_gsl(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=512,
        max_grid_size=(80, 1, 1),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device="cuda")
        y = torch.randn_like(x)
        out = add(x, y)
        torch.testing.assert_close(out, x + y)


@pytest.mark.skipif(
    torch.cuda.get_device_properties(0).total_memory < (80 * 1024**3),
    reason="This test requires a lot of memory.",
)
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_int64_index(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 1, 1),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=config)
    @triton.jit
    def f(x):
        return x * 2.0

    x = torch.randn((2, 1024, 1024, 1024), dtype=torch.float16, device="cuda")
    y1 = f(x)
    y2 = x * 2.0
    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_0d_task(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    shape = ()
    x = torch.randn(shape, device="cuda")
    y = torch.randn_like(x)
    out = add(x, y)
    torch.testing.assert_close(out, x + y)
