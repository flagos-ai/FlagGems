"""考核项 1：打通 C++/Python 跨语言调用的 ABI 接口（GN-0002 M1）。

技术要求：打通 Paddle 与 Triton 的互操作接口，即 C++/Python 跨语言调用的 ABI 接口。
验证点：被接管的 _C_ops 入口来自 C++ 扩展 libpaddle.so；按位置参数与关键字两种调用约定，
FlagGems 实现的入参与返回结构与 C++ 入口一致；Triton 侧 kernel 装载入口是编译产出的 C 扩展；
飞桨张量的裸地址与 stride 跨语言边界透传后数据正确。

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_mech \
      python -m pytest tests/test_paddle_mechanism_abi.py -q
"""

import sys

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest
import triton
import triton.language as tl

from .paddle_mechanism_utils import API_CALLS, REGISTERED_OPS, assert_same, make_inputs


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_native_entry_is_cpp_extension_symbol(op_name):
    """被接管的 _C_ops.<op> 入口本身是 C++ 扩展 libpaddle.so 导出的符号。"""
    entry = getattr(paddle._C_ops, op_name)
    assert type(entry).__name__ in ("builtin_function_or_method", "PyCapsule")
    assert "libpaddle" in entry.__module__
    assert paddle.base.libpaddle.__file__.endswith(".so")


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_abi_call_convention_compatible(op_name):
    """以 C++ 入口的位置参数约定调用，FlagGems 实现的入参与返回结构必须一致。"""
    dtype = "float32"
    g = make_inputs(dtype)
    call = API_CALLS[op_name]
    ref = call(g, dtype)
    with flag_gems.use_gems():
        res = call(g, dtype)
    assert_same(res, ref, dtype)


# 按飞桨公共 API 的关键字约定调用（关键字名须与被替换的 C++ 入口一致）
KEYWORD_CALLS = {
    "softmax": lambda g, d: paddle.nn.functional.softmax(g["x"], axis=-1),
    "sum": lambda g, d: paddle.sum(g["x"], axis=1, keepdim=False),
    "mean": lambda g, d: paddle.mean(g["x"], axis=1, keepdim=False),
    "triu": lambda g, d: paddle.triu(g["x"], diagonal=1),
    "addmm": lambda g, d: paddle.addmm(
        g["bias"], g["m1"], g["m2"], beta=0.7, alpha=1.3
    ),
    "all": lambda g, d: paddle.all(g["xb"], axis=1, keepdim=False),
    "amax": lambda g, d: paddle.amax(g["x"], axis=1, keepdim=False),
    "argmax": lambda g, d: paddle.argmax(g["x"], axis=1, keepdim=False),
    "diag": lambda g, d: paddle.diag(g["v"], offset=0),
    "embedding": lambda g, d: paddle.nn.functional.embedding(
        g["idx"], g["w"], padding_idx=None, sparse=False
    ),
    "ones": lambda g, d: paddle.ones(shape=[4, 5], dtype=d),
    "zeros": lambda g, d: paddle.zeros(shape=[4, 5], dtype=d),
    "zeros_like": lambda g, d: paddle.zeros_like(g["x"], dtype=d),
    "topk": lambda g, d: paddle.topk(g["x"], k=3, axis=-1, largest=True, sorted=True),
    "max": lambda g, d: paddle.max(g["x"], axis=1, keepdim=False),
    "min": lambda g, d: paddle.min(g["x"], axis=1, keepdim=False),
    "index_select": lambda g, d: paddle.index_select(g["x"], index=g["ii"], axis=0),
}


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", sorted(KEYWORD_CALLS))
def test_abi_keyword_convention_compatible(op_name):
    """以飞桨公共 API 的关键字约定调用，形参名与 C++ 入口一致。"""
    dtype = "float32"
    g = make_inputs(dtype)
    call = KEYWORD_CALLS[op_name]
    ref = call(g, dtype)
    with flag_gems.use_gems():
        res = call(g, dtype)
    assert_same(res, ref, dtype)


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
@pytest.mark.parametrize("op_name", ["amax", "argmax"])
def test_abi_accepts_torch_compat_dim_keyword(op_name):
    """torch 兼容层以 dim= 调用时同样兼容，与 GN-0001 的调用路径一致。"""
    x = paddle.randn([8, 16], dtype="float32")
    ref = getattr(paddle, op_name)(x, 1)
    with flag_gems.use_gems():
        res = getattr(paddle, op_name)(x, dim=1)
    assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
def test_triton_launcher_is_compiled_c_extension():
    """Triton 侧的 kernel 装载入口是编译产出的 C 扩展（cuda_utils*.so）。"""
    x = paddle.randn([64, 64], dtype="float32")
    with flag_gems.use_gems():
        paddle.nn.functional.softmax(x, -1)
    paddle.device.synchronize()
    utils = triton.runtime.driver.active.utils
    assert type(utils.load_binary).__name__ == "builtin_function_or_method"
    mod = sys.modules["cuda_utils"]
    assert mod.__file__.endswith(".so")
    assert type(mod.__loader__).__name__ == "ExtensionFileLoader"


@triton.jit
def _gather_strided_kernel(out_ptr, in_ptr, s0, s1, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    val = tl.load(in_ptr + row * s0 + cols * s1, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, val, mask=mask)


@pytest.mark.framework_mechanism
@pytest.mark.abi_interop
def test_raw_pointer_crosses_language_boundary():
    """飞桨张量直接交给 Triton kernel：裸地址与 stride 经 C 扩展透传后数据正确。"""
    x = paddle.randn([7, 9], dtype="float32")
    xt = x.t()  # 非连续视图
    out = paddle.empty([9, 7], dtype="float32")
    s0, s1 = xt.stride()
    _gather_strided_kernel[(9,)](out, xt, s0, s1, 7, BLOCK=8)
    paddle.device.synchronize()
    assert isinstance(xt.data_ptr(), int) and xt.data_ptr() > 0
    assert bool((out == xt.contiguous()).all())
