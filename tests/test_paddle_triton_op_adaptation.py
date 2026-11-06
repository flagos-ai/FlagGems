"""考核项 5：完成不少于 20 个核心算子的 Triton 适配。

技术要求：基于飞桨的 Triton 扩展机制，完成不少于 20 个核心算子的 Triton 适配。
验证点：
  ①注册到飞桨的算子不少于 20 个，且与技术规范书指定的 20 个核心算子一一对应；
  ②每个算子的注册实现位于 FlagGems 算子源码中，且所在模块定义了 Triton kernel；
  ③经飞桨公共 API 调用时，use_gems() 作用域内实际发射了 FlagGems 定义的 Triton kernel，
    作用域外同一调用不发射任何 Triton kernel（原生路径未被改动），输出为 GPU 上的飞桨张量；
  ④张量构造类算子经注册入口 _C_ops.<op> 与 torch 代理的 torch.<op> 调用时同样发射
    Triton kernel 且填充值正确。

Triton kernel 的发射通过 triton.knobs.runtime.launch_enter_hook 记录，每次 kernel 启动
（含 LibEntry 缓存命中后的直接启动）都会经过该钩子。

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_adapt \
      python -m pytest tests/test_paddle_triton_op_adaptation.py -q
"""

import importlib
import sys

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest
import triton

from .paddle_mechanism_utils import API_CALLS, make_inputs

# 技术规范书 / 测试大纲 7.1.1 指定的 20 个核心算子
CORE_OPS = (
    "softmax",
    "sum",
    "mean",
    "max",
    "min",
    "all",
    "amax",
    "argmax",
    "addmm",
    "bmm",
    "mv",
    "dot",
    "diag",
    "triu",
    "index_select",
    "embedding",
    "topk",
    "ones",
    "zeros",
    "zeros_like",
)
MIN_CORE_OPS = 20
FLOAT_DTYPES = ("float32", "float16", "bfloat16")
CONSTRUCTOR_OPS = ("ones", "zeros", "zeros_like")

# Paddle 原生在 GPU 上未注册这些 (算子, dtype) 的 kernel，只能跳过“原生路径不发射 Triton”这一步
NATIVE_UNSUPPORTED = {
    ("amax", "float16"),
    ("amax", "bfloat16"),
    ("mv", "float16"),
    ("mv", "bfloat16"),
}


@pytest.fixture
def triton_launches():
    """记录本用例内发射的 Triton kernel 名称。"""
    names = []

    def hook(metadata):
        names.append(metadata.get()["name"])

    triton.knobs.runtime.launch_enter_hook.add(hook)
    try:
        yield names
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(hook)


def _jit_name(obj):
    """LibEntry / Autotuner / Heuristics 层层包装，取最内层 JITFunction 的名称。"""
    if not isinstance(obj, triton.runtime.KernelInterface):
        return None
    while not isinstance(obj, triton.runtime.JITFunction):
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return obj.__name__


def _kernel_names(module):
    names = {_jit_name(v) for v in vars(module).values()}
    names.discard(None)
    return names


def _flag_gems_kernel_names():
    names = set()
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and mod_name.split(".")[0] == "flag_gems":
            names |= _kernel_names(mod)
    return names


def _impl_module(entry):
    """注册对象所在的模块；PyLayer.apply 取其所属 PyLayer 子类的模块。"""
    owner = getattr(entry, "__self__", None)
    if isinstance(owner, type):
        return importlib.import_module(owner.__module__)
    return importlib.import_module(entry.__module__)


def _expected_dtypes(op, dtype):
    if op == "all":
        return [paddle.bool]
    if op == "argmax":
        return [paddle.int64]
    if op == "topk":
        return [getattr(paddle, dtype), paddle.int64]
    return [getattr(paddle, dtype)]


def _assert_gpu_outputs(out, expected_dtypes):
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    assert [o.dtype for o in outs] == expected_dtypes
    for o in outs:
        assert isinstance(o, paddle.Tensor)
        assert o.place.is_gpu_place()


@pytest.mark.triton_op_adaptation
def test_core_op_count_at_least_20():
    """注册到飞桨的算子不少于 20 个，且覆盖技术规范书指定的全部核心算子。"""
    with flag_gems.use_gems():
        registered = {name.split(".")[-1] for name in flag_gems.all_ops()}
    assert len(CORE_OPS) == MIN_CORE_OPS
    assert len(registered) >= MIN_CORE_OPS
    assert set(CORE_OPS) <= registered, f"缺少: {sorted(set(CORE_OPS) - registered)}"


@pytest.mark.triton_op_adaptation
@pytest.mark.parametrize("op_name", CORE_OPS)
def test_registered_impl_is_flag_gems_triton_op(op_name):
    """注册实现位于 flag_gems.ops，且所在模块定义（或引入）了 Triton kernel。"""
    with flag_gems.use_gems():
        entry = getattr(paddle._C_ops, op_name)
    module = _impl_module(entry)
    assert module.__name__.startswith("flag_gems.ops."), module.__name__
    assert _kernel_names(module), f"{module.__name__} 中没有 Triton kernel"


def _public_api_cases():
    for op in CORE_OPS:
        dtypes = ("float32",) if op == "all" else FLOAT_DTYPES
        for dtype in dtypes:
            label = "bool" if op == "all" else dtype
            yield pytest.param(op, dtype, id=f"{op}-{label}")


@pytest.mark.triton_op_adaptation
@pytest.mark.parametrize("op_name, dtype", list(_public_api_cases()))
def test_public_api_launches_flag_gems_triton_kernel(op_name, dtype, triton_launches):
    """经飞桨公共 API 调用：作用域外不发射 Triton kernel，作用域内发射 FlagGems 的 Triton kernel。"""
    g = make_inputs(dtype)
    call = API_CALLS[op_name]

    if (op_name, dtype) not in NATIVE_UNSUPPORTED:
        call(g, dtype)
        paddle.device.synchronize()
        assert triton_launches == [], f"原生路径发射了 Triton kernel: {triton_launches}"

    triton_launches.clear()
    with flag_gems.use_gems():
        out = call(g, dtype)
    paddle.device.synchronize()

    launched = set(triton_launches)
    assert launched, f"use_gems() 内调用 {op_name} 未发射任何 Triton kernel"
    unknown = launched - _flag_gems_kernel_names()
    assert not unknown, f"发射了非 FlagGems 定义的 kernel: {sorted(unknown)}"
    _assert_gpu_outputs(out, _expected_dtypes(op_name, dtype))


@pytest.mark.triton_op_adaptation
@pytest.mark.parametrize("entry", ["_C_ops", "torch"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("op_name", CONSTRUCTOR_OPS)
def test_constructor_entry_launches_triton_kernel(op_name, dtype, entry, triton_launches):
    """张量构造类算子经注册入口 _C_ops.<op> 与 torch 代理的 torch.<op> 调用，
    均发射 Triton kernel 且填充值正确。"""
    import torch

    place = paddle.framework._current_expected_place()
    pd_dtype = getattr(paddle, dtype)
    x = paddle.randn([4, 5], dtype=dtype)
    calls = {
        ("ones", "_C_ops"): lambda: paddle._C_ops.ones([4, 5], pd_dtype, place),
        ("zeros", "_C_ops"): lambda: paddle._C_ops.zeros([4, 5], pd_dtype, place),
        ("zeros_like", "_C_ops"): lambda: paddle._C_ops.zeros_like(x, pd_dtype, place),
        ("ones", "torch"): lambda: torch.ones(
            [4, 5], dtype=pd_dtype, device=flag_gems.device
        ),
        ("zeros", "torch"): lambda: torch.zeros(
            [4, 5], dtype=pd_dtype, device=flag_gems.device
        ),
        ("zeros_like", "torch"): lambda: torch.zeros_like(x),
    }
    with flag_gems.use_gems():
        out = calls[(op_name, entry)]()
    paddle.device.synchronize()

    assert set(triton_launches) and set(triton_launches) <= _kernel_names(
        _impl_module(getattr(flag_gems, op_name))
    ), f"发射的 kernel: {triton_launches}"
    _assert_gpu_outputs(out, [pd_dtype])
    assert tuple(out.shape) == (4, 5)
    fill = 1.0 if op_name == "ones" else 0.0
    assert bool((paddle.cast(out, "float32") == fill).all())
