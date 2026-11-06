"""考核项 3：动态图算子动态分发（GN-0002 M3）。

技术要求：动态图支持 Triton 后端算子的动态分发。
验证点：use_gems() 作用域内 20 个 _C_ops 入口被替换为 FlagGems 实现、退出后还原为同一个
C++ 入口对象；公共 API 不经过 _C_ops 入口的 ones / zeros / zeros_like 另在 paddle 与 torch
代理两处被替换并还原；注册集合与 enable() 的 paddle 分支配置一致；调用确实进入 Triton 实现；
exclude 指定算子回退原生；多次进出作用域、不同 shape 与取值依赖的 python 分支下分发均有效。

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_mech \
      python -m pytest tests/test_paddle_mechanism_dispatch.py -q
"""

import logging

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest

from .paddle_mechanism_utils import PUBLIC_API_ENTRIES, REGISTERED_OPS, assert_same


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_entry_swapped_and_restored(op_name):
    """进入作用域后入口指向 FlagGems 实现，退出后还原为同一个 C++ 入口对象。"""
    original = getattr(paddle._C_ops, op_name)
    with flag_gems.use_gems():
        patched = getattr(paddle._C_ops, op_name)
        assert patched is not original
        assert patched is getattr(flag_gems, op_name)
    assert getattr(paddle._C_ops, op_name) is original


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_registered_op_list_matches_config():
    """实际注册的入口集合与 enable() 的 paddle 分支配置一致，不多不少。"""
    with flag_gems.use_gems():
        registered = sorted(flag_gems.all_ops())
    assert registered == sorted(
        [f"_C_ops.{name}" for name in REGISTERED_OPS] + list(PUBLIC_API_ENTRIES)
    )


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("op_name", PUBLIC_API_ENTRIES)
def test_public_entry_swapped_and_restored(op_name):
    """公共 API 不经过 _C_ops.<op> 的构造类算子：paddle.<op> 与 torch 代理的 torch.<op>
    在作用域内都指向 FlagGems 实现，退出后都还原为原函数。"""
    import torch

    original = getattr(paddle, op_name)
    assert getattr(torch, op_name) is original
    impl = getattr(flag_gems, f"{op_name}_api")
    with flag_gems.use_gems():
        assert getattr(paddle, op_name) is impl
        assert getattr(torch, op_name) is impl
    assert getattr(paddle, op_name) is original
    assert getattr(torch, op_name) is original


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_triton_implementation_is_actually_invoked(caplog):
    """同一调用语句：作用域外无 FlagGems 记录，作用域内进入 Triton 实现。"""
    x = paddle.randn([32, 64], dtype="float32")
    with caplog.at_level(logging.DEBUG, logger="flag_gems"):
        paddle.nn.functional.softmax(x, -1)
        native = [r for r in caplog.records if r.name.startswith("flag_gems")]
        with flag_gems.use_gems():
            paddle.nn.functional.softmax(x, -1)
        gems = [r for r in caplog.records if r.name.startswith("flag_gems")]
    assert native == []
    assert any("GEMS SOFTMAX" in r.getMessage() for r in gems)


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("excluded", ["triu_paddle", "sum_paddle"])
def test_excluded_op_falls_back_to_native(excluded):
    """exclude 指定的算子保持原生入口，其余算子照常分发。"""
    name = excluded.replace("_paddle", "")
    original = getattr(paddle._C_ops, name)
    with flag_gems.use_gems(exclude=[excluded]):
        assert getattr(paddle._C_ops, name) is original
        assert paddle._C_ops.softmax is flag_gems.softmax
    assert getattr(paddle._C_ops, name) is original


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_repeated_scopes_restore_entry_each_time():
    """连续三次进出作用域，入口每次都能复位且数值稳定。"""
    original = paddle._C_ops.softmax
    x = paddle.randn([8, 16], dtype="float32")
    ref = paddle.nn.functional.softmax(x, -1)
    for _ in range(3):
        with flag_gems.use_gems():
            res = paddle.nn.functional.softmax(x, -1)
        assert paddle._C_ops.softmax is original
        assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
@pytest.mark.parametrize("shape", [(1, 2), (8, 16), (64, 128), (200, 3)])
def test_dynamic_shape_dispatch(shape):
    """动态图下同一入口连续承接不同 shape，无需重新注册。"""
    x = paddle.randn(list(shape), dtype="float32")
    ref = paddle.nn.functional.softmax(x, -1)
    with flag_gems.use_gems():
        res = paddle.nn.functional.softmax(x, -1)
    assert_same(res, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.dynamic_dispatch
def test_data_dependent_control_flow_dispatch():
    """分支条件依赖张量取值（动态图特性），两条分支都落到 Triton 实现。"""
    x = paddle.randn([8, 16], dtype="float32")

    def run(t):
        if float(paddle.sum(t)) > 0:
            return paddle.nn.functional.softmax(t, -1)
        return paddle.triu(t, 1)

    pos, neg = paddle.abs(x), -paddle.abs(x)
    ref_pos, ref_neg = run(pos), run(neg)
    with flag_gems.use_gems():
        res_pos, res_neg = run(pos), run(neg)
    assert_same(res_pos, ref_pos, "float32")
    assert_same(res_neg, ref_neg, "float32")
