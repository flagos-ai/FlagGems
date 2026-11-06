"""考核项 4：无缝支持自动微分（GN-0002 M4）。

技术要求：无缝支持自动微分等框架核心功能。
验证点：
  ①输出可求导的 15 个算子全部以 paddle.autograd.PyLayer 接入动态图：softmax、embedding
    为手写 PyLayer，sum、mean、amax、max、min、triu、index_select、topk、bmm、mv、addmm、
    dot、diag 由 flag_gems.utils.paddle_autograd.with_autograd 包装；
  ②20 个算子在输入需要梯度时，输出的 stop_gradient 与原生一致（计算图不在任何算子处断开）；
    argmax、all 输出整数/布尔、ones、zeros、zeros_like 不依赖输入取值，与原生一样不建图；
  ③上述 13 个算子对每个可求导输入的梯度与 Paddle 原生一致（fp32/fp16/bf16，随机上游梯度），
    max、min、amax 含并列最大/最小值时按原生规则均分梯度；
  ④softmax 三种 dtype 的反向梯度、embedding 权重梯度与原生一致且反向确由 Triton kernel 承接；
  ⑤梯度跨多个 Triton 算子链式回传，backward 在 use_gems() 作用域内外执行结果一致；
  ⑥no_grad、stop_gradient、requires_grad 语义与原生一致。

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_mech \
      python -m pytest tests/test_paddle_mechanism_autograd.py -q
"""

import logging

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest

from .paddle_mechanism_utils import API_CALLS, REGISTERED_OPS, assert_same, make_inputs

# 输出可求导、由 with_autograd 包装为 PyLayer 的算子
WRAPPED_OPS = (
    "sum",
    "mean",
    "amax",
    "max",
    "min",
    "triu",
    "index_select",
    "topk",
    "bmm",
    "mv",
    "addmm",
    "dot",
    "diag",
)
# 输出不可求导或不依赖输入取值的算子：原生同样不建图
NO_GRAD_OPS = ("argmax", "all", "ones", "zeros", "zeros_like")
FLOAT_DTYPES = ("float32", "float16", "bfloat16")
# Paddle 原生在 GPU 上没有这些 (算子, dtype) 的前向 kernel，无法取得原生梯度作参照
NATIVE_UNSUPPORTED = {
    ("amax", "float16"),
    ("amax", "bfloat16"),
    ("mv", "float16"),
    ("mv", "bfloat16"),
}

# 每个算子：(可求导输入的构造, 以这些输入调用公共 API 的函数)
GRAD_CASES = {
    "sum": ({"x": [8, 16]}, lambda t: paddle.sum(t["x"], 1)),
    "mean": ({"x": [8, 16]}, lambda t: paddle.mean(t["x"], [0, 1], keepdim=True)),
    "amax": ({"x": [8, 16]}, lambda t: paddle.amax(t["x"], 1)),
    "max": ({"x": [8, 16]}, lambda t: paddle.max(t["x"], 0)),
    "min": ({"x": [8, 16]}, lambda t: paddle.min(t["x"])),
    "triu": ({"x": [8, 16]}, lambda t: paddle.triu(t["x"], 1)),
    "index_select": (
        {"x": [8, 16]},
        lambda t: paddle.index_select(t["x"], paddle.to_tensor([0, 2, 2, 5]), 0),
    ),
    "topk": ({"x": [8, 16]}, lambda t: paddle.topk(t["x"], 3)[0]),
    "bmm": ({"a": [4, 8, 16], "b": [4, 16, 8]}, lambda t: paddle.bmm(t["a"], t["b"])),
    "mv": ({"m": [8, 16], "v": [16]}, lambda t: paddle.mv(t["m"], t["v"])),
    "addmm": (
        {"bias": [8], "m1": [8, 16], "m2": [16, 8]},
        lambda t: paddle.addmm(t["bias"], t["m1"], t["m2"], 0.7, 1.3),
    ),
    "dot": ({"v": [16], "v2": [16]}, lambda t: paddle.dot(t["v"], t["v2"])),
    "diag": ({"x": [16]}, lambda t: paddle.diag(t["x"], 2)),
}

# 梯度经低精度算子累加，与原生的累加顺序不同，按 GN-0001 的容差放宽到一个量级
GRAD_RTOL = {"float32": 1e-5, "float16": 1e-2, "bfloat16": 0.05}
GRAD_ATOL = {"float32": 1e-5, "float16": 1e-3, "bfloat16": 1e-2}


def _leaf(shape, dtype):
    t = paddle.cast(paddle.randn(shape, dtype="float32"), dtype)
    t.stop_gradient = False
    return t


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _grads(fn, leaves, upstream, in_scope, backward_in_scope=False):
    """前向（可选在 use_gems() 内）后以给定上游梯度反传，返回输出与各叶子的梯度。"""
    for t in leaves.values():
        t.clear_grad()
    if in_scope:
        with flag_gems.use_gems():
            out = _first(fn(leaves))
            if backward_in_scope:
                out.backward(upstream)
    else:
        out = _first(fn(leaves))
    if not backward_in_scope:
        out.backward(upstream)
    grads = {k: t.grad.clone() for k, t in leaves.items()}
    for t in leaves.values():
        t.clear_grad()
    return out, grads


def _assert_grad_close(res, ref, dtype, name):
    assert res.dtype == ref.dtype and list(res.shape) == list(ref.shape), name
    assert bool(
        paddle.allclose(
            paddle.cast(res, "float32"),
            paddle.cast(ref, "float32"),
            rtol=GRAD_RTOL[dtype],
            atol=GRAD_ATOL[dtype],
        )
    ), f"{name} 梯度与原生不一致"


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_backward_path_goes_through_pylayer():
    """带反向的算子以 paddle.autograd.PyLayer 接入动态图自动微分。"""
    for name in ("softmax", "embedding"):
        entry = getattr(flag_gems, name)
        owner = getattr(entry, "__self__", None)
        assert owner is not None, f"{name} 未以 PyLayer.apply 形式注册"
        assert issubclass(owner, paddle.autograd.PyLayer)
    for name in WRAPPED_OPS:
        layer = getattr(getattr(flag_gems, name), "pylayer", None)
        assert layer is not None, f"{name} 未经 with_autograd 包装"
        assert issubclass(layer, paddle.autograd.PyLayer)


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("op_name", REGISTERED_OPS)
def test_graph_not_broken_by_any_op(op_name):
    """输入需要梯度时，20 个算子在作用域内的输出 stop_gradient 与原生一致。"""
    g = make_inputs("float32")
    for t in g.values():
        if paddle.is_floating_point(t):
            t.stop_gradient = False
    ref = _first(API_CALLS[op_name](g, "float32"))
    with flag_gems.use_gems():
        res = _first(API_CALLS[op_name](g, "float32"))
    assert res.stop_gradient == ref.stop_gradient
    assert res.stop_gradient == (op_name in NO_GRAD_OPS), f"{op_name} 输出的 stop_gradient 不符合预期"


# 输出可求导、必须有反向的算子：两份清单合起来恰好是 20 个注册算子
BACKWARD_OPS = tuple(op for op in REGISTERED_OPS if op not in NO_GRAD_OPS)


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_every_op_is_classified_for_backward():
    """20 个算子逐一归类：要么必须有反向，要么输出不可求导，不遗漏。"""
    assert set(BACKWARD_OPS) | set(NO_GRAD_OPS) == set(REGISTERED_OPS)
    assert not set(BACKWARD_OPS) & set(NO_GRAD_OPS)
    assert len(BACKWARD_OPS) == 15


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("op_name", BACKWARD_OPS)
def test_every_differentiable_op_has_backward(op_name):
    """15 个可求导算子在 use_gems() 内都有反向：原生有梯度的每个输入，作用域内同样拿到梯度且数值一致。"""
    g = make_inputs("float32")
    floats = {k: t for k, t in g.items() if paddle.is_floating_point(t)}
    for t in floats.values():
        t.stop_gradient = False
    names, leaves = list(floats), list(floats.values())

    def run(in_scope):
        if in_scope:
            with flag_gems.use_gems():
                out = _first(API_CALLS[op_name](g, "float32"))
        else:
            out = _first(API_CALLS[op_name](g, "float32"))
        upstream = paddle.ones(out.shape, out.dtype)
        return paddle.grad([out], leaves, [upstream], allow_unused=True)

    ref, res = run(False), run(True)
    used = [i for i, r in enumerate(ref) if r is not None]
    assert used, f"{op_name} 原生没有可求导输入"
    for i in used:
        assert res[i] is not None, f"{op_name} 对 {names[i]} 的反向缺失"
        _assert_grad_close(res[i], ref[i], "float32", f"{op_name}.{names[i]}")


def _grad_cases():
    for op in WRAPPED_OPS:
        for dtype in FLOAT_DTYPES:
            if (op, dtype) not in NATIVE_UNSUPPORTED:
                yield pytest.param(op, dtype, id=f"{op}-{dtype}")


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("op_name, dtype", list(_grad_cases()))
def test_op_backward_matches_native(op_name, dtype):
    """13 个经 with_autograd 接入的算子：对每个可求导输入的梯度与原生一致。"""
    paddle.seed(2024)
    shapes, fn = GRAD_CASES[op_name]
    leaves = {k: _leaf(s, dtype) for k, s in shapes.items()}
    with paddle.no_grad():
        out = _first(fn(leaves))
    upstream = paddle.cast(paddle.randn(out.shape, dtype="float32"), dtype)
    ref_out, ref = _grads(fn, leaves, upstream, in_scope=False)
    res_out, res = _grads(fn, leaves, upstream, in_scope=True)
    assert not res_out.stop_gradient
    for k in leaves:
        _assert_grad_close(res[k], ref[k], dtype, f"{op_name}.{k}")


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize(
    "op_name, call",
    [
        ("max", lambda t: paddle.max(t, 1)),
        ("max", lambda t: paddle.max(t)),
        ("min", lambda t: paddle.min(t, 1)),
        ("amax", lambda t: paddle.amax(t, 1)),
        ("amax", lambda t: paddle.amax(t)),
    ],
    ids=["max-axis", "max-all", "min-axis", "amax-axis", "amax-all"],
)
def test_extremum_ties_split_gradient_like_native(op_name, call):
    """存在并列最大/最小值时，梯度分配规则与原生一致。"""
    x = paddle.to_tensor([[1.0, 3.0, 3.0, -2.0], [2.0, 2.0, 0.0, 2.0]])
    x.stop_gradient = False
    leaves = {"x": x}
    upstream = None
    _, ref = _grads(lambda t: call(t["x"]), leaves, upstream, in_scope=False)
    _, res = _grads(lambda t: call(t["x"]), leaves, upstream, in_scope=True)
    _assert_grad_close(res["x"], ref["x"], "float32", op_name)


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("backward_in_scope", [False, True], ids=["bwd-outside", "bwd-inside"])
def test_long_chain_through_wrapped_ops(backward_in_scope):
    """梯度依次穿过 addmm、bmm、softmax、triu、index_select、topk、max、mv、dot、diag、mean、sum。"""
    paddle.seed(7)
    leaves = {
        "bias": _leaf([8], "float32"),
        "m1": _leaf([8, 16], "float32"),
        "m2": _leaf([16, 8], "float32"),
        "w": _leaf([2, 8, 8], "float32"),
        "v": _leaf([8], "float32"),
    }

    def fn(t):
        h = paddle.addmm(t["bias"], t["m1"], t["m2"], 0.5, 1.5)  # [8, 8]
        h = paddle.bmm(h.unsqueeze(0).expand([2, 8, 8]), t["w"])  # [2, 8, 8]
        h = paddle.nn.functional.softmax(h, -1)
        h = paddle.triu(paddle.sum(h, 0), -1)  # [8, 8]
        h = paddle.index_select(h, paddle.to_tensor([0, 3, 3, 7, 1, 2, 4, 5]), 0)
        vals = paddle.topk(h, 4)[0]  # [8, 4]
        r = paddle.max(vals, 1) + paddle.mv(h, t["v"])  # [8]
        s = paddle.dot(r, t["v"]) + paddle.mean(paddle.diag(r), [0, 1])
        return s

    upstream = paddle.to_tensor(1.0)
    ref_out, ref = _grads(fn, leaves, upstream, in_scope=False)
    res_out, res = _grads(fn, leaves, upstream, in_scope=True, backward_in_scope=backward_in_scope)
    assert_same(res_out, ref_out, "float32")
    for k in leaves:
        _assert_grad_close(res[k], ref[k], "float32", k)


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_constructor_requires_grad_matches_native():
    """ones、zeros、zeros_like：默认不建图；requires_grad=True 时为可累积梯度的叶子。"""
    x = _leaf([3, 4], "float32")
    for fn in (
        lambda: paddle.ones([3, 4]),
        lambda: paddle.zeros([3, 4]),
        lambda: paddle.zeros_like(x),
    ):
        ref = fn()
        with flag_gems.use_gems():
            res = fn()
        assert res.stop_gradient and ref.stop_gradient
    with flag_gems.use_gems():
        o = paddle.ones([3, 4], requires_grad=True)
        z = paddle.zeros_like(x)
    assert not o.stop_gradient and o.is_leaf
    (o * x + z).sum().backward()
    assert_same(o.grad, x.detach(), "float32")
    assert_same(x.grad, paddle.ones([3, 4]), "float32")


@pytest.mark.framework_mechanism
@pytest.mark.autograd
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_softmax_backward_matches_native(dtype):
    """softmax 反向梯度与 Paddle 原生一致，且输出参与建图。"""
    x = _leaf([32, 64], dtype)
    y = paddle.nn.functional.softmax(x, -1)
    y.backward(paddle.ones_like(y))
    ref = x.grad.clone()
    x.clear_grad()
    with flag_gems.use_gems():
        y2 = paddle.nn.functional.softmax(x, -1)
        assert not y2.stop_gradient
        y2.backward(paddle.ones_like(y2))
    assert_same(x.grad, ref, dtype)
    x.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_embedding_weight_grad_matches_native(caplog):
    """embedding 权重梯度与原生一致，且反向确实走 Triton 反向 kernel。"""
    w = _leaf([64, 32], "float32")
    idx = paddle.randint(0, 64, [8, 4])
    out = paddle.nn.functional.embedding(idx, w)
    out.backward(paddle.ones_like(out))
    ref = w.grad.clone()
    w.clear_grad()
    with caplog.at_level(logging.DEBUG, logger="flag_gems"):
        with flag_gems.use_gems():
            out2 = paddle.nn.functional.embedding(idx, w)
            out2.backward(paddle.ones_like(out2))
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("flag_gems")]
    assert any("GEMS EMBEDDING FORWARD" in m for m in messages)
    assert any("GEMS EMBEDDING BACKWARD" in m for m in messages)
    assert_same(w.grad, ref, "float32")
    w.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_chained_triton_ops_backward():
    """梯度跨 embedding → softmax → sum 三个 Triton 算子回传。"""
    w = _leaf([64, 32], "float32")
    idx = paddle.randint(0, 64, [8, 4])

    def forward():
        return paddle.sum(paddle.nn.functional.softmax(
            paddle.nn.functional.embedding(idx, w), -1))

    loss = forward()
    loss.backward()
    ref = w.grad.clone()
    w.clear_grad()
    with flag_gems.use_gems():
        loss2 = forward()
        loss2.backward()
    assert_same(w.grad, ref, "float32")
    assert_same(loss2, loss, "float32")
    w.clear_grad()


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_no_grad_scope_builds_no_graph():
    """no_grad 作用域内不建图，数值仍与原生一致。"""
    x = _leaf([8, 16], "float32")
    with flag_gems.use_gems(), paddle.no_grad():
        y = paddle.nn.functional.softmax(x, -1)
    assert y.stop_gradient
    ref = paddle.nn.functional.softmax(x, -1)
    assert_same(y, ref, "float32")


@pytest.mark.framework_mechanism
@pytest.mark.autograd
def test_stop_gradient_input_produces_no_grad():
    """输入 stop_gradient=True 时不产生梯度，与原生语义一致。"""
    x = paddle.randn([8, 16], dtype="float32")
    assert x.stop_gradient
    with flag_gems.use_gems():
        y = paddle.nn.functional.softmax(x, -1)
    assert y.stop_gradient
    assert x.grad is None
