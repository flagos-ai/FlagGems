"""飞桨后端机制测试共用的算子清单、输入构造与比对函数。

被 test_paddle_mechanism_*.py 与 test_paddle_triton_op_adaptation.py 共用。
"""

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle

# enable() 中 framework_name == "paddle" 分支注册的全部入口，形如 _C_ops.<name>
REGISTERED_OPS = (
    "softmax",
    "bmm",
    "sum",
    "mean",
    "triu",
    "addmm",
    "all",
    "amax",
    "argmax",
    "diag",
    "dot",
    "embedding",
    "ones",
    "zeros",
    "zeros_like",
    "topk",
    "max",
    "min",
    "mv",
    "index_select",
)

# paddle.ones / zeros / zeros_like 内部走 _C_ops.full / full_like，不经过上面的
# _C_ops.<op> 入口，enable() 另把这三个公共函数本身注册为 FlagGems 实现
PUBLIC_API_ENTRIES = ("ones", "zeros", "zeros_like")

# 与 flag_gems.testing.RESOLUTION 一致的相对容差
RTOL = {"float32": 1.3e-6, "float16": 1e-3, "bfloat16": 0.016}
ATOL = 1e-4


def assert_same(res, ref, dtype):
    """返回值的结构、shape、dtype 与数值都必须与 Paddle 原生入口一致。"""
    assert type(res) is type(ref), f"返回类型不一致: {type(res)} vs {type(ref)}"
    if isinstance(ref, (tuple, list)):
        assert len(res) == len(ref)
        for r, f in zip(res, ref):
            assert_same(r, f, dtype)
        return
    assert tuple(res.shape) == tuple(ref.shape)
    assert res.dtype == ref.dtype
    if res.dtype in (paddle.bool, paddle.int32, paddle.int64):
        assert bool((res == ref).all())
    else:
        assert bool(
            paddle.allclose(
                paddle.cast(res, "float32"),
                paddle.cast(ref, "float32"),
                rtol=RTOL[dtype],
                atol=ATOL,
            )
        )


def make_inputs(dtype):
    paddle.seed(1234)
    return {
        "x": paddle.randn([8, 16], dtype=dtype),
        "a": paddle.randn([4, 8, 16], dtype=dtype),
        "b": paddle.randn([4, 16, 8], dtype=dtype),
        "m1": paddle.randn([8, 16], dtype=dtype),
        "m2": paddle.randn([16, 8], dtype=dtype),
        "bias": paddle.randn([8, 8], dtype=dtype),
        "v": paddle.randn([16], dtype=dtype),
        "v2": paddle.randn([16], dtype=dtype),
        "xb": paddle.cast(paddle.randint(0, 2, [8, 16]), "bool"),
        "idx": paddle.randint(0, 8, [4, 3]),
        "w": paddle.randn([8, 16], dtype=dtype),
        "ii": paddle.to_tensor([0, 2, 4], dtype="int64"),
    }


# 全部按位置参数调用 Paddle 公共 API，调用约定与被替换的 C++ 入口 _C_ops.<op> 保持一致
API_CALLS = {
    "softmax": lambda g, d: paddle.nn.functional.softmax(g["x"], -1),
    "bmm": lambda g, d: paddle.bmm(g["a"], g["b"]),
    "sum": lambda g, d: paddle.sum(g["x"], 1),
    "mean": lambda g, d: paddle.mean(g["x"], 1),
    "triu": lambda g, d: paddle.triu(g["x"], 1),
    "addmm": lambda g, d: paddle.addmm(g["bias"], g["m1"], g["m2"], 0.7, 1.3),
    "all": lambda g, d: paddle.all(g["xb"]),
    "amax": lambda g, d: paddle.amax(g["x"], 1),
    "argmax": lambda g, d: paddle.argmax(g["x"], 1),
    "diag": lambda g, d: paddle.diag(g["v"]),
    "dot": lambda g, d: paddle.dot(g["v"], g["v2"]),
    "embedding": lambda g, d: paddle.nn.functional.embedding(g["idx"], g["w"]),
    "ones": lambda g, d: paddle.ones([4, 5], d),
    "zeros": lambda g, d: paddle.zeros([4, 5], d),
    "zeros_like": lambda g, d: paddle.zeros_like(g["x"]),
    "topk": lambda g, d: paddle.topk(g["x"], 3),
    "max": lambda g, d: paddle.max(g["x"]),
    "min": lambda g, d: paddle.min(g["x"]),
    "mv": lambda g, d: paddle.mv(g["x"], g["v"]),
    "index_select": lambda g, d: paddle.index_select(g["x"], g["ii"], 0),
}
