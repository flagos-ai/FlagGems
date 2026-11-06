"""考核项 2：经飞桨运行时解析张量属性（GN-0002 M2）。

技术要求：Triton wrapper 通过飞桨运行时与设备接口解析张量属性。
验证点：六种内存布局下 shape / stride / dtype / element_size / place / data_ptr 由飞桨运行时
解析且与实际内存布局自洽；视图首地址偏移按 dtype 换算正确；device_guard 经飞桨 place 接口
判定设备；非连续输入下 Triton 算子结果与原生一致。

执行方式：
  CUDA_VISIBLE_DEVICES=<卡号> TRITON_CACHE_DIR=/tmp/tc_mech \
      python -m pytest tests/test_paddle_mechanism_tensor_attr.py -q
"""

import math

import flag_gems  # noqa: F401  必须早于 import torch，由它装上 paddle 的 torch proxy
import paddle
import pytest

from flag_gems import runtime

from .paddle_mechanism_utils import assert_same


def _layout(name, dtype="float32"):
    """返回 (基张量, 待解析张量)。"""
    base = paddle.randn([4, 6], dtype=dtype)
    if name == "contiguous":
        return base, base
    if name == "transposed":
        return base, base.t()
    if name == "sliced":
        return base, base[1:3, 2:5]
    if name == "scalar":
        t = paddle.randn([], dtype=dtype)
        return t, t
    if name == "empty":
        t = paddle.empty([0, 3], dtype=dtype)
        return t, t
    t = paddle.randn([2, 3, 4, 5, 6], dtype=dtype)
    return t, t


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize(
    "layout", ["contiguous", "transposed", "sliced", "scalar", "empty", "rank5"]
)
def test_runtime_resolves_tensor_attributes(layout):
    """shape / stride / dtype / element_size / place / data_ptr 由飞桨运行时解析且自洽。"""
    base, t = _layout(layout)
    numel = math.prod(t.shape)
    assert tuple(t.shape) == t.numpy().shape
    assert len(t.stride()) == t.dim() == len(t.shape)
    assert t.dtype == paddle.float32
    assert t.element_size() == 4
    assert t.place.is_gpu_place()
    assert t.place.gpu_device_id() == (
        paddle.framework._current_expected_place().gpu_device_id()
    )
    if numel > 0:
        assert isinstance(t.data_ptr(), int) and t.data_ptr() > 0
    if layout == "transposed":
        assert not t.is_contiguous()
        assert tuple(t.shape) == (6, 4)
        assert tuple(t.stride()) == (1, 6)  # 转置后内层 stride 为基张量的行距
    if layout == "sliced":
        assert t.data_ptr() == base.data_ptr() + (1 * 6 + 2) * t.element_size()
        assert tuple(t.shape) == (2, 3)
    if layout == "scalar":
        assert t.dim() == 0 and tuple(t.stride()) == ()


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
def test_view_offset_pointer_arithmetic(dtype):
    """视图的首地址偏移 = 元素偏移 × element_size，按 dtype 逐一核对。"""
    base = paddle.zeros([4, 6], dtype=dtype)
    view = base[2:, 3:]
    assert view.data_ptr() - base.data_ptr() == (2 * 6 + 3) * base.element_size()
    assert tuple(view.shape) == (2, 3)
    assert tuple(view.stride()) == (6, 1)


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
def test_device_guard_uses_paddle_place_interface():
    """device_guard 经 _current_expected_place 判定设备，同设备走零开销路径。"""
    x = paddle.randn([8], dtype="float32")
    cur = paddle.framework._current_expected_place()
    assert runtime.device_guard(x) is runtime._null_guard
    assert runtime.device_guard(x.place) is runtime._null_guard
    with runtime.device_guard(x):
        assert paddle.framework._current_expected_place() == cur


@pytest.mark.framework_mechanism
@pytest.mark.tensor_attr
@pytest.mark.parametrize("op", ["softmax", "sum", "triu", "index_select"])
def test_noncontiguous_input_matches_native(op):
    """非连续输入下结果仍与原生一致，说明 stride 解析正确驱动了 kernel。"""
    x = paddle.randn([16, 24], dtype="float32")
    xt = x.t()
    calls = {
        "softmax": lambda t: paddle.nn.functional.softmax(t, -1),
        "sum": lambda t: paddle.sum(t, 1),
        "triu": lambda t: paddle.triu(t, 1),
        "index_select": lambda t: paddle.index_select(
            t, paddle.to_tensor([0, 3, 5], dtype="int64"), 0
        ),
    }
    ref = calls[op](xt)
    with flag_gems.use_gems():
        res = calls[op](xt)
    assert_same(res, ref, "float32")
