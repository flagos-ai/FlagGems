import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from _kunlunxin.utils.pointwise_dynamic import pointwise_dynamic

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    is_scatter_slice=True,
)


@pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy(src):
    return src


@pointwise_dynamic(
    is_tensor=(True,), promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def copy_slice(src):
    return src
