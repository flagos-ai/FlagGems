import os
from itertools import product
from typing import Generator, Tuple

import pytest
import torch

import flag_gems

from . import base, consts


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.mul
def test_mul():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
        # dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.mul_
def test_mul_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul_",
        torch_op=lambda a, b: a.mul_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


# ============================================================================
# MulPerfBenchmark — 支持广播乘、标量乘的 benchmark，兼容 qwen3.6 真实 mul shape
# shape 文件格式（每行一个 shape）：
#   M,N,same,COUNT             → 同形张量：torch.mul(A(M,N), B(M,N))
#   A_dims|B_dims,broadcast,COUNT → 广播乘：torch.mul(A(a_dims), B(b_dims))
#   A_dims,scalar,COUNT        → 标量乘：torch.mul(A(a_dims), scalar)
# ============================================================================
class MulPerfBenchmark(base.Benchmark):
    """支持广播和标量的 mul 性能 benchmark。
    读取自定义 txt shape 文件，逐行解析 shape 类型并生成对应的张量输入。
    """

    DEFAULT_DTYPES = [torch.bfloat16]

    def __init__(self, shape_file_path: str):
        super().__init__(
            op_name="mul",
            torch_op=torch.mul,
            dtypes=self.DEFAULT_DTYPES,
        )
        self.gems_op = flag_gems.mul
        self._parsed_shapes: list[Tuple[str, tuple, object]] = []
        self._shape_counts: dict = {}
        self._load_shapes(shape_file_path)

    def _load_shapes(self, path: str):
        """解析 shape 文件。格式：每行 'shape_desc,type,count'"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mul shape file not found: {path}")
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                # 最后一项是 count
                count = int(parts[-1])
                shape_type = parts[-2]

                if shape_type == "same":
                    # M,N,same,COUNT
                    dims = tuple(int(x) for x in parts[:-2])
                    self._parsed_shapes.append(("same", (dims, dims)))
                    self._shape_counts[(dims, "same")] = count
                elif shape_type == "broadcast":
                    # A_dims|B_dims,broadcast,COUNT
                    shape_part = ",".join(parts[:-2])
                    a_str, b_str = shape_part.split("|")
                    a_dims = tuple(int(x) for x in a_str.split(","))
                    b_dims = tuple(int(x) for x in b_str.split(","))
                    self._parsed_shapes.append(("broadcast", (a_dims, b_dims)))
                    self._shape_counts[(a_dims, b_dims)] = count
                elif shape_type == "scalar":
                    # A_dims,scalar,COUNT
                    dims = tuple(int(x) for x in parts[:-2] if x.strip().isdigit() or (x.strip().startswith('-') and x.strip()[1:].isdigit()))
                    # 重新解析：去掉末尾的 scalar 类型和 count
                    raw = ",".join(parts[:-2])
                    dims = tuple(int(x.strip()) for x in raw.split(",") if x.strip().lstrip('-').isdigit())
                    self._parsed_shapes.append(("scalar", dims))
                    self._shape_counts[(dims, "scalar")] = count
        self.shapes = self._parsed_shapes

    def set_shapes(self, shape_file_path=None):
        """覆盖父类，在 __init__ 中已加载 shapes"""
        pass

    def get_input_iter(self, dtype) -> Generator:
        """根据 shape 类型生成对应的张量输入"""
        for shape_info in self._parsed_shapes:
            kind = shape_info[0]
            if kind == "same":
                dims = shape_info[1][0]
                a = torch.randn(dims, dtype=dtype, device=self.device)
                b = torch.randn(dims, dtype=dtype, device=self.device)
            elif kind == "broadcast":
                a_dims, b_dims = shape_info[1]
                a = torch.randn(a_dims, dtype=dtype, device=self.device)
                b = torch.randn(b_dims, dtype=dtype, device=self.device)
            elif kind == "scalar":
                dims = shape_info[1]
                a = torch.randn(dims, dtype=dtype, device=self.device)
                b = 0.5  # scalar
            else:
                continue
            yield (a, b)

    def get_tflops(self, op, *args, **kwargs):
        """TFLOPS 估算：广播/标量的计算量基于实际参与运算的元素数"""
        if len(args) >= 2:
            a, b = args[0], args[1]
            n_a = a.numel() if hasattr(a, 'numel') else 1
            n_b = b.numel() if hasattr(b, 'numel') else 1
            return float(max(n_a, n_b))
        return 1.0


@pytest.mark.mul_perf
def test_mul_perf():
    """用 qwen3.6 真实 mul shape 做 benchmark（支持广播/标量）。
    通过 --shape_file 参数指定 shape 文件路径。
    """
    from .base import Config
    shape_file = getattr(Config, 'shape_file', None)
    if shape_file is None or not shape_file:
        # 回退到默认的 BinaryPointwiseBenchmark
        bench = base.BinaryPointwiseBenchmark(
            op_name="mul",
            torch_op=torch.mul,
            dtypes=[torch.bfloat16],
        )
    else:
        bench = MulPerfBenchmark(shape_file_path=shape_file)
    bench.run()
