import importlib
import logging
import math
import os
from typing import Callable, List, Mapping, Tuple, Union

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

from .vstack import vstack

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


class CatKernelGenerator(IndentedBuffer):
    overloads: Mapping[str, Callable] = {}

    def __init__(self):
        self.pid = os.getpid()
        self.cache = self.overloads
        super().__init__()

    def __init(
        self,
        tensors: List[torch.Tensor],
        dim: int,
        high_num: int,
        low_cat_accum: List[int],
    ):
        self.dim = dim
        self.high_num = high_num
        self.low_cat_accum = low_cat_accum
        self.tensor_num = len(tensors)
        even = all([t.numel() == tensors[0].numel() for t in tensors])

        if even and low_cat_accum[-1] // self.tensor_num <= 128:
            # Special case for tensors with small and even low size,
            # which means weak contiguity when storing the out tensor.
            # Divide each tensor into tiles of `BLOCK_LOW` size,
            # and each cta process tiles one by one.
            self.kernel_name = "_cat_kernel_small"
            self.wrapper_name = "_cat_wrapper_small"
            self.MODE = 0
        else:
            # General cases.
            # Divide tasks by high_num, each cta process parts of high of all tensors.
            self.kernel_name = "_cat_kernel_parthigh"
            self.wrapper_name = "_cat_wrapper_parthigh"
            self.MODE = 1

    def __call__(
        self,
        tensors: List[torch.Tensor],
        dim: int,
        high_num: int,
        low_cat_accum: List[int],
    ):
        self.__init(tensors, dim, high_num, low_cat_accum)
        key = f"{len(tensors)}_{high_num}_{low_cat_accum[-1]}"
        if key not in self.cache:
            self.codegen()

            filename = f"{self.kernel_name}_{key}.py"
            filepath = code_cache_dir() / filename
            write_atomic(filepath, self.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_{key}", filepath
            )
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, self.wrapper_name)
            self.cache[key] = overload
        overload = self.cache[key]
        return overload(tensors, dim, high_num, low_cat_accum)

    def gen_imports(self):
        self.writeline("import math")
        self.writeline("import copy")
        self.newline()
        self.writeline("import torch")
        self.writeline("import triton")
        self.writeline("import triton.language as tl")
        self.newline()
        self.writeline("from flag_gems.runtime import torch_device_fn")
        self.writeline("from flag_gems.runtime.backend import vendor_module")
        self.writeline("from flag_gems.utils import libentry, libtuner")
        self.newline()
        self.writeline("TOTAL_CORE_NUM = vendor_module.TOTAL_CORE_NUM")
        self.newline()
        self.newline()

    def gen_wrapper(self):
        self.writeline(
            f"def {self.wrapper_name}(tensors, dim, high_num, low_cat_accum):"
        )
        with self.indent():
            self.writeline("device = tensors[0].device")
            self.writeline("dtype = tensors[0].dtype")
            self.writeline("tensor_num = len(tensors)")
            self.writeline("cat_dim_size = sum([t.shape[dim] for t in tensors])")
            self.writeline("out_shape = list(tensors[0].shape)")
            self.writeline("out_shape[dim] = cat_dim_size")
            self.writeline("out_cat_num = low_cat_accum[-1]")
            self.writeline("out = torch.empty(out_shape, device=device, dtype=dtype)")
            for i in range(self.tensor_num):
                self.writeline(f"in{i}_stride_high = tensors[{i}].stride(dim - 1)")
                self.writeline(f"in{i}_stride_low = tensors[{i}].stride(-1)")
            self.writeline("out_stride_high = out.stride(dim - 1)")
            self.writeline("out_stride_low = out.stride(-1)")
            self.writeline(
                "grid = lambda meta: (TOTAL_CORE_NUM // meta['num_warps'], )"
            )
            self.writeline("with torch_device_fn.device(device):")
            with self.indent():
                self.writeline(
                    f"{self.kernel_name}[grid]({self.gen_kernel_args(is_declare=False)})"
                )
            self.writeline("return out")
        self.newline()
        self.newline()

    def gen_decorators(self):
        self.writeline("@libentry()")
        self.writeline("@libtuner(")
        with self.indent():
            self.writeline("configs=[")
            with self.indent():
                if self.MODE == 0:
                    self.writeline(
                        """
        triton.Config({'BLOCK_LOW': 2 ** i}, num_stages=1, num_warps=1) for i in range(7, 12)
                        """
                    )
                elif self.MODE == 1:
                    self.writeline(
                        """
        triton.Config({'BLOCK_HIGH': i, 'BLOCK_LOW': 2 ** j}, num_stages=1, num_warps=1)
        for i in [6, 11, 22]
        for j in range(8, 12)
                        """
                    )
                self.writeline("],")
            self.writeline("key=['high_num', 'out_cat_num'],")
            self.writeline("strategy=['log', 'log'],")
            self.writeline("restore_value=['out'],")
        self.writeline(")")
        self.writeline("@triton.jit")

    def gen_kernel(self):
        self.writeline(f"def {self.kernel_name}({self.gen_kernel_args()}):")
        with self.indent():
            self.writeline("pid = tl.program_id(0)")
            self.writeline("programs_num = tl.num_programs(0)")
            if self.MODE == 0:
                self.writeline(
                    "tiles_per_tensor = tl.cdiv(high_num * tl.cdiv(out_cat_num, tensor_num), BLOCK_LOW)"
                )
                self.writeline("num_tiles = tiles_per_tensor * tensor_num")
                self.writeline("tiles_per_cta = tl.cdiv(num_tiles, programs_num)")
                self.writeline("for i in range(tiles_per_cta):")
                with self.indent():
                    self.writeline("tile_id = pid + i * programs_num")
                    self.writeline("tensor_id = tile_id // tiles_per_tensor")
                    self.writeline("tile_id = tile_id % tiles_per_tensor")
                    for j in range(self.tensor_num):
                        self.writeline(f"if tensor_id == {j}:")
                        with self.indent():
                            self.writeline(
                                f"low_cat = low_cat_accum{j + 1} - low_cat_accum{j}"
                            )
                            self.writeline("offsets = tl.arange(0, BLOCK_LOW)")
                            self.writeline("in_offsets = tile_id * BLOCK_LOW + offsets")
                            self.writeline("mask = in_offsets < high_num * low_cat")
                            self.writeline(
                                f"data = tl.load(in{j} + in_offsets, mask=mask)"
                            )
                            high_part = "(in_offsets // low_cat) * out_cat_num"
                            low_part = f"low_cat_accum{j} + (in_offsets % low_cat)"
                            self.writeline(f"out_offsets = {high_part} + {low_part}")
                            self.writeline(
                                "tl.store(out + out_offsets, data, mask=mask)"
                            )
            elif self.MODE == 1:
                self.writeline("num_tiles = tl.cdiv(high_num, BLOCK_HIGH)")
                self.writeline("tiles_per_cta = tl.cdiv(num_tiles, programs_num)")
                self.writeline("for i in range(tiles_per_cta):")
                with self.indent():
                    self.writeline("tile_id = pid + i * programs_num")
                    self.writeline("high_offset = tile_id * BLOCK_HIGH")
                    for j in range(self.tensor_num):
                        self.writeline(
                            f"low_cat = low_cat_accum{j + 1}-low_cat_accum{j}"
                        )
                        self.writeline(
                            "for low_offset in range(0, low_cat, BLOCK_LOW):"
                        )
                        with self.indent():
                            self.writeline(
                                "high_offsets = high_offset + tl.arange(0, BLOCK_HIGH)"
                            )
                            self.writeline(
                                "low_offsets = low_offset + tl.arange(0, BLOCK_LOW)"
                            )
                            high_part = f"high_offsets[:, None] * in{j}_stride_high"
                            low_part = f"low_offsets[None, :] * in{j}_stride_low"
                            self.writeline(f"in_offsets = {high_part} + {low_part}")
                            self.writeline(
                                "in_mask = (high_offsets < high_num)[:,None] & (low_offsets < low_cat)[None,:]"
                            )
                            self.writeline(
                                f"data = tl.load(in{j}+in_offsets, mask=in_mask)"
                            )
                            high_part = "high_offsets[:, None] * out_stride_high"
                            low_part = f"(low_cat_accum{j} + low_offsets[None, :]) * out_stride_low"
                            self.writeline(f"out_offsets = {high_part} + {low_part}")
                            self.writeline(
                                "tl.store(out+out_offsets, data, mask=in_mask)"
                            )

    def gen_kernel_args(self, is_declare=True):
        in_args = ", ".join(
            f"in{i}" if is_declare else f"tensors[{i}]" for i in range(self.tensor_num)
        )
        low_cat_accum_args = ", ".join(
            f"low_cat_accum{i}" if is_declare else f"low_cat_accum[{i}]"
            for i in range(self.tensor_num + 1)
        )
        stride_args = (
            ", ".join(
                f"in{i}_stride_high, in{i}_stride_low" for i in range(self.tensor_num)
            )
            + ", out_stride_high, out_stride_low"
        )

        kernel_args = f"{in_args}, out, {stride_args}, tensor_num, high_num, {low_cat_accum_args}, out_cat_num, "
        ex_args = "BLOCK_LOW: tl.constexpr, num_warps: tl.constexpr"
        if self.MODE == 1:
            ex_args += ", BLOCK_HIGH: tl.constexpr"

        return kernel_args if not is_declare else kernel_args + ex_args

    def codegen(self):
        self.gen_imports()
        self.gen_wrapper()
        self.gen_decorators()
        self.gen_kernel()


def cat(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON CAT")

    # Check empty inputs.
    if len(tensors) == 0:
        raise RuntimeError(
            "Expected a non-empty list or tuple/list of non-empty torch.Tensor"
        )

    # Check dimensions.
    ndim = tensors[0].ndim
    assert dim >= -ndim and dim < ndim, f"Invalid concat dimension: {dim}"
    dim %= ndim

    # Check shapes and zero element tensors.
    device = tensors[0].device
    dtypes = [t.dtype for t in tensors]
    dtype = dtypes[0]
    for ty in dtypes[1:]:
        dtype = torch.promote_types(dtype, ty)
    shape = tensors[0].shape
    valid_tensors = []

    for _, tensor in enumerate(tensors):
        assert (
            tensor.ndim == ndim
        ), f"Requires same ndim of inputs, but got {ndim} and {tensor.ndim}"
        assert (
            tensor.device == device
        ), f"Requires same device of inputs, but got {device} and {tensor.device}"
        for d_idx, (size, base_size) in enumerate(zip(tensor.shape, shape)):
            assert (
                dim == d_idx or size == base_size
            ), f"Requires same dim sizes of dim {d_idx}, but got {size} and {base_size}"
        if tensor.numel() != 0:
            tensor = tensor.contiguous()
            valid_tensors.append(tensor.to(dtype) if tensor.dtype != dtype else tensor)

    tensor_num = len(valid_tensors)

    # Deal with special cases.
    if tensor_num == 1:
        return valid_tensors[0]

    cat_dim_sizes = [_.shape[dim] for _ in tensors]
    out_shape = list(tensors[0].shape)
    out_shape[dim] = sum(cat_dim_sizes)

    if tensor_num == 0:
        return torch.empty(out_shape, dtype=dtype, device=device)

    # Preprocess kernel parameters.
    high_num = int(math.prod(out_shape[:dim]))
    low_num = int(math.prod(out_shape[dim + 1 :]))
    out_cat_num = 0
    low_cat_accum = [0]

    for size in cat_dim_sizes:
        out_cat_num += size * low_num
        low_cat_accum.append(out_cat_num)

    # Launch kernel.
    if high_num == 1:
        # Vstack and Concat results in the same storage arrangement when high_num == 1.
        valid_tensors = [t.view(t.shape[dim], -1) for t in valid_tensors]
        return vstack(valid_tensors).view(out_shape)
    else:
        # Dealing with concat situations that having arbitary nums of inputs via template code genertaor.
        return CatKernelGenerator()(valid_tensors, dim, high_num, low_cat_accum)
