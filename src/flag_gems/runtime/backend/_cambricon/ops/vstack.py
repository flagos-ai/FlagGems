import importlib
import logging
import math
import os
from typing import Callable, Mapping

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


class VstackKernelCode(IndentedBuffer):
    """
    Vstack kernel template.
    """

    overloads: Mapping[str, Callable] = {}

    def __init__(self):
        self.pid = os.getpid()
        self.cache = self.overloads
        self.kernel_name = "_vstack_jit_kernel"
        self.wrapper_func_name = "_wrapper"
        self.vstack_small_limit = 49152
        super(VstackKernelCode, self).__init__()

    def __init(self, tensors):
        """Initialize the vstack kernel."""
        self.device = tensors[0].device
        dtypes = [t.dtype for t in tensors]
        dtype = dtypes[0]
        for ty in dtypes[1:]:
            dtype = torch.promote_types(dtype, ty)
        self.dtype = dtype
        for i, tensor in enumerate(tensors):
            assert (
                tensor.device == self.device
                and tensors[0].shape[1:] == tensor.shape[1:]
            )
            if tensor.dtype != self.dtype:
                tensors[i] = tensor.to(self.dtype)
        c_tensors = [t.contiguous() for t in tensors]
        self.inputs = []
        self.idxs = [0]
        self.total_size = 0
        for tensor in c_tensors:
            self.total_size += tensor.numel()
            self.idxs.append(self.total_size)
            self.inputs.append(tensor)
        self.deal_num = math.ceil(self.total_size / TOTAL_CORE_NUM)
        self.input_num = len(self.inputs)
        flag = (self.total_size / self.input_num) == self.idxs[1]
        if (
            self.total_size < self.vstack_small_limit
            and self.input_num <= TOTAL_CORE_NUM
            and flag
        ):
            self.is_small = True
        else:
            self.is_small = False

    def __imports(self):
        """Generate imports for the kernel code."""
        self.tpl(
            """
import math
import torch
import triton
from triton import language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.runtime.backend import vendor_module
TOTAL_CORE_NUM = vendor_module.TOTAL_CORE_NUM
MAX_NRAM_SIZE = vendor_module.MAX_NRAM_SIZE
        """
        )

    def __wrapper(self):
        """Generate wrapper function for the kernel code."""
        self.newline()
        self.tpl(
            """
def {wrapper_name}(tensors, inputs, idx, total_size, input_num, deal_num, is_small):
    tensors = torch.atleast_2d(tensors)
    num_tensors = len(tensors)
    assert num_tensors > 0
    if num_tensors == 1:
        return tensors[0]
    device = tensors[0].device
    dtype = tensors[0].dtype
    input = [i for i in inputs]
    c_tensors = [t.contiguous() for t in tensors]
    total_rows = sum(tensor.shape[0] for tensor in c_tensors)
    output_shape = list(c_tensors[0].shape)
    output_shape[0] = total_rows
    output = torch.empty(output_shape, device=device, dtype=dtype)
    with torch_device_fn.device(device):
        {kernel_name}[(TOTAL_CORE_NUM,)]({args})
    return output
        """,
            wrapper_name=self.wrapper_func_name,
            kernel_name=self.kernel_name,
            args=self.__kernel_args(is_declare=False),
        )

    def __config(self):
        """Generate config for the kernel code."""
        # generate config key.
        self.newline()
        self.tpl(
            """
@libentry()
@libtuner(
    configs=[
        triton.Config({{'BLOCK_SIZE' : 512}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 2048}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 4096}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 8192}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 10240}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 14336}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 18000}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 22000}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 28000}}, num_warps=1),
        triton.Config({{'BLOCK_SIZE' : 32000}}, num_warps=1),
    ],
    key = [{config_keys}],
)
@triton.jit
        """,
            config_keys="'total_size'",
        )

    def __kernel(self):
        """Generate kernel code body."""
        # configuration.
        self.__config()
        kernel_signature = f"def {self.kernel_name}({self.__kernel_args()}):"
        self.idx_1 = 1
        self.idx_0 = 0
        self.writeline(kernel_signature)
        with self.indent():
            self.writeline("pid_x = tl.program_id(0)")
            self.writeline("block = tl.arange(0, BLOCK_SIZE)")
            self.writeline("if is_small:")
            with self.indent():
                self.writeline("for i in range(input_num):")
                with self.indent():
                    for i in range(self.input_num):
                        self.writeline(f"if pid_x == {i} and pid_x == i:")
                        with self.indent():
                            self.writeline(
                                f"for num in range(0, idx_{i + 1} - idx_{i}, BLOCK_SIZE):"
                            )
                            with self.indent():
                                self.writeline("in_offset = num + block")
                                self.writeline(f"dst_offset = idx_{i} + num + block")
                                self.writeline(
                                    f"x = tl.load(input_{i} + in_offset, mask = in_offset < idx_{i + 1} - idx_{i})"
                                )
                                self.writeline(
                                    f"tl.store(output + dst_offset, x, mask = dst_offset < idx_{i + 1})"
                                )
            self.writeline("else:")
            with self.indent():
                self.writeline("condidate_num = idx_1")
                self.writeline("input_iter = 0")
                self.writeline("for pid in range(pid_x + 1):")
                with self.indent():
                    self.writeline("need_num = deal_num")
                    self.writeline("while(need_num > 0):")
                    with self.indent():
                        self.writeline("per_fetch_num = min(condidate_num, need_num)")
                        self.writeline("if pid == pid_x:")
                        with self.indent():
                            self.writeline("if input_iter == 0:")
                            with self.indent():
                                self.writeline("offset = idx_1 - idx_0 - condidate_num")
                                self.writeline("deal_rem = deal_num - per_fetch_num")
                                self.writeline(
                                    "for i in range(0, deal_num, BLOCK_SIZE):"
                                )
                                with self.indent():
                                    self.writeline("in_offset = offset + i + block")
                                    self.writeline("dst_offset = in_offset")
                                    self.writeline(
                                        "x = tl.load(input_0 + in_offset, mask=in_offset < idx_1 - idx_0)"
                                    )
                                    self.writeline(
                                        "tl.store(output + dst_offset, x, mask=dst_offset<idx_1)"
                                    )
                            self.writeline("else:")
                            with self.indent():
                                for i in range(1, self.input_num, 1):
                                    idx = i + 1
                                    self.writeline(f"if input_iter == {i}:")
                                    with self.indent():
                                        self.writeline(
                                            f"offset = idx_{idx} - idx_{i} - condidate_num"
                                        )
                                        self.writeline("if need_num != deal_num:")
                                        with self.indent():
                                            self.writeline(
                                                "deal_rem = deal_num - per_fetch_num"
                                            )
                                            self.writeline(
                                                "for i in range(0, need_num, BLOCK_SIZE):"
                                            )
                                            with self.indent():
                                                self.writeline(
                                                    "in_offset = offset + i + block"
                                                )
                                                self.writeline(
                                                    f"dst_offset = idx_{i} + in_offset"
                                                )
                                                self.writeline(
                                                    f"x = tl.load(input_{i} + in_offset, mask=in_offset < need_num)"
                                                )
                                                self.writeline(
                                                    f"tl.store(output + dst_offset, x, \
                                                        mask=dst_offset<idx_{i}+per_fetch_num)"
                                                )
                                        self.writeline("else:")
                                        with self.indent():
                                            self.writeline(
                                                "for i in range(0, need_num, BLOCK_SIZE):"
                                            )
                                            with self.indent():
                                                self.writeline(
                                                    "in_offset = offset + i + block"
                                                )
                                                self.writeline(
                                                    f"dst_offset = idx_{i} + in_offset"
                                                )
                                                self.writeline(
                                                    f"x = tl.load(input_{i} + in_offset, \
                                                        mask=in_offset < idx_{idx}-idx_{i})"
                                                )
                                                self.writeline(
                                                    f"tl.store(output + dst_offset, x, mask=dst_offset<idx_{idx})"
                                                )
                        self.writeline("condidate_num -= per_fetch_num")
                        self.writeline("need_num -= per_fetch_num")
                        self.writeline("if (condidate_num <= 0):")
                        with self.indent():
                            for i in range(1, self.input_num, 1):
                                idx = i + 1
                                input_idx = i - 1
                                if self.input_num == 2:
                                    self.writeline(
                                        f"condidate_num = idx_{idx} - idx_{i}"
                                    )
                                else:
                                    if i == 1:
                                        self.writeline(f"if input_iter == {input_idx}:")
                                        with self.indent():
                                            self.writeline(
                                                f"condidate_num = idx_{idx} - idx_{i}"
                                            )
                                    else:
                                        if i < self.input_num - 1:
                                            self.writeline(
                                                f"elif input_iter == {input_idx}:"
                                            )
                                            with self.indent():
                                                self.writeline(
                                                    f"condidate_num = idx_{idx} - idx_{i}"
                                                )
                                        else:
                                            self.writeline("else:")
                                            with self.indent():
                                                self.writeline(
                                                    f"condidate_num = idx_{idx} - idx_{i}"
                                                )

                            self.writeline("input_iter += 1")

    def __gen_code(self):
        """Entry point for code generation of vstack."""
        # generate imports.
        self.__imports()
        # generate wrapper function.
        self.__wrapper()

        # generate kernel.
        self.__kernel()

    def __kernel_args(self, is_declare=True):
        input_args = []
        idxs_args = []
        if is_declare:
            for i in range(self.input_num):
                input_args.append(f"input_{i}")
            for i in range(len(self.idxs)):
                idxs_args.append(f"idx_{i}")
        else:
            for i in range(self.input_num):
                input_args.append(f"input[{i}]")
            for i in range(len(self.idxs)):
                idxs_args.append(f"idx[{i}]")
        input_args_str = ", ".join(input_args)
        idxs_args_str = ", ".join(idxs_args)

        extra_args_str = f"{input_args_str}, {idxs_args_str}"
        if is_declare:
            return f"{extra_args_str}, output, total_size, input_num, deal_num, is_small, BLOCK_SIZE: tl.constexpr"
        else:
            return (
                f"{extra_args_str}, output, total_size, input_num, deal_num, is_small"
            )

    def __call__(self, tensors: list) -> torch.Tensor:
        # get overload kernel.
        self.__init(tensors)

        vstack_input_num = "_".join(str(self.input_num))

        self.kernel_name = self.kernel_name + "_vstack_" + vstack_input_num
        key = f"{self.total_size}_{self.input_num}"
        if key not in self.cache:
            # generate code and cache.
            self.__gen_code()
            file_name = f"vstack_{key}_pid_{self.pid}.py"
            filepath = cache_dir() / file_name
            write_atomic(filepath, self.getvalue())
            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_{key}_pid_{self.pid}", filepath
            )
            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, self.wrapper_func_name)
            self.cache[key] = overload
        overload = self.cache[key]
        return overload(
            tensors,
            self.inputs,
            self.idxs,
            self.total_size,
            self.input_num,
            self.deal_num,
            self.is_small,
        )


def vstack(tensors: list):
    logger.debug("GEMS_CAMBRICON VSTACK")

    return VstackKernelCode()(tensors)
