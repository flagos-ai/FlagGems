import importlib
import logging
import os
from typing import Callable, Mapping

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


class FlipKernelCode(IndentedBuffer):
    """
    Flip kernel template.
    """

    overloads: Mapping[str, Callable] = {}

    def __init__(self):
        self.pid = os.getpid()
        self.cache = self.overloads
        self.kernel_name = "_flip_jit_kernel"
        self.wrapper_func_name = "_wrapper"
        super(FlipKernelCode, self).__init__()

    def __init(self, x, dims):
        """Initialize the flip kernel."""
        dim_size = x.dim()

        flip_dims = list(dims)
        flip_dims_flags = [False for _ in x.stride()]
        for i in range(len(flip_dims)):
            dim = flip_dims[i]
            assert (
                dim >= -dim_size and dim < dim_size
            ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -dim_size, dim_size - 1, dim
            )
            if dim < 0:
                flip_dims[i] = dim_size + dim
            assert not flip_dims_flags[
                dim
            ], "dim {} appears multiple times in the list of dims".format(dim)
            flip_dims_flags[dim] = True

        # merge shapes and flip_dims_flags by flip flags.
        self.merge_shapes = []
        self.merge_strides = []
        flag = flip_dims_flags[0]
        self.merge_flip_dims_flags = []
        self.merge_flip_dim = 0
        shape = 1
        for i in range(dim_size):
            if (flag == flip_dims_flags[i]) or x.shape[i] == 1:
                shape *= x.shape[i]
            else:
                self.merge_shapes.append(shape)
                self.merge_strides.append(x.stride(i - 1))
                self.merge_flip_dims_flags.append(flag)
                if flag:
                    self.merge_flip_dim += 1
                flag = flip_dims_flags[i]
                shape = x.shape[i]
        self.merge_shapes.append(shape)
        self.merge_strides.append(1)
        self.merge_flip_dims_flags.append(flag)
        if flag:
            self.merge_flip_dim += 1

        self.merge_dim_size = len(self.merge_shapes)

    def __imports(self):
        """Generate imports for the kernel code."""
        self.tpl(
            """
import math
import torch
import triton
from triton import language as tl

from flag_gems.utils import libentry
from flag_gems.runtime.backend import vendor_module
TOTAL_CORE_NUM = vendor_module.utils.TOTAL_CORE_NUM
MAX_NRAM_SIZE = vendor_module.utils.MAX_NRAM_SIZE


        """
        )

    def __wrapper(self):
        """Generate wrapper function for the kernel code."""
        self.newline()
        self.tpl(
            """
def {wrapper_name}(x, merge_shapes, merge_strides, merge_dim_size):
    if merge_dim_size == 0 or x.numel() <= 1:
        return x.clone()

    low_task = merge_shapes[merge_dim_size - 1]
    sub_dim = 1

    high_task = 1
    if merge_dim_size > 1:
        sub_dim = merge_shapes[merge_dim_size - 2]
        low_task *= sub_dim
        for i in range(merge_dim_size - 2):
            high_task *= merge_shapes[i]
    y = 1
    if high_task < TOTAL_CORE_NUM:
        for i in range(1, sub_dim + 1):
            if sub_dim % i == 0:
                y = i
                if y * high_task >= TOTAL_CORE_NUM:
                    break

    grid = lambda meta: (min(high_task, TOTAL_CORE_NUM), y, )

    # in case of one-dim.
    if (high_task == 1) and (y == 1) and (merge_dim_size == 1):
        if low_task <= 1024:
            grid = lambda meta: (1, 1, )
        else:
            grid = lambda meta: (1, TOTAL_CORE_NUM, )

    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        {kernel_name}[grid]({args})
    return out
        """,
            wrapper_name=self.wrapper_func_name,
            kernel_name=self.kernel_name,
            args=self.__kernel_args(is_declare=False),
        )

    def __config(self):
        """Generate config for the kernel code."""
        # generate config key.
        merge_shapes_args_str = ", ".join(
            [f"'merge_shape_{i}'" for i in range(self.merge_dim_size)]
        )
        merge_strides_args_str = ", ".join(
            [f"'merge_stride_{i}'" for i in range(self.merge_dim_size)]
        )

        self.newline()
        self.tpl(
            """

def get_h_dim(args):
    merge_dim_size = args['merge_dim_size'];
    high = 0
    if merge_dim_size > 1:
        high = args['merge_shape_{merge_dim_size_2}']
    width = args['merge_shape_{merge_dim_size_1}']
    max_nram_size = 3072
    if max_nram_size >= width:
        tmp_h = max_nram_size // width
        if tmp_h < high:
            return tmp_h
        return high
    return 0

def get_w_dim(args):
    merge_dim_size = args['merge_dim_size'];
    width = args['merge_shape_{merge_dim_size_1}']
    max_nram_size = 3072
    if max_nram_size >= width:
        return width
    return max_nram_size

@libentry()
@triton.autotune(
    configs=[
        triton.Config({{}}, num_stages=3, num_warps=1),
    ],
    key = [{config_keys}],
)
@triton.heuristics(
    values={{
        "H_DIM": get_h_dim,
        "W_DIM": get_w_dim,
    }},
)
@triton.jit
        """,
            merge_dim_size_2=str(self.merge_dim_size - 2),
            merge_dim_size_1=str(self.merge_dim_size - 1),
            config_keys=f"'x_ptr', {merge_shapes_args_str}, {merge_strides_args_str}",
        )

    def __kernel_flip_2d(self):
        """Generate kernel for 2d buffer flip."""
        self.writeline(f"step = merge_shape_{self.merge_dim_size - 2} // num_y")
        self.writeline(
            f"src_offset += pid_y * step * merge_shape_{self.merge_dim_size - 1}"
        )
        if self.merge_flip_dims_flags[self.merge_dim_size - 2]:
            # [flip, no-flip]
            self.writeline("# flip low-2d [flip, no-flip]")
            self.writeline(
                f"dst_offset += (num_y - pid_y - 1) * step * merge_shape_{self.merge_dim_size - 1}"
            )
            self.writeline("if H_DIM != 0:")
            with self.indent():
                self.writeline(
                    "offset = tl.arange(0, H_DIM)[:,None]*W_DIM + tl.arange(0, W_DIM)[None,:]"
                )
                self.writeline("tail = step % H_DIM")
                self.writeline("iter = step // H_DIM")
                self.writeline("for i in range(0, iter):")
                with self.indent():
                    self.writeline("in_offset = src_offset + i * H_DIM*W_DIM")
                    self.writeline(
                        "out_offset = dst_offset + tail * W_DIM + (iter - i - 1) * H_DIM*W_DIM"
                    )
                    self.writeline(
                        "src = tl.load(x_ptr + offset + in_offset, cache_modifier='.cg')"
                    )
                    self.writeline("src = tl.flip(src, [0])")
                    self.writeline(
                        "tl.store(out_ptr + offset + out_offset, src, cache_modifier='.cg')"
                    )
                self.writeline("if tail > 0:")
                with self.indent():
                    self.writeline("# process tail.")
                    self.writeline("in_offset = src_offset + iter * H_DIM*W_DIM")
                    self.writeline("out_offset = dst_offset - (H_DIM-tail)*W_DIM")
                    self.writeline("mask = offset < tail*W_DIM")
                    self.writeline(
                        "src = tl.load(x_ptr + offset + in_offset, mask=mask, other=0.0, cache_modifier='.cg')"
                    )
                    self.writeline("src = tl.flip(src, [0])")
                    self.writeline("mask = offset >= (H_DIM - tail) * W_DIM")
                    self.writeline(
                        "tl.store(out_ptr + offset + out_offset, src, mask=mask, cache_modifier='.cg')"
                    )
            self.writeline("else:")
            with self.indent():
                self.writeline("offset = tl.arange(0, W_DIM)")
                self.writeline(f"iter = merge_shape_{self.merge_dim_size - 1} // W_DIM")
                self.writeline(f"tail = merge_shape_{self.merge_dim_size - 1} % W_DIM")
                self.writeline("src = tl.zeros((W_DIM,), dtype=x_ptr.dtype.element_ty)")
                self.writeline("for i in range(0, step):")
                with self.indent():
                    self.writeline(
                        f"in_offset = src_offset + i * merge_shape_{self.merge_dim_size - 1}"
                    )
                    self.writeline(
                        f"out_offset = dst_offset + (step - i - 1) * merge_shape_{self.merge_dim_size - 1}"
                    )
                    self.writeline("for j in range(0, iter):")
                    with self.indent():
                        self.writeline("new_offset = offset + j*W_DIM")
                        self.writeline(
                            "src = tl.load(x_ptr + in_offset + new_offset, cache_modifier='.cg')"
                        )
                        self.writeline(
                            "tl.store(out_ptr + out_offset + new_offset, src, cache_modifier='.cg')"
                        )
                    self.writeline("if tail > 0:")
                    with self.indent():
                        self.writeline("new_offset = offset + iter*W_DIM")
                        self.writeline("mask = offset < tail")
                        self.writeline(
                            "src = tl.load(x_ptr + in_offset + new_offset, mask=mask, cache_modifier='.cg')"
                        )
                        self.writeline(
                            "tl.store(out_ptr + out_offset + new_offset, src, mask=mask, cache_modifier='.cg')"
                        )
        else:
            # [no-flip, flip]
            self.writeline("# flip low-2d [no-flip, flip]")
            self.writeline(
                f"dst_offset += pid_y * step * merge_shape_{self.merge_dim_size - 1}"
            )
            self.writeline("if H_DIM != 0:")
            with self.indent():
                self.writeline(
                    "offset = tl.arange(0, H_DIM)[:,None]*W_DIM + tl.arange(0, W_DIM)[None,:]"
                )
                self.writeline("tail = step % H_DIM")
                self.writeline("iter = step // H_DIM")
                self.writeline("for i in range(0, iter):")
                with self.indent():
                    self.writeline("in_offset = src_offset + i * H_DIM*W_DIM")
                    self.writeline("out_offset = dst_offset + i * H_DIM*W_DIM")
                    self.writeline(
                        "src = tl.load(x_ptr + offset + in_offset, cache_modifier='.cg')"
                    )
                    self.writeline("src = tl.flip(src, [1])")
                    self.writeline(
                        "tl.store(out_ptr + offset + out_offset, src, cache_modifier='.cg')"
                    )
                self.writeline("if tail > 0:")
                with self.indent():
                    self.writeline("# process tail.")
                    self.writeline("in_offset = src_offset + iter * H_DIM*W_DIM")
                    self.writeline("out_offset = dst_offset + iter * H_DIM*W_DIM")
                    self.writeline("mask = offset < tail*W_DIM")
                    self.writeline(
                        "src = tl.load(x_ptr + offset + in_offset, mask=mask, other=0.0, cache_modifier='.cg')"
                    )
                    self.writeline("src = tl.flip(src, [1])")
                    self.writeline(
                        "tl.store(out_ptr + offset + out_offset, src, mask=mask, cache_modifier='.cg')"
                    )
            self.writeline("else:")
            with self.indent():
                self.writeline("offset = tl.arange(0, W_DIM)")
                self.writeline("src = tl.zeros((W_DIM,), dtype=x_ptr.dtype.element_ty)")
                self.writeline(f"tail = merge_shape_{self.merge_dim_size - 1} % W_DIM")
                self.writeline(f"iter = merge_shape_{self.merge_dim_size - 1} // W_DIM")
                self.writeline("for i in range(0, step):")
                with self.indent():
                    self.writeline(
                        f"in_offset = src_offset + i * merge_shape_{self.merge_dim_size - 1}"
                    )
                    self.writeline(
                        f"out_offset = dst_offset + i * merge_shape_{self.merge_dim_size - 1}"
                    )
                    self.writeline("if tail > 0:")
                    with self.indent():
                        self.writeline("new_offset = in_offset + iter * W_DIM")
                        self.writeline("mask = offset < tail")
                        self.writeline(
                            "src = tl.load(x_ptr + new_offset + offset, mask=mask, cache_modifier='.cg')"
                        )
                        self.writeline("src = tl.flip(src, [0])")
                        self.writeline("mask = offset >= (W_DIM-tail)")
                        self.writeline(
                            "tl.store(out_ptr + out_offset - (W_DIM - tail) + offset, \
                                src, mask=mask, cache_modifier='.cg')"
                        )
                    self.writeline("for j in range(0, iter):")
                    with self.indent():
                        self.writeline("new_in_offset = in_offset + j * W_DIM")
                        self.writeline(
                            "new_out_offset = tail + out_offset + (iter - j - 1) * W_DIM"
                        )
                        self.writeline(
                            "src = tl.load(x_ptr + new_in_offset + offset, cache_modifier='.cg')"
                        )
                        self.writeline("src = tl.flip(src, [0])")
                        self.writeline(
                            "tl.store(out_ptr + new_out_offset + offset, src, cache_modifier='.cg')"
                        )

    def __kernel(self):
        """Generate kernel code body."""
        # configuration.
        self.__config()
        kernel_signature = f"def {self.kernel_name}({self.__kernel_args()}):"
        self.writeline(kernel_signature)
        with self.indent():
            self.writeline("pid_x = tl.program_id(0)")
            self.writeline("num_x = tl.num_programs(0)")
            self.writeline("pid_y = tl.program_id(1)")
            self.writeline("num_y = tl.num_programs(1)")
            # iteration on high dimension.
            self.writeline("for high_id in range(pid_x, high_task, num_x):")
            with self.indent():
                self.writeline("src_offset = 0")
                self.writeline("dst_offset = 0")
                self.writeline("temp_high_id = high_id")
                # get src_offset and dst offset
                if self.merge_dim_size > 2:
                    for i in range(self.merge_dim_size - 2):
                        self.writeline(f"tmp_stride = merge_stride_{i} // low_task")
                        self.writeline(f"id_{i} = temp_high_id // tmp_stride")
                        self.writeline("temp_high_id = temp_high_id % tmp_stride")
                        self.writeline(f"src_offset += id_{i} * merge_stride_{i}")
                        if not self.merge_flip_dims_flags[i]:
                            self.writeline(f"dst_offset += id_{i} * merge_stride_{i}")
                        else:
                            self.writeline(
                                f"dst_offset += (merge_shape_{i} - id_{i} -1) * merge_stride_{i}"
                            )
                    self.__kernel_flip_2d()
                elif self.merge_dim_size == 2:
                    self.__kernel_flip_2d()
                elif self.merge_dim_size == 1:
                    assert self.merge_flip_dims_flags[0]
                    self.writeline("offset = tl.arange(0, W_DIM)")
                    self.writeline(
                        f"step = merge_shape_{self.merge_dim_size - 1} // num_y"
                    )
                    self.writeline(
                        f"tail = merge_shape_{self.merge_dim_size - 1} % num_y"
                    )
                    self.writeline("# process step.")
                    self.writeline("src_offset = pid_y * step")
                    self.writeline("dst_offset = tail + (num_y - pid_y - 1) * step")
                    self.writeline("step_iter = step // W_DIM")
                    self.writeline("step_tail = step % W_DIM")
                    self.writeline("for i in range(0, step_iter):")
                    with self.indent():
                        self.writeline("in_offset = src_offset + i * W_DIM")
                        self.writeline(
                            "out_offset = dst_offset + step_tail + (step_iter - i - 1) * W_DIM"
                        )
                        self.writeline(
                            "src = tl.load(x_ptr + offset + in_offset, cache_modifier='.cg')"
                        )
                        self.writeline("src = tl.flip(src, [0])")
                        self.writeline(
                            "tl.store(out_ptr + offset + out_offset, src, cache_modifier='.cg')"
                        )
                    self.writeline("if step_tail > 0:")
                    with self.indent():
                        self.writeline("in_offset = src_offset + step_iter * W_DIM")
                        self.writeline("out_offset = dst_offset")
                        self.writeline("mask = offset < step_tail")
                        self.writeline(
                            "src = tl.load(x_ptr + offset + in_offset, mask=mask, cache_modifier='.cg')"
                        )
                        self.writeline("src = tl.flip(src, [0])")
                        self.writeline("mask = offset >= (W_DIM - step_tail)")
                        self.writeline(
                            "tl.store(out_ptr + offset + out_offset - (W_DIM - step_tail), \
                                src, mask=mask, cache_modifier='.cg')"
                        )
                    self.writeline("if pid_y == num_y - 1:")
                    with self.indent():
                        self.writeline("# process tail.")
                        self.writeline("src_offset = num_y * step")
                        self.writeline("dst_offset = 0")
                        self.writeline("tail_iter = tail // W_DIM")
                        self.writeline("tail_remain = tail % W_DIM")
                        self.writeline("for i in range(0, tail_iter):")
                        with self.indent():
                            self.writeline("in_offset = src_offset + i * W_DIM")
                            self.writeline(
                                "out_offset = dst_offset + tail_remain + (tail_iter - i - 1) * W_DIM"
                            )
                            self.writeline(
                                "src = tl.load(x_ptr + offset + in_offset, cache_modifier='.cg')"
                            )
                            self.writeline("src = tl.flip(src, [0])")
                            self.writeline(
                                "tl.store(out_ptr + offset + out_offset, src, cache_modifier='.cg')"
                            )
                        self.writeline("if tail_remain > 0:")
                        with self.indent():
                            self.writeline("in_offset = src_offset + tail_iter * W_DIM")
                            self.writeline("out_offset = dst_offset")
                            self.writeline("mask = offset < tail_remain")
                            self.writeline(
                                "src = tl.load(x_ptr + offset + in_offset, mask=mask, cache_modifier='.cg')"
                            )
                            self.writeline("src = tl.flip(src, [0])")
                            self.writeline("mask = offset >= (W_DIM-tail_remain)")
                            self.writeline(
                                "tl.store(out_ptr + offset + out_offset - (W_DIM - tail_remain), \
                                    src, mask=mask, cache_modifier='.cg')"
                            )
                else:
                    raise RuntimeError(f"merge dim size error({self.merge_dim_size})")

    def __gen_code(self):
        """Entry point for code generation of flip."""
        # generate imports.
        self.__imports()
        # generate wrapper function.
        self.__wrapper()

        # generate kernel.
        self.__kernel()

    def __kernel_args(self, is_declare=True):
        """Generate string type of jit kernel arguments."""
        merge_shapes_args = []
        merge_strides_args = []
        for i in range(self.merge_dim_size):
            if is_declare:
                merge_shapes_args.append(f"merge_shape_{i}")
                merge_strides_args.append(f"merge_stride_{i}")
            else:
                merge_shapes_args.append(f"merge_shapes[{i}]")
                merge_strides_args.append(f"merge_strides[{i}]")
        merge_shapes_args_str = ", ".join(merge_shapes_args)
        merge_strides_args_str = ", ".join(merge_strides_args)

        extra_args_str = f"{merge_shapes_args_str}, {merge_strides_args_str}"
        if is_declare:
            return f"x_ptr, out_ptr, {extra_args_str}, merge_dim_size, high_task: tl.constexpr, \
                low_task: tl.constexpr, H_DIM: tl.constexpr, W_DIM: tl.constexpr"
        else:
            return f"x, out, {extra_args_str}, merge_dim_size, high_task, low_task"

    def __call__(self, x: torch.Tensor, dims) -> torch.Tensor:
        """Call flip kernel."""
        # initialize the funtion.
        # note:
        # - This function must be call first and only once.
        self.__init(x, dims)
        if (self.merge_flip_dim == 0) or (self.merge_dim_size == 0 or x.numel() <= 1):
            return x.clone()
        # get overload kernel.
        flip_dim_str = "_".join([str(i) for i in self.merge_flip_dims_flags])
        self.kernel_name = self.kernel_name + "_flip_" + flip_dim_str
        key = f"{self.merge_dim_size}_{flip_dim_str}"
        if key not in self.cache:
            # generate code and cache.
            self.__gen_code()

            file_name = f"flip_{key}_pid_{self.pid}.py"
            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(self.getvalue())
            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_{key}_pid_{self.pid}", f.name
            )
            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, self.wrapper_func_name)
            self.cache[key] = overload

        overload = self.cache[key]
        return overload(x, self.merge_shapes, self.merge_strides, self.merge_dim_size)


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON FLIP")
    if not A.is_contiguous():
        A = A.contiguous()
    return FlipKernelCode()(A, dims)
