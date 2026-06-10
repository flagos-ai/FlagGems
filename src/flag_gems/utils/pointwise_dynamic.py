import importlib
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import triton
from triton.runtime.jit import JITFunction

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.codegen_config_utils import CodeGenConfig, get_codegen_config
from flag_gems.utils.device_info import get_device_capability
from flag_gems.utils.shape_utils import (
    MemOverlap,
    all_c_contiguous,
    all_the_same_shape,
    all_the_same_stride,
    broadcast_shapes,
    broadcasted_stride,
    check_tensor_attributes,
    has_internal_overlapping,
)
from flag_gems.utils.tensor_wrapper import StridedBuffer
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion


# ------------------ Operation Description ---------------------------
def _type_name(type) -> str:
    "Render typename as string, work for both (bool, int, float, str) and torch.dtype object"
    if type in (bool, int, float, str):
        return type.__name__
    if isinstance(type, torch.dtype):
        return str(type)
    return str(type)


def _check_typed_list(container, type):
    for item in container:
        assert isinstance(item, type)


def _check_sized_list(container, size):
    assert len(container) == size


def _tuple_content(strings: Sequence[str]) -> str:
    # comma separated list
    if len(strings) == 0:
        return ""
    if len(strings) == 1:
        return f"{strings[0]},"
    else:
        return ", ".join(strings)


def _cs(strings: Iterable[str]) -> str:
    return ", ".join(strings)


def _broadcast_vec(i, ndim):
    axes = [":" if j == i else "None" for j in range(ndim)]
    return f"[{_cs(axes)}]"


class FunctionSchema:
    _num_inputs: int
    _is_tensor: List[bool]
    _dtypes: List[Optional[type]]

    _num_input_tensors: int
    _num_non_tensor_inputs: int

    _num_outputs: int
    _promotion_methods: List[Tuple[int, ...]]

    def __init__(
        self,
        *,
        num_inputs: Optional[int] = None,
        is_tensor: Optional[List[bool]] = None,
        dtypes: Optional[List[Optional[type]]] = None,
        num_outputs: Optional[int] = None,
        promotion_methods=None,
    ):
        if is_tensor is not None:
            _check_typed_list(is_tensor, bool)
        if dtypes is not None:
            _check_typed_list(dtypes, (type, type(None)))

        if promotion_methods is None:
            raise ValueError(
                "No type promotion method provided! You must provide type promotion method for each output!"
            )
        else:
            self._promotion_methods = self.canonicalize_promotion_methods(
                promotion_methods
            )
        if num_inputs is not None:
            self._num_inputs = num_inputs
            if is_tensor is not None:
                _check_sized_list(is_tensor, num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [True] * num_inputs

            if dtypes is not None:
                _check_sized_list(dtypes, num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * num_inputs
        elif is_tensor is not None:
            self._num_inputs = len(is_tensor)
            self._is_tensor = is_tensor
            if dtypes is not None:
                _check_sized_list(dtypes, self._num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * self._num_inputs
        elif dtypes is not None:
            self._num_inputs = len(dtypes)
            self._dtypes = dtypes
            if is_tensor is not None:
                _check_sized_list(is_tensor, self._num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [item is None for item in dtypes]
        else:
            raise ValueError(
                "Cannot create FunctionSchema when none of (num_inputs, is_tensor, dtypes) is specified."
            )

        if num_outputs is not None:
            self._num_outputs = num_outputs
            _check_sized_list(promotion_methods, num_outputs)
        else:
            self._num_outputs = len(promotion_methods)

        assert self._num_inputs >= 1
        assert self._num_outputs >= 1

        self._num_input_tensors = sum(self._is_tensor)
        self._num_non_tensor_inputs = self._num_inputs - self._num_input_tensors
        self._input_id = self._compute_input_id()

    @staticmethod
    def canonicalize_promotion_methods(promotion_methods):
        canonicalized = []
        for item in promotion_methods:
            *arg_indices, method = item
            canonicalized.append(
                (*arg_indices, ELEMENTWISE_TYPE_PROMOTION_KIND[method])
            )
        return canonicalized

    def num_inputs(self):
        # num of arguments, outputs not included
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def is_tensor(self, arg_id: int) -> bool:
        return self._is_tensor[arg_id]

    def input_type(self, arg_id) -> Optional[type]:
        return self._dtypes[arg_id]

    def output_type(self, i):
        return self._promotion_methods[i]

    def num_input_tensors(self) -> int:
        return self._num_input_tensors

    def num_output_tensors(self) -> int:
        return self._num_outputs

    def num_non_tensor_args(self) -> int:
        return self._num_non_tensor_inputs

    def signature(self, outputs_in_arg: bool = False) -> str:
        input_types = []
        for is_tensor, dtype in zip(self._is_tensor, self._dtypes):
            if is_tensor:
                input_types.append("StridedBuffer")
            else:
                if dtype is None:
                    input_types.append("scalar")
                else:
                    input_types.append(_type_name(dtype))

        output_types = []

        if outputs_in_arg:
            for i in range(self.num_outputs()):
                output_types.append(f"StridedBuffer(a{1}!)")
            input_types.extend(output_types)
        else:
            for _ in range(self.num_outputs()):
                output_types.append("StridedBuffer")
        sig = f'Pointwise: {", ".join(input_types)} -> {", ".join(output_types)}'
        return sig

    def _compute_input_id(self):
        input_tensor_index = 0
        non_tensor_index = 0
        mapping: List[int] = []
        for i in range(self.num_inputs()):
            if self.is_tensor(i):
                mapping.append(input_tensor_index)
                input_tensor_index += 1
            else:
                mapping.append(non_tensor_index)
                non_tensor_index += 1
        return mapping

    def input_index(self, idx):
        return self._input_id[idx]

    def __str__(self) -> str:
        return self.signature(outputs_in_arg=False)


class KernelGenerator:
    def __init__(
        self,
        function_schema: FunctionSchema,
        scalar_fn: triton.JITFunction,
        rank: int,
        name: str,
        config: CodeGenConfig,
    ):
        self.fx = function_schema
        self.fn = scalar_fn
        self.ndim = rank
        self.name = name
        self.config = config

        self.fn_name = scalar_fn.__name__
        self.fn_module = scalar_fn.__module__

    def gen_import_function(self, code: IndentedBuffer):
        code.writeline("@triton.jit")
        code.writemultiline(self.fn.src)
        code.newline()

    def gen_decorators(self, code):
        code.writeline("@libentry()")
        num_non_tensor_args = self.fx.num_non_tensor_args()
        if num_non_tensor_args > 0:
            # we do not specialize non tensor args since they are passed into the inlined function
            # which means that their values may not deserve specialization
            non_specialize_arg_names = [f"val{i}" for i in range(num_non_tensor_args)]
            code.writeline(f"@triton.jit(do_not_specialize={non_specialize_arg_names})")
        else:
            code.writeline("@triton.jit")

    def input_name(self, i):
        is_tensor = self.fx.is_tensor(i)
        name = "in" if is_tensor else "val"
        index = self.fx.input_index(i)
        return f"{name}{index}"

    def output_name(self, i):
        return f"out{i}"

    def gen_signature(self, code, with_block_pointer=False):
        code.writeline(f"def {self.name}(")
        with code.indent():
            input_tensor_index = 0
            non_tensor_index = 0
            output_tensor_index = 0

            schema = self.fx
            # signature: inputs ptrs & non tensor inputs
            for i in range(schema.num_inputs()):
                if schema.is_tensor(i):
                    code.writeline(
                        f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                    )
                    input_tensor_index += 1
                else:
                    if schema.input_type(i) is not None:
                        code.writeline(
                            f"val{non_tensor_index}: {_type_name(schema.input_type(i))},"
                        )
                    else:
                        code.writeline(f"val{non_tensor_index},")
                    non_tensor_index += 1

            # signature: output ptrs
            for i in range(schema.num_outputs()):
                code.writeline(
                    f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                output_tensor_index += 1

            # signature: strides, for each tensor arguments
            ndim = self.ndim
            if ndim > 0:
                # strides for inputs
                for i in range(schema.num_input_tensors()):
                    stride_args = _cs(f"in{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for in{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"in{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(f"{stride_order_args}, # stride order for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(f"out{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for out{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"out{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(
                            f"{stride_order_args}, # stride order for out{i}"
                        )

                # task space, used to reconstruct multi index
                task_space_args = _cs(f"s{i}" for i in range(ndim))
                code.writeline(f"{task_space_args}, # task_space")

                # number of tasks, used to compute mask
                code.writeline("num_tasks,")

            # tile size & tiles_per_cta, gsl style
            if ndim > 0:
                code.writeline("tiles_per_cta: int,")
                tile_sizes = _cs(f"tile_size{i}: tl.constexpr" for i in range(ndim))
                code.writeline(f"{tile_sizes},")
                code.writeline("one_tile_per_cta: tl.constexpr,")
        code.writeline("):")

    def gen_signature_1d_tile(self, code):
        code.writeline(f"def {self.name}(")
        with code.indent():
            input_tensor_index = 0
            non_tensor_index = 0
            output_tensor_index = 0

            schema = self.fx
            # signature: inputs ptrs & non tensor inputs
            for i in range(schema.num_inputs()):
                if schema.is_tensor(i):
                    code.writeline(
                        f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                    )
                    input_tensor_index += 1
                else:
                    if schema.input_type(i) is not None:
                        code.writeline(
                            f"val{non_tensor_index}: {_type_name(schema.input_type(i))},"
                        )
                    else:
                        code.writeline(f"val{non_tensor_index},")
                    non_tensor_index += 1

            # signature: output ptrs
            for i in range(schema.num_outputs()):
                code.writeline(
                    f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                output_tensor_index += 1

            # signature: strides, for each tensor arguments
            ndim = self.ndim
            if ndim > 0:
                # strides for inputs
                for i in range(schema.num_input_tensors()):
                    stride_args = _cs(f"in{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(f"out{i}_stride{j}: int" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for out{i}")

                # task space, used to reconstruct multi index
                task_space_args = _cs(f"s{i}" for i in range(ndim))
                code.writeline(f"{task_space_args}, # task_space")

                # number of tasks, used to compute mask
                code.writeline("num_tasks,")

            # tile size & tiles_per_cta, gsl style
            if ndim > 0:
                code.writeline("tiles_per_cta: int,")
                code.writeline("tile_size: tl.constexpr,")
                code.writeline("one_tile_per_cta: tl.constexpr,")
        code.writeline("):")

    def gen_num_tiles(self, code):
        # tile-grid size
        ndim = self.ndim
        for i in range(ndim):
            if i < ndim:
                code.writeline(f"num_tiles{i} = tl.cdiv(s{i}, tile_size{i})")

    def gen_body_for_0d(self, code):
        schema = self.fx
        inputs_to_scalar_fn = [self.input_name(i) for i in range(schema.num_inputs())]
        outputs_to_scalar_fn = [
            self.output_name(i) for i in range(schema.num_output_tensors())
        ]
        inputs_to_scalar_fn = _cs(inputs_to_scalar_fn)
        outputs_to_scalar_fn = _cs(outputs_to_scalar_fn)

        code.writeline("# loads")
        for i in range(schema.num_input_tensors()):
            code.writeline(
                f"in{i} = tl.load(in{i}_ptr).to(in{i}_ptr.type.element_ty) "
                "# workaround the bug on bool, we should use the pointer's dtype)"
            )
        code.newline()

        code.writeline("# compute")
        code.writeline(
            f"{outputs_to_scalar_fn} = {self.fn_name}({inputs_to_scalar_fn})"
        )
        code.newline()

        code.writeline("# stores")
        for i in range(schema.num_output_tensors()):
            code.writeline(
                f"tl.store(out{i}_ptr, out{i}.to(out{i}_ptr.type.element_ty))"
            )
        code.newline()
        return code

    # nd tile 1d grid kernel with block pointer
    def gen_body_one_tile_per_cta_with_bptr(self, code):
        ndim = self.ndim
        schema = self.fx

        # block pointer for each operand
        shape = _tuple_content(tuple(f"s{i}" for i in range(ndim)))
        offsets = _tuple_content(tuple(f"offset{i}" for i in range(ndim)))
        tile_sizes = _tuple_content(tuple(f"tile_size{i}" for i in range(ndim)))

        # reconstruct pid multi index
        code.writeline(
            "# pid multi index recontruction: we use c ordering, right axes changes fastest"
        )
        for i in reversed(range(ndim)):
            if i > 0:
                code.writeline(f"tile_id{i} = tile_id % num_tiles{i}")
                code.writeline(f"tile_id //= num_tiles{i}")
            else:
                code.writeline(f"tile_id{i} = tile_id")
        code.newline()

        # cta_offsets
        code.writeline("# tile offsets")
        for i in range(ndim):
            # Or else: AssertionError: Block pointers only support 32 bit
            # `offsets/block_shape`, add a `.to(tl.int32)` or use regular indexing
            # for 64 bit support
            code.writeline(f"offset{i} = (tile_id{i} * tile_size{i}).to(tl.int32)")

        # loads
        code.writeline("# loads")
        for i in range(schema.num_input_tensors()):
            strides = _tuple_content(tuple(f"in{i}_stride{j}" for j in range(ndim)))
            order = _tuple_content(tuple(f"in{i}_stride_order{j}" for j in range(ndim)))
            code.writeline(
                f"in{i}_bptr = tl.make_block_ptr("
                f"in{i}_ptr, ({shape}), ({strides}), ({offsets}), ({tile_sizes}), order=({order}))"
            )
            code.writeline(
                f"in{i} = tl.load(in{i}_bptr, boundary_check=({order})).to(in{i}_ptr.type.element_ty) "
                "# workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)"
            )
        code.newline()

        # compute
        # TODO: sepearate this part
        inputs_to_scalar_fn = [self.input_name(i) for i in range(schema.num_inputs())]
        outputs_to_scalar_fn = [
            self.output_name(i) for i in range(schema.num_output_tensors())
        ]
        inputs_to_scalar_fn = _cs(inputs_to_scalar_fn)
        outputs_to_scalar_fn = _cs(outputs_to_scalar_fn)

        code.writeline("# compute")
        code.writeline(
            f"{outputs_to_scalar_fn} = {self.fn_name}({inputs_to_scalar_fn})"
        )
        code.newline()

        # stores
        code.writeline(
            "# stores, note that store to block pointer does not automatically cast the value to the pointer's dtype"
        )
        for i in range(schema.num_output_tensors()):
            strides = _tuple_content(tuple(f"out{i}_stride{j}" for j in range(ndim)))
            order = _tuple_content(
                tuple(f"out{i}_stride_order{j}" for j in range(ndim))
            )
            code.writeline(
                f"out{i}_bptr = tl.make_block_ptr("
                f"out{i}_ptr, ({shape}), ({strides}), ({offsets}), ({tile_sizes}), order=({order}))"
            )
            code.writeline(
                f"tl.store(out{i}_bptr, out{i}.to(out{i}_bptr.type.element_ty), boundary_check=({order}))"
            )

    def gen_body_gsl_with_bptr(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in range(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_with_bptr(code)

    def gen_body_one_tile_per_cta_without_bptr(self, code):
        ndim = self.ndim
        schema = self.fx

        # reconstruct pid multi index
        code.writeline(
            "# pid multi index recontruction: we use c ordering, right axes changes fastest"
        )
        for i in reversed(range(ndim)):
            if i > 0:
                code.writeline(f"tile_id{i} = tile_id % num_tiles{i}")
                code.writeline(f"tile_id //= num_tiles{i}")
            else:
                code.writeline(f"tile_id{i} = tile_id")
        code.newline()

        # offsets
        for i in range(ndim):
            code.writeline(
                f"offsets{i} = tile_id{i} * tile_size{i} + tl.arange(0, tile_size{i})"
            )

        # masks
        for i in range(ndim):
            code.writeline(f"mask{i} = offsets{i} < s{i}")
        masks = tuple(f"mask{i}{_broadcast_vec(i, ndim)}" for i in range(ndim))
        mask_combine = " & ".join(masks)
        code.writeline(f"mask = {mask_combine}")

        # loads
        code.writeline("# loads")
        for i in range(schema.num_input_tensors()):
            offsets = tuple(
                f"offsets{j}{_broadcast_vec(j, ndim)} * in{i}_stride{j}"
                for j in range(ndim)
            )
            offset_combine = " + ".join(offsets)
            code.writeline(
                f"in{i} = tl.load(in{i}_ptr + {offset_combine}, mask=mask).to(in{i}_ptr.type.element_ty)"
            )

        code.newline()

        # compute
        # TODO: sepearate this part
        inputs_to_scalar_fn = [self.input_name(i) for i in range(schema.num_inputs())]
        outputs_to_scalar_fn = [
            self.output_name(i) for i in range(schema.num_output_tensors())
        ]
        inputs_to_scalar_fn = _cs(inputs_to_scalar_fn)
        outputs_to_scalar_fn = _cs(outputs_to_scalar_fn)

        code.writeline("# compute")
        code.writeline(
            f"{outputs_to_scalar_fn} = {self.fn_name}({inputs_to_scalar_fn})"
        )
        code.newline()

        # stores
        for i in range(schema.num_output_tensors()):
            offsets = tuple(
                f"offsets{j}{_broadcast_vec(j, ndim)} * out{i}_stride{j}"
                for j in range(ndim)
            )
            offset_combine = " + ".join(offsets)
            code.writeline(
                f"in{i} = tl.store(out{i}_ptr + {offset_combine}, out{i}, mask=mask)"
            )

    def gen_body_gsl_without_bptr(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in range(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_without_bptr(code)

    def codegen_nd_tile_with_bptr(self, code):
        """Generate kernel nd tile & 1d grid with gsl support with block pointer."""
        self.gen_import_function(code)
        self.gen_decorators(code)
        self.gen_signature(code, with_block_pointer=True)

        # function body for rank-0
        if self.ndim == 0:
            with code.indent():
                self.gen_body_for_0d(code)
            return code

        with code.indent():
            code.writeline("pid = tle.program_id(0)")
            self.gen_num_tiles(code)
            # monolitic kernel: one_tile_per_cta, it may requires a very large grid to compute
            code.writeline("if one_tile_per_cta: # monolitic kernel style")
            with code.indent():
                code.writeline("tile_id = pid")
                self.gen_body_one_tile_per_cta_with_bptr(code)
            # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
            code.writeline("else: # grid-stride-loop style kernel")
            with code.indent():
                self.gen_body_gsl_with_bptr(code)
        code.newline()
        return code

    def codegen_nd_tile_without_bptr(self, code):
        self.gen_import_function(code)
        self.gen_decorators(code)
        self.gen_signature(code, with_block_pointer=False)

        # function body for rank-0
        if self.ndim == 0:
            with code.indent():
                self.gen_body_for_0d(code)
            return code

        with code.indent():
            code.writeline("pid = tle.program_id(0)")
            self.gen_num_tiles(code)
            # monolitic kernel: one_tile_per_cta, it may requires a very large grid to compute
            code.writeline("if one_tile_per_cta: # monolitic kernel style")
            with code.indent():
                code.writeline("tile_id = pid")
                self.gen_body_one_tile_per_cta_without_bptr(code)
            # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
            code.writeline("else: # grid-stride-loop style kernel")
            with code.indent():
                self.gen_body_gsl_without_bptr(code)
        code.newline()
        return code

    def codegen_nd_tile(self, code):
        use_block_pointer = self.config.prefer_block_pointer
        if use_block_pointer:
            self.codegen_nd_tile_with_bptr(code)
        else:
            self.codegen_nd_tile_without_bptr(code)
        return code

    def gen_body_one_tile_per_cta_1d_tile(self, code):
        ndim = self.ndim
        schema = self.fx

        # tile id
        code.writeline("tid = tile_id * tile_size + tl.arange(0, tile_size)")
        code.writeline("mask = tid < num_tasks")

        # multi index reconstruction
        for i in reversed(range(ndim)):
            if i > 0:
                code.writeline(f"i{i} = tid % s{i}")
                code.writeline(f"tid //= s{i}")
            else:
                code.writeline(f"i{i} = tid")
        code.newline()

        # loads
        code.writeline("# loads")
        for i in range(schema.num_input_tensors()):
            offsets = tuple(f"i{j} * in{i}_stride{j}" for j in range(ndim))
            offset_combine = " + ".join(offsets)
            code.writeline(
                f"in{i} = tl.load(in{i}_ptr + {offset_combine}, mask=mask).to(in{i}_ptr.type.element_ty)"
            )

        code.newline()

        # compute
        # TODO: sepearate this part
        inputs_to_scalar_fn = [self.input_name(i) for i in range(schema.num_inputs())]
        outputs_to_scalar_fn = [
            self.output_name(i) for i in range(schema.num_output_tensors())
        ]
        inputs_to_scalar_fn = _cs(inputs_to_scalar_fn)
        outputs_to_scalar_fn = _cs(outputs_to_scalar_fn)

        code.writeline("# compute")
        code.writeline(
            f"{outputs_to_scalar_fn} = {self.fn_name}({inputs_to_scalar_fn})"
        )
        code.newline()

        # stores
        for i in range(schema.num_output_tensors()):
            offsets = tuple(f"i{j} * out{i}_stride{j}" for j in range(ndim))
            offset_combine = " + ".join(offsets)
            code.writeline(
                f"in{i} = tl.store(out{i}_ptr + {offset_combine}, out{i}, mask=mask)"
            )

    def gen_body_gsl_1d_tile(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in range(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_1d_tile(code)

    def codegen_1d_tile(self, code):
        """Generate kernel 1d tile & 1d grid with gsl support."""
        self.gen_import_function(code)
        self.gen_decorators(code)
        self.gen_signature_1d_tile(code)

        # function body for rank-0
        if self.ndim == 0:
            with code.indent():
                self.gen_body_for_0d(code)
            return code

        with code.indent():
            code.writeline("pid = tle.program_id(0)")
            # code.writeline("num_ctas = te.num_programs(0)")
            # monolitic kernel: one_tile_per_cta, it may requires a very large grid to compute
            code.writeline("if one_tile_per_cta: # monolitic kernel style")
            with code.indent():
                code.writeline("tile_id = pid")
                self.gen_body_one_tile_per_cta_1d_tile(code)
            # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
            code.writeline("else: # grid-stride-loop style kernel")
            with code.indent():
                self.gen_body_gsl_1d_tile(code)
        code.newline()
        return code


class WrapperGenerator:
    def __init__(
        self,
        function_schema: FunctionSchema,
        jit_fn_name: str,
        ndim: int,
        name: str,
        config: CodeGenConfig,
    ):
        self.fx = function_schema
        self.jit_fn_name = jit_fn_name
        self.ndim = ndim
        self.name = name
        self.config = config

    def input_name(self, i):
        is_tensor = self.fx.is_tensor(i)
        name = "in" if is_tensor else "val"
        index = self.fx.input_index(i)
        return f"{name}{index}"

    def output_name(self, i):
        return f"out{i}"

    def gen_signature(self, code: IndentedBuffer):
        # TODO: check if triton handles constexprs transitively
        schema = self.fx
        params: List[str] = []
        for i in range(schema.num_inputs()):
            if schema.is_tensor(i):
                params.append(
                    f"{self.input_name(i)}: Union[torch.Tensor, StridedBuffer]"
                )
            else:
                arg_type = schema.input_type(i)
                if arg_type is not None:
                    params.append(f"{self.input_name(i)}: {_type_name(arg_type)}")
                else:
                    params.append(f"{self.input_name(i)}")
        # NOTE: [the wrapper's signature and rules for passing parameters ]
        # input params: must be passed by position, since the names are renamed to
        # in0, in1, val0, val1, ..., So passing these parameters by keyword is wierd
        # So we enforce that these parameters must be passed by position.
        # maybe we can fix it later
        # output parameters: must be passed by keyword, since the scalar function
        # do not have output parameters(think of it as some scalar function, output
        # parameter does not make sense in this case.) They are added to allow destination
        # passing style API. Output parameter is convenient in cases where we want
        # to use some pre-defiend outputs(especially when they are some views of other
        # tensors). We emphasize that these parameters are added in-addition, we enforce
        # that they be passed by keyword. After all, out0, out1, ... does not mismatch
        # names form the scalar function, since it does not have output parameters.
        params.append("/")
        params.append("*")  # output params must be passed by keyword

        for i in range(schema.num_output_tensors()):
            params.append(f"{self.output_name(i)}: Union[torch.Tensor, StridedBuffer]")
        code.writeline(f"def {self.name}({_cs(params)}): ")

    def gen_docstring(self, code: IndentedBuffer):
        schema = self.fx
        doc = f'"""Generated wrapper function with {schema.signature(outputs_in_arg=True)}"""'
        code.writeline(doc)

    def gen_same_shape_check(self, code: IndentedBuffer):
        schema: FunctionSchema = self.fx
        params = [f"in{i}.shape" for i in range(schema.num_input_tensors())] + [
            f"out{i}.shape" for i in range(schema.num_output_tensors())
        ]
        check: str = " == ".join(params)
        code.writeline(f"assert {check}, 'operand shapes mismatch'")

    def gen_task_partition(self, code: IndentedBuffer):
        code.writeline("# task partitioning")
        ndim = self.ndim
        if ndim == 0:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        else:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = out0.numel()")
            code.writeline("if num_tasks == 0:")
            with code.indent():
                self.gen_return(code)
            max_tile_size = self.config.max_tile_size
            # Check if all input and output dtypes are complex
            all_complex = True
            for i in range(self.fx.num_inputs()):
                if self.fx.is_tensor(i):
                    input_dtype = self.fx.input_type(i)
                    if input_dtype is not None and not (
                        input_dtype == torch.complex64
                        or input_dtype == torch.complex128
                    ):
                        all_complex = False
                        break
            if all_complex:
                # If all inputs are complex, set max_tile_size to half
                max_tile_size = max_tile_size // 2
            major, _ = get_device_capability()
            if self.name.find("fill_scalar") != -1 and major >= 9:
                code.writeline("tile_sizes = tuple([64])")
            else:
                code.writeline(
                    f"tile_sizes = heuristics_for_tile_size({max_tile_size}, *shape)"
                )
            code.writeline("tile_size = math.prod(tile_sizes)")
            code.writeline(
                "num_tiles = math.prod(triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes))"
            )

            if self.name.find("fill_scalar") != -1 and major >= 9:
                code.writeline("num_ctas = num_tiles")
            else:
                max_grid_size0 = self.config.max_grid_size[0]
                code.writeline(f"num_ctas = min({max_grid_size0}, num_tiles)")

            code.writeline("tiles_per_cta = triton.cdiv(num_tiles, num_ctas)")
            code.writeline("num_warps = heuristics_for_num_warps(tile_size)")
            code.writeline("one_tile_per_cta = tiles_per_cta==1")
        code.writeline("grid = (num_ctas, 1, 1)")

    def gen_task_partition_1d(self, code: IndentedBuffer):
        code.writeline("# task partitioning")
        ndim = self.ndim
        if ndim == 0:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        else:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = out0.numel()")
            code.writeline("if num_tasks == 0:")
            with code.indent():
                self.gen_return(code)
            max_tile_size = self.config.max_tile_size
            # Check if all input and output dtypes are complex
            all_complex = True
            for i in range(self.fx.num_inputs()):
                if self.fx.is_tensor(i):
                    input_dtype = self.fx.input_type(i)
                    if input_dtype is not None and not (
                        input_dtype == torch.complex64
                        or input_dtype == torch.complex128
                    ):
                        all_complex = False
                        break
            if all_complex:
                max_tile_size = max_tile_size // 2
            major, _ = get_device_capability()
            if self.name.find("fill_scalar") != -1 and major >= 9:
                code.writeline("tile_sizes = tuple([1024])")
            else:
                code.writeline(
                    f"tile_sizes = heuristics_for_tile_size({max_tile_size}, num_tasks)"
                )

            code.writeline("tile_size = tile_sizes[0]")
            code.writeline("num_tiles = triton.cdiv(num_tasks, tile_size)")

            if self.name.find("fill_scalar") != -1 and major >= 9:
                code.writeline("num_ctas = num_tiles")
            else:
                max_grid_size0 = self.config.max_grid_size[0]
                code.writeline(f"num_ctas = min({max_grid_size0}, num_tiles)")

            code.writeline("tiles_per_cta = triton.cdiv(num_tiles, num_ctas)")
            code.writeline("num_warps = heuristics_for_num_warps(tile_size)")
            code.writeline("one_tile_per_cta = tiles_per_cta==1")
        code.writeline("grid = (num_ctas, 1, 1)")

    def gen_kernel_launch(
        self,
        code: IndentedBuffer,
    ):
        schema = self.fx
        ndim = self.ndim

        with_block_pointer = self.config.prefer_block_pointer

        code.writeline("# kernel launch")
        for i in range(schema.num_input_tensors()):
            code.writeline(f"in{i}_strides = in{i}.stride()")
            if not with_block_pointer:
                continue
            if ndim >= 2:  # where ndim is 1, we don't need to compute stride order
                code.writeline(f"in{i}_stride_order = stride_order(in{i}_strides)")
            else:
                code.writeline(f"in{i}_stride_order = (0,)")
        for i in range(schema.num_output_tensors()):
            code.writeline(f"out{i}_strides = out{i}.stride()")
            if not with_block_pointer:
                continue
            if ndim >= 2:
                code.writeline(f"out{i}_stride_order = stride_order(out{i}_strides)")
            else:
                code.writeline(f"out{i}_stride_order = (0,)")

        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            code.writeline(f"{self.jit_fn_name}[grid](")
            with code.indent():
                params = []
                # NOTE: WRAP
                for i in range(schema.num_inputs()):
                    if schema.is_tensor(i):
                        params.append(f"{self.input_name(i)}")
                    else:
                        params.append(self.input_name(i))
                for i in range(schema.num_output_tensors()):
                    params.append(f"{self.output_name(i)}")

                code.writeline(f"{_cs(params)},")

                if ndim > 0:
                    for i in range(schema.num_input_tensors()):
                        s = ", ".join(f"in{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for in{i}")
                        if not with_block_pointer:
                            continue
                        order = ", ".join(
                            f"in{i}_stride_order[{j}]" for j in range(ndim)
                        )
                        code.writeline(f"{order}, # stride order for in{i}")

                    for i in range(schema.num_output_tensors()):
                        s = ", ".join(f"out{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for out{i}")
                        if not with_block_pointer:
                            continue
                        order = ", ".join(
                            f"out{i}_stride_order[{j}]" for j in range(ndim)
                        )
                        code.writeline(f"{order}, # stride orderfor out{i}")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(ndim))
                    code.writeline(f"{shape_args}, # task indexing space")
                    code.writeline("num_tasks, # num tasks")
                    code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                    for i in range(ndim):
                        code.writeline(f"tile_size{i}=tile_sizes[{i}],")
                    code.writeline("one_tile_per_cta=one_tile_per_cta,")
                code.writeline("num_warps=num_warps,")
            code.writeline(")")

    def gen_kernel_launch_1d(
        self,
        code: IndentedBuffer,
    ):
        schema = self.fx
        ndim = self.ndim

        code.writeline("# kernel launch")
        for i in range(schema.num_input_tensors()):
            code.writeline(f"in{i}_strides = in{i}.stride()")
        for i in range(schema.num_output_tensors()):
            code.writeline(f"out{i}_strides = out{i}.stride()")

        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            code.writeline(f"{self.jit_fn_name}[grid](")
            with code.indent():
                params = []
                # NOTE: WRAP
                for i in range(schema.num_inputs()):
                    if schema.is_tensor(i):
                        params.append(f"{self.input_name(i)}")
                    else:
                        params.append(self.input_name(i))
                for i in range(schema.num_output_tensors()):
                    params.append(f"{self.output_name(i)}")

                code.writeline(f"{_cs(params)},")

                if ndim > 0:
                    for i in range(schema.num_input_tensors()):
                        s = ", ".join(f"in{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for in{i}")
                    for i in range(schema.num_output_tensors()):
                        s = ", ".join(f"out{i}_strides[{j}]" for j in range(ndim))
                        code.writeline(f"{s}, # stride for out{i}")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(ndim))
                    code.writeline(f"{shape_args}, # task indexing space")
                    code.writeline("num_tasks, # num tasks")
                    code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                    code.writeline("tile_size=tile_size,")
                    code.writeline("one_tile_per_cta=one_tile_per_cta,")
                code.writeline("num_warps=num_warps,")
            code.writeline(")")

    def gen_return(self, code: IndentedBuffer):
        return_exprs = _cs(f"out{i}" for i in range(self.fx.num_output_tensors()))
        code.writeline(f"return {return_exprs}")

    def codegen_nd_tile(self, code):
        self.gen_signature(code)

        with code.indent():
            self.gen_docstring(code)
            self.gen_same_shape_check(code)
            self.gen_task_partition(code)
            self.gen_kernel_launch(code)
            self.gen_return(code)
        code.newline()
        return code

    def codegen_1d_tile(self, code):
        self.gen_signature(code)

        with code.indent():
            self.gen_docstring(code)
            self.gen_same_shape_check(code)
            self.gen_task_partition_1d(code)
            self.gen_kernel_launch_1d(code)
            self.gen_return(code)
        code.newline()
        return code


class ModuleGenerator:
    def __init__(
        self,
        function_schema: FunctionSchema,
        scalar_fn: triton.JITFunction,
        ndim: int,
        jit_fn_name: str,
        wrapper_name: str,
        config: CodeGenConfig,
    ):
        self.config = config
        self.scalar_fn = scalar_fn
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )

    @staticmethod
    def _collect_jit_deps(scalar_fn):
        """Collect extra imports and local @triton.jit helper sources.

        Parses the source module where scalar_fn is defined using AST.
        Returns a tuple of:
          - extra_imports: dict of module_path -> set of names
          - local_sources: list of source strings for local @triton.jit
            functions (those NOT decorated with @pointwise_dynamic)
        """
        import ast
        import inspect

        py_fn = getattr(scalar_fn, "fn", scalar_fn)
        module_name = getattr(py_fn, "__module__", None)
        if not module_name:
            return {}, []
        try:
            mod = importlib.import_module(module_name)
            source_file = inspect.getfile(mod)
        except (ImportError, TypeError, OSError):
            return {}, []
        try:
            with open(source_file) as f:
                module_source = f.read()
            source_lines = module_source.splitlines(keepends=True)
            tree = ast.parse(module_source)
        except (OSError, SyntaxError):
            return {}, []

        # Collect non-standard import-from lines
        ALREADY_IMPORTED = {
            "math",
            "typing",
            "torch",
            "triton",
            "triton.language",
            "flag_gems.utils.shape_utils",
            "flag_gems.utils.tensor_wrapper",
            "flag_gems.utils.libentry",
            "flag_gems.utils",
            "flag_gems.runtime",
            "flag_gems.utils.pointwise_dynamic",
        }
        extra_imports = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in ALREADY_IMPORTED:
                    continue
                names = {alias.name for alias in node.names}
                extra_imports.setdefault(node.module, set()).update(names)

        # Collect local @triton.jit functions (without @pointwise_dynamic)
        def _has_decorator(func_node, name):
            for dec in func_node.decorator_list:
                src = "".join(source_lines[dec.lineno - 1 : dec.end_lineno])
                if name in src:
                    return True
            return False

        def _extract_source(func_node):
            start = func_node.lineno - 1
            if func_node.decorator_list:
                start = func_node.decorator_list[0].lineno - 1
            end = func_node.end_lineno
            return "".join(source_lines[start:end])

        local_sources = []
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not _has_decorator(node, "triton.jit") and not _has_decorator(
                node, "jit"
            ):
                continue
            if _has_decorator(node, "pointwise_dynamic"):
                continue
            local_sources.append(_extract_source(node))

        return extra_imports, local_sources

    def generate_imports(self, code: IndentedBuffer) -> IndentedBuffer:
        code.writeline("import math")
        code.writeline("from typing import Union")
        code.writeline("import torch")
        code.writeline("import triton")
        code.writeline("from triton import language as tl")
        code.newline()
        code.writeline("from flag_gems.utils.shape_utils import (")
        code.writeline("    heuristics_for_tile_size,")
        code.writeline("    heuristics_for_num_warps,")
        code.writeline("    stride_order,")
        code.writeline(")")
        code.writeline("from flag_gems.utils.tensor_wrapper import StridedBuffer")
        code.writeline("from flag_gems.utils.libentry import libentry")
        code.writeline("from flag_gems.utils import triton_lang_extension as tle")
        code.writeline("from flag_gems.runtime import torch_device_fn")

        # Generate extra imports and local JIT deps of the scalar function
        jit_dep_imports, local_jit_sources = self._collect_jit_deps(self.scalar_fn)
        for module_path, names in sorted(jit_dep_imports.items()):
            sorted_names = ", ".join(sorted(names))
            code.writeline(f"from {module_path} import {sorted_names}")

        code.newline()
        code.newline()

        # Emit local @triton.jit helper functions
        for source in local_jit_sources:
            for line in source.splitlines():
                code.writeline(line)
            code.newline()

        return code

    def codegen(self, code: IndentedBuffer):
        code = self.generate_imports(code)
        if self.config.prefer_1d_tile:
            code = self.wrapper_gen.codegen_1d_tile(code)
            code = self.kernel_gen.codegen_1d_tile(code)
        else:
            code = self.wrapper_gen.codegen_nd_tile(code)
            code = self.kernel_gen.codegen_nd_tile(code)
        return code


@dataclass
class KernelInfo:
    """Information about a generated kernel for C++ integration."""

    file_path: str
    kernel_name: str
    wrapper_name: str
    ndim: int


class ComplexMode(Enum):
    NONE = auto()
    ELEMENTWISE = auto()  # add/sub: view_as_real → same kernel → view_as_complex
    CROSS = auto()  # mul/div: split ar/ai/br/bi → cross_kernel


@dataclass
class ComplexStrategy:
    mode: ComplexMode = ComplexMode.NONE
    cross_kernel: object = None
    tensorize_scalars: bool = False
    fallback_target: object = None


_REAL_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


class PointwiseDynamicFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    def __init__(self, op_desc: FunctionSchema, scalar_fn: JITFunction, config=None):
        self.fx = op_desc

        assert isinstance(scalar_fn, JITFunction)
        self._scalar_fn = scalar_fn
        self._scalar_fn_cache_key = scalar_fn.cache_key
        self.pid = os.getpid()

        self.config: CodeGenConfig = config or get_codegen_config()

        # instantiated & cached overloads
        self.overloads: Mapping[str, Callable] = {}
        # cached kernel info for C++ integration
        self._kernel_info_cache: Mapping[str, KernelInfo] = {}

        # complex dispatch support
        self.complex_strategy = ComplexStrategy()
        self._operand_indices = self._infer_operand_indices()

    # -------------------- operand index inference --------------------

    def _infer_operand_indices(self):
        """Infer operand indices from schema._promotion_methods, done once at init."""
        indices = set()
        for pm in self.fx._promotion_methods:
            for idx in pm[:-1]:
                indices.add(idx)
        return frozenset(indices)

    # -------------------- register_complex --------------------

    def register_complex(
        self, mode, cross_kernel=None, tensorize_scalars=False, fallback_target=None
    ):
        """Register complex number support for this kernel.

        Args:
            mode: ComplexMode.ELEMENTWISE (add/sub) or ComplexMode.CROSS (mul/div).
            cross_kernel: A PointwiseDynamicFunction for cross-term ops (mul/div).
            tensorize_scalars: If True, scalar operands are converted to tensors
                before delegating to fallback_target.
            fallback_target: A PointwiseDynamicFunction (tensor-tensor version)
                to delegate to after tensorizing scalar operands.
        """
        self.complex_strategy = ComplexStrategy(
            mode=mode,
            cross_kernel=cross_kernel,
            tensorize_scalars=tensorize_scalars,
            fallback_target=fallback_target,
        )
        return self

    # -------------------- call entry --------------------

    def __call__(self, *args, **kwargs):
        if self._should_use_complex_path(args):
            return self._call_complex_dispatch(*args, **kwargs)
        return self._call_real_impl(*args, **kwargs)

    def _call_real_impl(self, *args, **kwargs):
        """Single entry point for real kernel invocation."""
        ndim, args, kwargs = self.prepare_args(*args, **kwargs)
        overload = self.instantiate(ndim)
        out = overload(*args, **kwargs)
        return self._unwrap(out)

    # -------------------- complex helpers --------------------

    @staticmethod
    def _is_complex_arg(a):
        return (isinstance(a, torch.Tensor) and a.is_complex()) or isinstance(
            a, complex
        )

    def _should_use_complex_path(self, args):
        if self.complex_strategy.mode == ComplexMode.NONE:
            return False
        return any(
            self._is_complex_arg(args[i])
            for i in self._operand_indices
            if i < len(args)
        )

    def _split_args(self, args):
        """Split args into operands and others by original position index."""
        operands = {}
        others = {}
        for i, a in enumerate(args):
            if i in self._operand_indices:
                operands[i] = a
            else:
                others[i] = a
        return operands, others

    def _merge_args(self, operands, others):
        """Rebuild args tuple from operands and others by original position index."""
        total = len(operands) + len(others)
        merged = [None] * total
        for i, v in operands.items():
            merged[i] = v
        for i, v in others.items():
            merged[i] = v
        return tuple(merged)

    def _classify_complex_inputs(self, operands):
        """Classify operands as 'all_complex', 'mixed', or 'real'."""
        complex_count = sum(1 for v in operands.values() if self._is_complex_arg(v))
        if complex_count == len(operands):
            return "all_complex"
        elif complex_count > 0:
            return "mixed"
        return "real"

    def _infer_device(self, operands):
        for v in operands.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return None

    def _infer_complex_dtype(self, operands):
        return torch.result_type(*operands.values())

    def _tensorize_scalar_operands(self, operands, dtype, device):
        """Convert scalar operands to tensors."""
        result = {}
        for i, v in operands.items():
            if not isinstance(v, torch.Tensor):
                if isinstance(v, complex):
                    result[i] = torch.tensor(v, dtype=dtype, device=device)
                elif isinstance(v, float):
                    result[i] = torch.tensor(v, dtype=torch.float32, device=device)
                elif isinstance(v, (int, bool)):
                    result[i] = torch.tensor(v, dtype=torch.int64, device=device)
                else:
                    result[i] = v
            else:
                result[i] = v
        return result

    def _to_complex_tensor(self, a, target_dtype, device):
        """Convert a scalar or real tensor to a complex tensor."""
        if isinstance(a, torch.Tensor):
            if a.is_complex():
                return a
            if a.is_floating_point():
                cdtype = _REAL_TO_COMPLEX.get(a.dtype, torch.complex64)
            else:
                a = a.to(torch.float32)
                cdtype = torch.complex64
            return torch.complex(a, torch.zeros_like(a)).to(cdtype)
        elif isinstance(a, complex):
            return torch.tensor(a, dtype=target_dtype, device=device)
        elif isinstance(a, (int, float)):
            return torch.tensor(complex(a, 0), dtype=target_dtype, device=device)
        return a

    # -------------------- complex dispatch --------------------

    def _call_complex_dispatch(self, *args, **kwargs):
        """Unified complex dispatch entry point."""
        strategy = self.complex_strategy
        operands, others = self._split_args(args)

        device = self._infer_device(operands)
        result_dtype = self._infer_complex_dtype(operands)

        # tensorize scalar operands and delegate to fallback_target
        if strategy.tensorize_scalars and strategy.fallback_target is not None:
            operands = self._tensorize_scalar_operands(operands, result_dtype, device)
            new_args = self._merge_args(operands, others)
            return strategy.fallback_target(*new_args, **kwargs)

        # convert all operands to complex tensors
        for i in list(operands.keys()):
            operands[i] = self._to_complex_tensor(operands[i], result_dtype, device)

        # broadcast complex tensor operands
        complex_tensors = [operands[i] for i in sorted(operands.keys())]
        complex_tensors = torch.broadcast_tensors(*complex_tensors)
        for idx, key in enumerate(sorted(operands.keys())):
            operands[key] = complex_tensors[idx]

        classification = self._classify_complex_inputs(operands)

        if strategy.mode == ComplexMode.CROSS and classification == "all_complex":
            return self._call_complex_cross(operands, result_dtype)
        elif classification in ("all_complex", "mixed"):
            return self._call_complex_elementwise(
                operands, others, result_dtype, kwargs
            )
        else:
            new_args = self._merge_args(operands, others)
            return self._call_real_impl(*new_args, **kwargs)

    def _call_complex_elementwise(self, operands, others, result_dtype, kwargs):
        """Elementwise: view_as_real -> call real kernel -> view_as_complex."""
        real_tensors = {i: torch.view_as_real(t) for i, t in operands.items()}

        # promote to common real dtype
        dtypes = [t.dtype for t in real_tensors.values()]
        common_dtype = dtypes[0]
        for d in dtypes[1:]:
            common_dtype = torch.promote_types(common_dtype, d)
        real_tensors = {i: t.to(common_dtype) for i, t in real_tensors.items()}

        new_args = self._merge_args(real_tensors, others)
        out_real = self._call_real_impl(*new_args, **kwargs)
        return torch.view_as_complex(out_real.contiguous()).to(result_dtype)

    def _call_complex_cross(self, operands, result_dtype):
        """Cross-term: split ar/ai/br/bi -> call cross_kernel -> stack -> view_as_complex."""
        sorted_keys = sorted(operands.keys())
        A, B = operands[sorted_keys[0]], operands[sorted_keys[1]]
        Ar = torch.view_as_real(A)
        Br = torch.view_as_real(B)
        ar, ai = Ar[..., 0], Ar[..., 1]
        br, bi = Br[..., 0], Br[..., 1]

        common_dtype = torch.promote_types(ar.dtype, br.dtype)
        ar, ai = ar.to(common_dtype), ai.to(common_dtype)
        br, bi = br.to(common_dtype), bi.to(common_dtype)

        real, imag = self.complex_strategy.cross_kernel(ar, ai, br, bi)

        out = torch.stack((real, imag), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    @staticmethod
    def use_fast_path(tensors):
        return all_the_same_shape(tensors) and (
            all_c_contiguous(tensors)
            or (
                all_the_same_stride(tensors)
                and torch.ops.aten.is_non_overlapping_and_dense(tensors[0])
            )
        )

    def prepare_args(self, *args, _skip_tensor_check=False, **kwargs):
        # output allocation(when needed)
        # task simplification & task-rank infernece & input-output reinterpretation
        schema = self.fx
        outputs_that_need_allocation: List[int] = []
        out_tensors = []
        for i in range(schema.num_output_tensors()):
            k = f"out{i}"
            if k in kwargs:
                out_tensors.append(kwargs[k])
            else:
                outputs_that_need_allocation.append(i)
        # input arguments must be passed by position
        if not _skip_tensor_check and schema._is_tensor is not None:
            if not check_tensor_attributes(args, (schema._is_tensor)):
                raise ValueError(
                    "Input arguments must be passed by position, and the corresponding dtype must be specified."
                )
        in_tensors = [item for i, item in enumerate(args) if schema.is_tensor(i)]

        # output dtype promotions
        outputs_dtypes_for_allocation = []
        for i in outputs_that_need_allocation:
            *arg_indices, method = schema._promotion_methods[i]
            promote_args = (args[j] for j in arg_indices)
            _, dtype = type_promotion(*promote_args, type_promotion=method)
            outputs_dtypes_for_allocation.append(dtype)

        tensors = out_tensors + in_tensors
        INT32_MAX = torch.iinfo(torch.int32).max
        if tensors[0].numel() > INT32_MAX:
            self.config.prefer_block_pointer = False
        if self.use_fast_path(tensors):  # dimension collapse & use physical ordering
            allocated_outputs = [
                torch.empty_like(tensors[0], dtype=dtype)
                for dtype in outputs_dtypes_for_allocation
            ]
            task_shape = (tensors[0].numel(),)
            strides = (1,)
            ndim = 1
            args = tuple(
                (
                    StridedBuffer(item, task_shape, strides)
                    if schema.is_tensor(i)
                    else item
                )
                for i, item in enumerate(args)
            )
            kwargs = {
                k: StridedBuffer(item, task_shape, strides)
                for k, item in kwargs.items()
            }
            for seq_id, output_id in enumerate(outputs_that_need_allocation):
                kwargs[f"out{output_id}"] = StridedBuffer(
                    allocated_outputs[seq_id], task_shape, strides
                )
        else:
            # a simple strategy: all the undefined tensors will follow the first
            # tensor that is not broadcated, no attempts to simplify task, no reordering,
            # no dimenion collapsing
            shapes = tuple(item.shape for item in in_tensors)

            task_shape = broadcast_shapes(shapes)

            if out_tensors:
                for index, item in enumerate(out_tensors):
                    if list(item.shape) != list(task_shape):
                        raise RuntimeError(
                            f"out tensor at index {index} shape is invalid, should be {task_shape} but is {item.shape}!"
                        )
                    # output arguments must not have internal overlapping for pointwise operation
                    if has_internal_overlapping(item) == MemOverlap.Yes:
                        raise RuntimeError(
                            "Pointwise Input arguments should not have internal overlapping."
                        )

            ndim = len(task_shape)
            for item in tensors:
                if item.shape == task_shape:
                    allocated_outputs = [
                        torch.empty_like(item, dtype=dtype)
                        for dtype in outputs_dtypes_for_allocation
                    ]
                    break
            else:  # nobreak
                device = tensors[0].device
                allocated_outputs = [
                    torch.empty(task_shape, dtype=dtype, device=device)
                    for dtype in outputs_dtypes_for_allocation
                ]
            args = tuple(
                (
                    StridedBuffer(
                        item,
                        task_shape,
                        broadcasted_stride(item.shape, item.stride(), task_shape),
                    )
                    if schema.is_tensor(i)
                    else item
                )
                for i, item in enumerate(args)
            )
            kwargs = {
                k: StridedBuffer(
                    item,
                    task_shape,
                    broadcasted_stride(item.shape, item.stride(), task_shape),
                )
                for k, item in kwargs.items()
            }
            for seq_id, output_id in enumerate(outputs_that_need_allocation):
                item = allocated_outputs[seq_id]
                kwargs[f"out{output_id}"] = StridedBuffer(
                    item,
                    task_shape,
                    broadcasted_stride(item.shape, item.stride(), task_shape),
                )
        return (ndim, args, kwargs)

    def _unwrap(self, tensors):
        # unwrap StridedBuffer to get Tensor
        if self.fx.num_output_tensors() == 1:
            item = tensors
            return item.unwrap()
        return tuple(item.unwrap() for item in tensors)

    def _compute_kernel_names(self, ndim: int) -> Tuple[str, str, str]:
        """Compute kernel name, wrapper name, and file path for a given ndim.

        This is the single source of truth for naming, used by both instantiate()
        and get_kernel_info() to ensure consistency.

        Returns:
            Tuple of (kernel_name, wrapper_name, file_path)
        """
        scalar_fn_name = self._scalar_fn.__name__
        kernel_name = f"{scalar_fn_name}_kernel_rank_{ndim}"
        wrapper_name = f"{scalar_fn_name}_wrapper_rank_{ndim}"

        file_name = (
            f"pointwise_dynamic_{self._scalar_fn_cache_key}_{kernel_name}_"
            f"{'1d_tile_' if self.config.prefer_1d_tile else ''}"
            f"{'bptr' if (not self.config.prefer_1d_tile and self.config.prefer_block_pointer) else ''}"
            ".py"
        )
        file_path = str(code_cache_dir() / file_name)

        return kernel_name, wrapper_name, file_path

    def instantiate(self, ndim):
        # NOTE: manually instantiated overload does not have `prepare_args` as
        # preprocessing, so you have to manually allocate output and make sure that
        # the inputs & ouputs actually fits the manually instantiated overload
        key = f"{ndim}_{self.config.prefer_block_pointer}"
        if key in self.overloads:
            return self.overloads[key]

        code = IndentedBuffer()

        # Use helper to compute names (single source of truth)
        kernel_name, wrapper_name, file_path = self._compute_kernel_names(ndim)

        module_gen = ModuleGenerator(
            self.fx,
            self._scalar_fn,
            ndim,
            kernel_name,
            wrapper_name,
            self.config,
        )
        module_gen.codegen(code)

        # NOTE: [why write the generated code to a file]
        # triton uses inpsect to get the source of the jitted function, which requires
        # that the source code can be found by inspect
        # We write it into a file, since inspect cannot find the source of functions dynamically
        # created via exec string. We can help inspect to find the source by hacking linecache
        # library, but we find generating a module simpler, since we can generating 2 functions
        # the kernel and the wrapper, and the wrapper calls the kernel.
        write_atomic(file_path, code.getvalue())

        # load
        spec = importlib.util.spec_from_file_location(
            f"_gen_module_{self._scalar_fn_cache_key}_rank_{ndim}",
            file_path,
        )
        m = importlib.util.module_from_spec(spec)
        # do not expose it to sys.modules
        # sys.modules["_add_module"] = m

        # NOTE: [why not import the scalar function]
        # we do not re-import the scalar function, although the generated kernel **calls** it
        # Since a function's __name__ may be changed, from the module where it is defined import its
        # __name__ is not same; Also the same may be rebind to something else, importing via name
        # cannot guarantee that scalar function is imported.
        # So we copy the scalar function and its __globals__ to the generated module to do this
        # https://stackoverflow.com/questions/11170949/how-to-make-a-copy-of-a-python-module-at-runtime
        spec.loader.exec_module(m)
        m.__dict__.update(self._scalar_fn.__globals__)
        m.__dict__[self._scalar_fn.__name__] = self._scalar_fn

        overload = getattr(m, wrapper_name)
        self.overloads[key] = overload

        # Cache kernel info for C++ integration
        self._kernel_info_cache[key] = KernelInfo(
            file_path=file_path,
            kernel_name=kernel_name,
            wrapper_name=wrapper_name,
            ndim=ndim,
        )

        return overload

    def get_kernel_info(self, ndim: int) -> KernelInfo:
        """Get kernel information for a given ndim.

        This method is useful for C++ integration to get the file path and
        kernel name without duplicating the naming logic.

        If the kernel hasn't been instantiated yet, this will instantiate it first.

        Args:
            ndim: The rank of the task space

        Returns:
            KernelInfo with file_path, kernel_name, wrapper_name, and ndim
        """
        key = f"{ndim}_{self.config.prefer_block_pointer}"

        # Ensure the kernel is instantiated
        if key not in self._kernel_info_cache:
            self.instantiate(ndim)

        return self._kernel_info_cache[key]


def pointwise_dynamic(
    f: Optional[JITFunction] = None,
    *,
    num_inputs: Optional[int] = None,
    is_tensor: Optional[List[bool]] = None,
    dtypes: Optional[List[Optional[type]]] = None,
    num_outputs: Optional[int] = None,
    promotion_methods: Optional[Tuple[int, ...]] = None,
    config: Optional[CodeGenConfig] = None,
):
    def decorator(fn):
        nonlocal num_inputs
        if (num_inputs is None) and (is_tensor is None) and (dtypes is None):
            num_inputs = len(fn.arg_names)
        op_desc = FunctionSchema(
            num_inputs=num_inputs,
            is_tensor=is_tensor,
            dtypes=dtypes,
            num_outputs=num_outputs,
            promotion_methods=promotion_methods,
        )
        return PointwiseDynamicFunction(op_desc, fn, config)

    if f is not None:
        return decorator(f)
    return decorator
