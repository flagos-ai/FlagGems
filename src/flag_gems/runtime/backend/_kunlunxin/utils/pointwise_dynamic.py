import importlib
import os
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import triton
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig, get_codegen_config
from triton.runtime.jit import JITFunction

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
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
        code.writeline(f'"""Quoted source of {self.fn_name}:')
        code.writemultiline(self.fn.src)
        code.writeline('"""')
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
                    stride_args = _cs(
                        f"in{i}_stride{j}: tl.constexpr" for j in range(ndim)
                    )
                    code.writeline(f"{stride_args}, # strides for in{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"in{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(f"{stride_order_args}, # stride order for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(
                        f"out{i}_stride{j}: tl.constexpr" for j in range(ndim)
                    )
                    code.writeline(f"{stride_args}, # strides for out{i}")
                    if with_block_pointer:
                        stride_order_args = _cs(
                            f"out{i}_stride_order{j}: tl.constexpr" for j in range(ndim)
                        )
                        code.writeline(
                            f"{stride_order_args}, # stride order for out{i}"
                        )

                # task space, used to reconstruct multi index
                task_space_args = _cs(f"s{i}: tl.constexpr" for i in range(ndim))
                code.writeline(f"{task_space_args}, # task_space")

                # number of tasks, used to compute mask
                code.writeline("num_tasks: tl.constexpr,")

            # tile size & tiles_per_cta, gsl style
            if ndim > 0:
                code.writeline("tiles_per_cta: tl.constexpr,")
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
                    stride_args = _cs(
                        f"in{i}_stride{j}: tl.constexpr" for j in range(ndim)
                    )
                    code.writeline(f"{stride_args}, # strides for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(
                        f"out{i}_stride{j}: tl.constexpr" for j in range(ndim)
                    )
                    code.writeline(f"{stride_args}, # strides for out{i}")

                # task space, used to reconstruct multi index
                task_space_args = _cs(f"s{i}: tl.constexpr" for i in range(ndim))
                code.writeline(f"{task_space_args}, # task_space")

                # number of tasks, used to compute mask
                code.writeline("num_tasks: tl.constexpr,")

            # tile size & tiles_per_cta, gsl style
            if ndim > 0:
                code.writeline("tiles_per_cta: tl.constexpr,")
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
        # code.writeline("print(\"in0\", in0)")

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
        # code.writeline("print(\"in0\", in0)")

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
            code.writeline(
                f"tile_sizes = heuristics_for_tile_size({max_tile_size}, *shape)"
            )
            code.writeline("tile_size = math.prod(tile_sizes)")
            code.writeline(
                "num_tiles = math.prod(triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes))"
            )
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
            code.writeline(
                f"tile_sizes = heuristics_for_tile_size({max_tile_size}, num_tasks)"
            )
            code.writeline("tile_size = tile_sizes[0]")
            code.writeline("num_tiles = triton.cdiv(num_tasks, tile_size)")
            # max_grid_size0 = self.config.max_grid_size[0]
            # code.writeline(f"num_ctas = min({max_grid_size0}, num_tiles)")
            determine_num_ctas_and_tiles = [
                "if sum(out0.shape) <= 2048*64:",
                "   num_ctas = 1 # XPU BLOCK_NUM",
                "   num_tiles = 1 # XPU BLOCK_NUM",
                "else:",
                "   num_ctas = 12 # XPU BLOCK_NUM",
                "   num_tiles = 12 # XPU BLOCK_NUM",
            ]
            if self.config.kunlunAutoGrid is True:
                code.writelines(determine_num_ctas_and_tiles)
            else:
                code.writeline("num_ctas = 12 # XPU BLOCK_NUM")
                code.writeline("num_tiles = 12 # XPU BLOCK_NUM")
            code.writeline(
                "tile_size = triton.next_power_of_2(triton.cdiv(num_tasks, num_tiles)) # XPU BLOCK_NUM"
            )

            # code.writeline("element_size = get_element_size(in0.dtype)")
            # code.writeline(
            #     "tile_size = min("
            #     "triton.next_power_of_2(triton.cdiv(num_tasks, 12)), "
            #     "triton.cdiv(2048 * 64, element_size)"
            #     ")"
            # )
            # code.writeline("num_tiles = triton.cdiv(num_tasks, tile_size)")
            # code.writeline("num_ctas = num_tiles")

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

        code.writeline("with torch_device_fn._DeviceGuard(in0.device.index):")
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
                if self.config.is_scatter_slice:
                    code.writeline("buffer_size_limit=512,")
                    code.writeline("isCloseOffsetAnalysis=True,")
                elif self.config.is_cat:
                    code.writeline("buffer_size_limit=512,")
                elif self.config.buffer_size_limit:
                    code.writeline(
                        f"buffer_size_limit={self.config.buffer_size_limit},"
                    )
                else:
                    code.writeline("buffer_size_limit=2048,")
                if self.config.isCloseVectorization:
                    code.writeline("isCloseVectorization=True,")
                if self.config.isCloseInterleave:
                    code.writeline("isCloseInterleave=True,")
                if self.config.isCloseDtypeConvert:
                    code.writeline("isCloseDtypeConvert=True,")
                if not self.config.isCloseMemoryAsync:
                    code.writeline("isCloseMemoryAsync=False,")
                if os.getenv("XPU_cmp_nan") == "1":
                    code.writeline("isOpenCmpNan=True,")
                if self.config.unroll_num:
                    code.writeline(f"unroll_num={self.config.unroll_num},")
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

        code.writeline("with torch_device_fn._DeviceGuard(in0.device.index):")
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
                if self.config.is_scatter_slice:
                    code.writeline("buffer_size_limit=512,")
                    code.writeline("isCloseOffsetAnalysis=True,")
                elif self.config.is_cat:
                    code.writeline("buffer_size_limit=512,")
                elif self.config.buffer_size_limit:
                    code.writeline(
                        f"buffer_size_limit={self.config.buffer_size_limit},"
                    )
                else:
                    code.writeline("buffer_size_limit=2048,")
                if self.config.isCloseVectorization:
                    code.writeline("isCloseVectorization=True,")
                if self.config.isCloseInterleave:
                    code.writeline("isCloseInterleave=True,")
                if self.config.isCloseDtypeConvert:
                    code.writeline("isCloseDtypeConvert=True,")
                if not self.config.isCloseMemoryAsync:
                    code.writeline("isCloseMemoryAsync=False,")
                if os.getenv("XPU_cmp_nan") == "1":
                    code.writeline("isOpenCmpNan=True,")
                if self.config.unroll_num:
                    code.writeline(f"unroll_num={self.config.unroll_num},")
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
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )

    @staticmethod
    def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
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
        # code.writeline("from _kunlunxin.utils.block_size_utils import get_element_size")
        code.newline()
        code.newline()
        return code

    def codegen(self, code: IndentedBuffer):
        # the only runtime determined factor is the rank of the task space
        code = self.generate_imports(code)
        if self.config.prefer_1d_tile:
            code = self.wrapper_gen.codegen_1d_tile(code)
            code = self.kernel_gen.codegen_1d_tile(code)
        else:
            code = self.wrapper_gen.codegen_nd_tile(code)
            code = self.kernel_gen.codegen_nd_tile(code)
        return code


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
        self.overloads: Mapping[int, Callable] = {}

    def __call__(self, *args, **kwargs):
        # inputs must be passed by position, outputs must be passed by keyword
        ndim, args, kwargs = self.prepare_args(*args, **kwargs)
        overload = self.instantiate(ndim)
        out = overload(*args, **kwargs)
        # NOTE: overload keeps the type of outputs:
        # if a pre-defiend output is a Tensor or StridedBuffer, the corresponding
        # output is also a Tensor StridedBuffer, respectively
        # since prepare_args Wraps all the arguments, the outputs are all StridedBuffer
        # but if manually instantiated overload is directly called, take care of
        # that manually
        return self._unwrap(out)

    @staticmethod
    def use_fast_path(tensors):
        return all_the_same_shape(tensors) and (
            all_c_contiguous(tensors)
            or (
                all_the_same_stride(tensors)
                and torch.ops.aten.is_non_overlapping_and_dense(tensors[0])
            )
        )

    def prepare_args(self, *args, **kwargs):
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
        if schema._is_tensor is not None:
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

    def instantiate(self, ndim):
        # NOTE: manually instantiated overload does not have `prepare_args` as
        # preprocessing, so you have to manually allocate output and make sure that
        # the inputs & ouputs actually fits the manually instantiated overload
        if ndim in self.overloads:
            return self.overloads[ndim]

        code = IndentedBuffer()

        scalar_fn_name = self._scalar_fn.__name__
        kernel_name = f"{scalar_fn_name}_kernel_rank_{ndim}"
        wrapper_name = f"{scalar_fn_name}_wrapper_rank_{ndim}"
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
        file_name = (
            f"pointwise_dynamic_{self._scalar_fn_cache_key}_{kernel_name}_"
            f"{'1d_tile_' if self.config.prefer_1d_tile else ''}"
            f"{'bptr_' if (not self.config.prefer_1d_tile and self.config.prefer_block_pointer) else ''}"
            f"pid_{self.pid}.py"
        )
        with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
            f.write(code.getvalue())

        # load
        spec = importlib.util.spec_from_file_location(
            f"_gen_module_{self._scalar_fn_cache_key}_rank_{ndim}_pid_{self.pid}",
            f.name,
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
        self.overloads[ndim] = overload
        return overload


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
