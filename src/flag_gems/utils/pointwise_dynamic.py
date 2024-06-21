import importlib
import os
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch
import triton
from triton import language as tl
from triton.runtime.jit import JITFunction

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace
from flag_gems.utils.inliner import inline_function
from flag_gems.utils.shape_utils import broadcast_shapes


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


class OPDesc:
    _num_inputs: int
    _is_tensor: List[bool]
    _dtypes: List[Optional[type]]

    _num_input_tensors: int
    _num_non_tensor_inputs: int

    _num_outputs: int
    _output_dtypes: List[torch.dtype]

    def __init__(
        self,
        *,
        num_inputs: Optional[int] = None,
        is_tensor: Optional[List[bool]] = None,
        dtypes: Optional[List[Optional[type]]] = None,
        num_outputs: Optional[int] = None,
        output_dtypes: Optional[List[torch.dtype]] = None,
    ):
        if is_tensor is not None:
            _check_typed_list(is_tensor, bool)
        if dtypes is not None:
            _check_typed_list(dtypes, (type, type(None)))

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
                "Cannot make OPDesc when none of (num_inputs, is_tensor, dtypes) is specified."
            )

        if output_dtypes is not None:
            _check_typed_list(output_dtypes, torch.dtype)

        if num_outputs is not None:
            self._num_outputs = num_outputs
            if output_dtypes is not None:
                _check_sized_list(output_dtypes, num_outputs)
                self._output_dtypes = output_dtypes
            else:
                self._output_dtypes = [None] * num_outputs  # infer from the 1st input
        elif output_dtypes is not None:
            self._num_outputs = len(output_dtypes)
            self._output_dtypes = output_dtypes
        else:
            self._num_outputs = 1
            self._output_dtypes = [None]

        assert self._num_inputs >= 1
        assert self._num_outputs >= 1

        self._num_input_tensors = sum(self._is_tensor)
        self._num_non_tensor_inputs = self._num_inputs - self._num_input_tensors

    def num_inputs(self):
        # num of arguments, outputs not included
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def is_tensor(self, arg_id: int) -> bool:
        return self._is_tensor[arg_id]

    def input_type(self, arg_id) -> Optional[type]:
        return self._dtypes[arg_id]

    def output_dtype(self, output_id) -> torch.dtype:
        return self._output_dtypes[output_id]

    def num_input_tensors(self) -> int:
        return self._num_input_tensors

    def num_output_tensors(self) -> int:
        return self._num_outputs

    def num_non_tensor_args(self) -> int:
        return self._num_non_tensor_inputs

    def signature(self, outputs_in_arg: bool = False):
        input_types = []
        for is_tensor, dtype in zip(self._is_tensor, self._dtypes):
            if is_tensor:
                input_types.append("Tensor")
            else:
                if dtype is None:
                    input_types.append("scalar")
                else:
                    input_types.append(_type_name(dtype))

        output_types = []
        for dtype in self._output_dtypes:
            if dtype is None:
                output_types.append("Tensor")
            else:
                output_types.append(f"Tensor[{_type_name(dtype)}]")
        if outputs_in_arg:
            input_types.extend(output_types)
        sig = f'Pointwise: ({", ".join(input_types)}) -> ({", ".join(output_types)})'
        return sig

    def __str__(self) -> str:
        return self.signature(outputs_in_arg=False)


# --------------------------- pointwise wrapper genration -----------------------------------
def parameter_for_wrapper(op_desc: OPDesc, include_outputs: bool = False) -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}: torch.Tensor")
            input_tensor_index += 1
        else:
            if op_desc.input_type(i) is not None:
                parameters.append(
                    f"val{non_tensor_index}: {_type_name(op_desc.input_type(i))}"
                )
            else:
                parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    if include_outputs:
        output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            parameters.append(f"out{output_tensor_index}: torch.Tensor")
            output_tensor_index += 1

    return ", ".join(parameters)


def parameter_ref_for_wrapper(op_desc: OPDesc, include_outputs: bool = False) -> str:
    """Generate parameter reference for wrapper function.
    Example: in0, val0, out0
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}")
            input_tensor_index += 1
        else:
            parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    if include_outputs:
        output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            parameters.append(f"out{output_tensor_index}")
            output_tensor_index += 1

    return ", ".join(parameters)


def output_ref_for_wrapper(op_desc: OPDesc) -> str:
    """Generate output variable refernece for wrapper function.
    Example: out0, out1
    """
    parameters: List[str] = [f"out{i}" for i in range(op_desc.num_outputs())]
    return ", ".join(parameters)


def docstring_for_functional_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {str(op_desc)}"""'
    return doc


def docstring_for_destination_passing_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {op_desc.signature(outputs_in_arg=True)}"""'
    return doc


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline("from flag_gems.utils.shape_utils import (")
    code.writeline("    broadcast_shapes,")
    code.writeline("    broadcasted_stride,")
    code.writeline("    c_contiguous_stride,")
    code.writeline("    volume,")
    code.writeline("    Stride,")
    code.writeline(")")
    code.writeline("from flag_gems.utils.libentry import libentry")
    code.newline()
    code.newline()
    return code


def generate_functional_pointwise_wrapper(
    op_desc: OPDesc,
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=False)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_functional_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        shapes_str = ", ".join(
            f"in{i}.shape" for i in range(op_desc.num_input_tensors())
        )
        code.writeline(f"shape = broadcast_shapes([{shapes_str}])")

        # output allocation
        num_output_tensor_index = 0
        for i in range(op_desc.num_outputs()):
            if op_desc.output_dtype(i) is None:
                code.writeline(
                    (
                        f"out{num_output_tensor_index} = "
                        f"torch.empty(shape, dtype=in0.dtype, device=in0.device)"
                    )
                )
            else:
                code.writeline(
                    (
                        f"out{num_output_tensor_index} = "
                        f"torch.empty(shape, dtype={_type_name(op_desc.output_dtype(i))}, "
                        f"device=in0.device)"
                    )
                )
            num_output_tensor_index += 1

        # call destination_passing_func
        output_names: str = output_ref_for_wrapper(op_desc)
        call_str = (
            f"{output_names} = {destination_passing_func_name}"
            f"({parameter_ref_for_wrapper(op_desc, include_outputs=True)})"
        )
        code.writeline(call_str)

        return_str = f"return {output_names}"
        code.writeline(return_str)
        code.newline()
        code.newline()
    return code


def generate_destination_passing_pointwise_wrapper(
    op_desc: OPDesc,
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=True)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    # task partitioning, 1d task indexing
    tile_size = 512
    num_warps = 4
    if rank == 0:  # special case with rank-0, only 1 element to compute
        tile_size = 32
        num_warps = 1

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_destination_passing_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        shapes_str = ", ".join(
            f"in{i}.shape" for i in range(op_desc.num_input_tensors())
        )
        code.writeline(f"shape = broadcast_shapes([{shapes_str}])")
        code.writeline("num_tasks = volume(shape)")
        code.newline()

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            for i in range(op_desc.num_input_tensors()):
                code.writeline(
                    f"in{i}_strides = broadcasted_stride(in{i}.shape, in{i}.stride(), shape)"
                )

            for i in range(op_desc.num_output_tensors()):
                code.writeline(f"out{i}_strides = out{i}.stride()")

            code.newline()

        # grid
        code.writeline("# kernel launch")
        grid_stmt: str = f"grid = triton.cdiv(num_tasks, {tile_size}), 1, 1"
        code.writeline(grid_stmt)

        # launch kernel
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline(
                f"{parameter_ref_for_wrapper(op_desc, include_outputs=True)},"
            )

            if rank > 0:
                for i in range(op_desc.num_input_tensors()):
                    s = ", ".join(f"in{i}_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for in{i}")

                for i in range(op_desc.num_output_tensors()):
                    s = ", ".join(f"out{i}_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for out{i}")

                shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                if rank > 0:
                    code.writeline(f"{shape_args}, # task indexing space")
                code.writeline("num_tasks, # num tasks")

            code.writeline(f"tile_size={tile_size},")
            code.writeline(f"num_warps={num_warps},")
        code.writeline(")")

        # return
        code.writeline(f"return {output_ref_for_wrapper(op_desc)}")
        code.newline()
        code.newline()
    return code


def generate_pointwise_kernel(
    op_desc: OPDesc,
    scalar_fn: JITFunction,
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    if op_desc.num_non_tensor_args() > 0:
        non_specialize_arg_names = [
            f"val{i}" for i in range(op_desc.num_non_tensor_args())
        ]
        code.writeline(f"@triton.jit(do_not_specialize={non_specialize_arg_names})")
    else:
        code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")

    function_ns = NameSpace()
    # signature
    with code.indent():
        input_tensor_index = 0
        non_tensor_index = 0
        output_tensor_index = 0
        # inputs ptrs & non tensor inputs
        for i in range(op_desc.num_inputs()):
            if op_desc.is_tensor(i):
                code.writeline(
                    f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                function_ns.create_name(f"in{input_tensor_index}_ptr")
                input_tensor_index += 1
            else:
                if op_desc.input_type(i) is not None:
                    code.writeline(
                        f"val{non_tensor_index}: {_type_name(op_desc.input_type(i))},"
                    )
                else:
                    code.writeline(f"val{non_tensor_index},")
                function_ns.create_name(f"val{non_tensor_index}")
                non_tensor_index += 1

        # output ptrs
        for i in range(op_desc.num_outputs()):
            code.writeline(
                f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
            )
            function_ns.create_name(f"out{output_tensor_index}_ptr")
            output_tensor_index += 1

        if rank > 0:
            # strides for inputs
            for i in range(op_desc.num_input_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"in{i}_stride{j}")
                stride_args = ", ".join(f"in{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for in{i}")

            # strides for outputs
            for i in range(op_desc.num_output_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"out{i}_stride{j}")
                stride_args = ", ".join(f"out{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for out{i}")

            # task space, used to reconstruct multi index
            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            for i in range(rank):
                function_ns.create_name(f"s{i}")
            code.writeline(f"{task_space_args}, # task_space")

            # number of tasks, used to compute mask
            code.writeline("num_tasks: int,")
            function_ns.create_name("num_tasks")

        code.writeline("tile_size: tl.constexpr,")
        function_ns.create_name("tile_size")
    code.writeline("):")

    # function body
    with code.indent():
        # get pid
        code.writeline("# task id & masking")
        pid_stmt = "pid = tl.program_id(0)"
        code.writeline(pid_stmt)
        function_ns.create_name("pid")

        # get tid (a.k.a task id)
        tid_stmt = "tid = pid * tile_size + tl.arange(0, tile_size)"
        code.writeline(tid_stmt)
        function_ns.create_name("tid")

        if rank > 0:
            # only apply masking when rank > 0
            # since we only load a value instead of a block of values when the rank is 0
            mask_stmt: str = "mask = tid < num_tasks"
            code.writeline(mask_stmt)
            function_ns.create_name("mask")
            code.newline()

        # reconstruct multi index
        if rank > 0:
            code.writeline("# multi index recontruction")
            for i in reversed(range(rank)):
                code.writeline(f"i{i} = tid % s{i}")
                function_ns.create_name(f"{i}")
                if i > 0:
                    code.writeline(f"tid //= s{i}")
            code.newline()

        # loads
        code.writeline("# loads")
        for i in range(op_desc.num_input_tensors()):
            if rank > 0:
                ptrs_expr: str = " + ".join(
                    f"i{j} * in{i}_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
            else:
                ptrs_expr: str = f"in{i}_ptr"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr})"
            function_ns.create_name(f"in{i}")  # add to the namespace
            code.writeline(load_stmt)
        code.newline()

        # compute
        code.writeline("# compute")

        inputs_to_scalar_fn = []
        input_tensor_index = 0
        non_tensor_index = 0
        for i in range(op_desc.num_inputs()):
            if op_desc.is_tensor(i):
                inputs_to_scalar_fn.append(f"in{input_tensor_index}")
                input_tensor_index += 1
            else:
                inputs_to_scalar_fn.append(f"val{non_tensor_index}")
                non_tensor_index += 1

        outputs_to_scalar_fn = [f"out{i}" for i in range(op_desc.num_outputs())]

        compute_body = inline_function(
            scalar_fn,
            inputs_to_scalar_fn,
            outputs_to_scalar_fn,
            function_ns,
        )
        for line in compute_body.strip().splitlines():
            code.writeline(line)
        code.newline()

        # stores
        code.writeline("# stores")
        for i in range(op_desc.num_output_tensors()):
            if rank > 0:
                ptrs_expr: str = " + ".join(
                    f"i{j} * out{i}_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"out{i}_ptr + {ptrs_expr}"
                store_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
            else:
                ptrs_expr: str = f"out{i}_ptr"
                store_stmt: str = f"tl.store({ptrs_expr}, out{i})"
            code.writeline(store_stmt)
        code.newline()
    return code


def generate_code(
    op_desc: OPDesc,
    scalar_fn: JITFunction,
    inputs: Tuple[Any],
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    assert (
        len(inputs) == op_desc.num_inputs()
    ), "the number of inputs does not match {str(op_desc)}"
    input_tensor_ids = [i for i in range(op_desc.num_inputs()) if op_desc.is_tensor(i)]
    tensor_shapes = [inputs[i].shape for i in input_tensor_ids]
    shape = broadcast_shapes(tensor_shapes)
    rank = len(shape)

    # the only runtime determined factor is the rank of the task space
    code = generate_imports(code)
    code = generate_functional_pointwise_wrapper(
        op_desc, wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_pointwise_wrapper(
        op_desc, rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_pointwise_kernel(op_desc, scalar_fn, rank, kernel_name, code)
    return code


class PointwiseDynamicFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    def __init__(self, op_desc: OPDesc, scalar_fn: JITFunction):
        self._op_desc = op_desc

        assert isinstance(scalar_fn, JITFunction)
        self._scalar_fn = scalar_fn
        self._scalar_fn_cache_key = scalar_fn.cache_key
        self.pid = os.getpid()

        # instantiated & cached overloads
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args):
        # It does not accept kwargs
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            # generate file & import it
            code = IndentedBuffer()
            code = generate_code(
                self._op_desc,
                self._scalar_fn,
                args,
                "_wrapper",
                "_wrapper_out",
                "_jit_function",
                code,
            )

            file_name = f"pointwise_dynamic_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}.py"
            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}",
                f.name,
            )
            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(*args)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


def pointwise_dynamic(
    f: Optional[JITFunction] = None,
    *,
    num_inputs: Optional[int] = None,
    is_tensor: Optional[List[bool]] = None,
    dtypes: Optional[List[Optional[type]]] = None,
    num_outputs: Optional[int] = None,
    output_dtypes: Optional[List[type]] = None,
):
    def decorator(fn):
        nonlocal num_inputs
        if (num_inputs is None) and (is_tensor is None) and (dtypes is None):
            num_inputs = len(fn.arg_names)
        op_desc = OPDesc(
            num_inputs=num_inputs,
            is_tensor=is_tensor,
            dtypes=dtypes,
            num_outputs=num_outputs,
            output_dtypes=output_dtypes,
        )
        return PointwiseDynamicFunction(op_desc, fn)

    if f is not None:
        return decorator(f)
    return decorator


if __name__ == "__main__":

    @pointwise_dynamic(is_tensor=[True, False, True], dtypes=[None, float, None])
    @triton.jit
    def saxpy(x, alpha, y):
        return x * alpha + y

    x = torch.randn((3, 4), device="cuda")
    y = torch.randn((4,), device="cuda")
    out1 = saxpy(x, 2.0, y)
    out2 = x * 2.0 + y
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic(is_tensor=[True, False, True])
    @triton.jit
    def saxpy(x, alpha, y):
        return x * alpha + y

    out1 = saxpy(x, 2.0, y)
    out2 = x * 2.0 + y
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic(output_dtypes=[torch.bool])
    @triton.jit
    def ge(x, y):
        return x > y

    out1 = ge(x, y)
    out2 = x > y
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic()
    @triton.jit
    def ordinary(x, y):
        return tl.sin(x) + tl.cos(y)

    out1 = ordinary(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic
    @triton.jit
    def ordinary2(x, y):
        return tl.sin(x) + tl.cos(y)

    out1 = ordinary2(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic
    @triton.jit
    def ordinary2(x, y):
        return tl.sin(x) + tl.cos(y)

    x = torch.tensor(1.0, device="cuda")
    y = torch.tensor(2.0, device="cuda")
    out1 = ordinary2(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()

    @pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
    @triton.jit
    def eq(x, y):
        return x.to(tl.float32) == y.to(
            tl.float32
        )  # ensures that y is not used for specialization

    x = torch.arange(10, device="cuda")
    y = 1
    # by default value 1 is treated as constexpr even thought it is not marked as constexpr
    # do_not_specialize avoids this
    out1 = eq(x, y)
    out2 = x == y
    print(out1)
    print(out2)
    torch.testing.assert_close(out1, out2)
    print()
