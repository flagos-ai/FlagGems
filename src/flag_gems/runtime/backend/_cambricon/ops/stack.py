import importlib
import logging
import math
import os
import textwrap
from typing import Callable, List, Mapping, Tuple, Union

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

from ..utils import TOTAL_CORE_NUM
from .vstack import vstack

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def get_dtype_size(dtype):
    try:
        return torch.finfo(dtype).bits // 8
    except TypeError:
        try:
            return torch.iinfo(dtype).bits // 8
        except TypeError:
            if dtype == torch.bool:
                return 1
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")


class StackKernelCode(IndentedBuffer):
    """
    Stack kernel template.
    """

    overloads: Mapping[str, Callable] = {}

    def __init__(self):
        self.pid = os.getpid()
        self.cache = self.overloads
        self.kernel_name = "_stack_jit_kernel"
        self.wrapper_func_name = "_wrapper"
        super(StackKernelCode, self).__init__()

    def __imports(self):
        """Generate imports for the kernel code."""
        tpl = """\
            import math
            import torch
            import triton
            from triton import language as tl
            from typing import List, Tuple, Union
            from flag_gems.utils import libentry
            from flag_gems.runtime.backend import vendor_module
            TOTAL_CORE_NUM = vendor_module.TOTAL_CORE_NUM
            MAX_NRAM_SIZE = vendor_module.MAX_NRAM_SIZE

            """
        self.tpl(textwrap.dedent(tpl))

    def __wrapper(self):
        """Generate wrapper function for the kernel code."""
        self.newline()
        tpl = """\
            def {wrapper_name}(
                tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
            ) -> torch.Tensor:
                if len(tensors) == 0:
                    raise RuntimeError("stack expected a non-empty TensorList")

                inp_shapes = [list(_.shape) for _ in tensors]
                inp0_shape = inp_shapes[0]
                for i, s in enumerate(inp_shapes[1:]):
                    if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
                        raise IndexError(
                            "Dimension out of range (expected to be in range of [{{}}, {{}}], but got {{}})".format(
                                -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                            )
                        )
                    if s != inp0_shape:
                        raise RuntimeError(
                            f"stack expects each tensor to be equal size, \
                                but got {{inp0_shape}} at entry 0 and {{s}} at entry {{i+1}}"
                        )

                if dim < 0:
                    dim = dim + len(inp0_shape) + 1
                out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
                high = int(math.prod(out_shape[:dim]))
                low = int(math.prod(out_shape[dim+1:]))
                tensor_num = len(tensors)
                out0 = torch.empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
                def grid(meta):
                    if meta['BLOCK_SIZE']>0:
                        task_x = high
                        task_y = tensor_num
                        task_z = triton.cdiv(low ,meta['BLOCK_SIZE'])
                        return (task_x, task_y, task_z)
                    else:
                        total_task = high * tensor_num
                        if meta['LOW_NUM']>0:
                            core_used = triton.cdiv(total_task // meta['LOW_NUM'], meta['TASK_PER_CORE'])
                        elif meta['N_LOW_NUM']>0:
                            core_used = triton.cdiv(high, meta['TASK_PER_CORE'])
                        return (core_used,)
                {kernel_name}[grid](
                    out0,
                    *tensors,
                    high,
                    tensor_num,
                    low,
                    )
                return out0
            """
        self.tpl(
            textwrap.dedent(tpl),
            wrapper_name=self.wrapper_func_name,
            kernel_name=self.kernel_name,
        )

    def __config(self, tensor_num, high, low, dtype):
        """Generate config for the kernel code."""
        dtyp_bytest = get_dtype_size(dtype)
        # Since the kernel has three branches, each branch has its own parameters,
        # so for a certain branch, the other parameters can be directly set to zero.

        # 1）N_LOW_NUM branch： NRAM can hold at least one set of `tensor_num * low * dtyp_bytest`.
        #    This parameter is used to indicate how many `tensor_num * low` are processed in a single core.

        # 2) LOW_NUM branch: NRAM can hold at least one set of `low * dtyp_bytest`,
        #    but cannot hold tensor_num * low * dtyp_bytest.
        #    This parameter is used to indicate how many `low` are processed in a single core.

        # 3) BLOCK_SIZE branch: NRAM is not enough to store a set of `low`,
        #    so it can only loop multiple times to process a set of `low`.
        #    This parameter is used to indicates how many elements to load
        #    at a time when looping over and processing low.
        tpl = """\
            def cfggen():
                N_LOW_NUM = {n_low_num_options}
                LOW_NUM = {low_num_options}
                BLOCK_SIZE = {block_size_options}
                warps = [1]
                num_stages = {num_stages}
                configs = [
                    triton.Config(
                        {{
                            "BLOCK_SIZE": block_size,
                            "N_LOW_NUM": n_low_num,
                            "LOW_NUM": low_num,
                        }},
                        num_warps=w,
                        num_stages=s)
                    for block_size in BLOCK_SIZE
                    for n_low_num in N_LOW_NUM
                    for low_num in LOW_NUM
                    for w in warps for s in num_stages
                ]
                return configs
            performance_related_keys = {keys}
            """

        # If `tensor_num * low * dtyp_bytest` is less than `nram_threshold`,
        # use N_LOW_NUM branch, otherwise LOW_NUM branch.
        nram_threshold = 170000
        # The maximum number of elements in triton is 1048576.
        max_elements_num = 1048576
        # after removing the overhead of pipeline and temporary variables.
        if tensor_num * low * dtyp_bytest <= nram_threshold:
            n_low_per_core = math.ceil(high / TOTAL_CORE_NUM)
            limited_by_nram = nram_threshold // dtyp_bytest // (tensor_num * low)
            limited_by_triton = max_elements_num // (tensor_num * low)
            best_opt = min(n_low_per_core, limited_by_triton, limited_by_nram)
            self.tpl(
                textwrap.dedent(tpl),
                n_low_num_options=f"{[best_opt]}",
                low_num_options=r"[0]",
                block_size_options=r"[0]",
                num_stages=r"[1]",
                keys=r'["high"]',
            )
        elif low * dtyp_bytest <= nram_threshold:
            self.tpl(
                textwrap.dedent(tpl),
                n_low_num_options=r"[0]",
                low_num_options=r"[1,2,3]",
                block_size_options=r"[0]",
                num_stages=f"{[1]}",
                keys=r'["high", "tensor_num", "low"]',
            )
        else:
            self.tpl(
                textwrap.dedent(tpl),
                n_low_num_options=r"[0]",
                low_num_options=r"[0]",
                block_size_options=r"[8192, 16384, 32768, 65536, 131072, 262144]",
                num_stages=r"[1]",
                keys=r'["low"]',
            )

    def __kernel(self, tensor_num):
        """Generate kernel code body."""
        tpl = """\
            def stack_heuristics(args, need_key):
                ret = {{
                    'TASK_PER_CORE': 0,
                    'TASK_LAST_CORE_REPEAT': 0,
                    'TASK_LAST_CORE_REMAIN': 0,
                }}
                total_task = args['high']*args['tensor_num']
                if args['LOW_NUM']>0:
                    LOW_NUM = args['LOW_NUM'] if total_task > args['LOW_NUM'] else total_task
                    ret['TASK_PER_CORE'] = triton.cdiv(total_task // LOW_NUM, TOTAL_CORE_NUM)
                    assert ret['TASK_PER_CORE']>0, ret['TASK_PER_CORE']
                    core_used = triton.cdiv(total_task // LOW_NUM, ret['TASK_PER_CORE'])
                    task_last_core = total_task-(core_used-1)*ret['TASK_PER_CORE']*LOW_NUM
                    ret['TASK_LAST_CORE_REPEAT'] = task_last_core//LOW_NUM
                    ret['TASK_LAST_CORE_REMAIN'] = task_last_core%LOW_NUM
                elif args['N_LOW_NUM']>0:
                    ret['TASK_PER_CORE'] = triton.cdiv(args['high'], TOTAL_CORE_NUM)
                    core_used = triton.cdiv(args['high'], ret['TASK_PER_CORE'])
                    ret['TASK_LAST_CORE_REPEAT'] = args['high'] -(core_used-1)*ret['TASK_PER_CORE']
                return ret[need_key]

            @triton.jit()
            def load_trans_store(
                low: tl.constexpr,
                tensor_num: tl.constexpr,
                {tensors},
                offset,
                buffer,
                buffer_offset,
                output,
                out_offset,
            ):
                if low >64:
            {low_gt_64_code}
                    tl.store(output+offset*tensor_num+out_offset, buffer)
                else:
            {low_le_64_code}
                    tl.store(output+offset*tensor_num+out_offset, tl.trans(buffer, 1, 0, 2))

            @triton.jit()
            def load_and_store(
                output_ptr,
                buffer,
                buffer_offset,
                task_id,
                LOW_NUM: tl.constexpr,
                low: tl.constexpr,
                LOW_OFFSET: tl.constexpr,
                tensor_num: tl.constexpr,
                {tensors}
            ):
                for low_idx in tl.range(LOW_NUM):
                    cur_low_id = task_id + low_idx
                    tensor_idx = cur_low_id%tensor_num
                    high_idx = cur_low_id//tensor_num
                    load_start = high_idx *low
            {load_and_store_code}
                tl.store(output_ptr+buffer_offset, buffer)

            @libentry()
            @triton.autotune(configs=cfggen(), key=performance_related_keys)
            @triton.heuristics(
                {{
                "TASK_PER_CORE": lambda args: stack_heuristics(args, "TASK_PER_CORE"),
                "TASK_LAST_CORE_REPEAT": lambda args: stack_heuristics(args, "TASK_LAST_CORE_REPEAT"),
                "TASK_LAST_CORE_REMAIN": lambda args: stack_heuristics(args, "TASK_LAST_CORE_REMAIN"),
                }}
            )
            @triton.jit()
            def {kernel_name}(
                output,
                {tensors},
                high: tl.constexpr,
                tensor_num: tl.constexpr,
                low: tl.constexpr,
                N_LOW_NUM: tl.constexpr,
                LOW_NUM: tl.constexpr,
                TASK_PER_CORE: tl.constexpr,
                TASK_LAST_CORE_REPEAT: tl.constexpr,
                TASK_LAST_CORE_REMAIN: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
                if N_LOW_NUM>0:
                    # The memory space is sufficient to hold at least one set of "tensor_num* low * type_bytes"
                    core_idx = tl.program_id(0)
                    core_used = tl.num_programs(0)
                    if core_idx>=core_used:
                        return
                    in_offset = core_idx*TASK_PER_CORE*low
                    if low >64:
                        buffer_repeat = tl.empty(shape=[N_LOW_NUM, tensor_num, low], dtype=output.dtype.element_ty)
                    else:
                        buffer_repeat = tl.empty(shape=[tensor_num, N_LOW_NUM, low], dtype=output.dtype.element_ty)
                    buffer_repeat_offset = tl.arange(0, N_LOW_NUM)[:, None]*low+tl.arange(0, low)[None,:]
                    out_repeat_offset= \\
                        tl.arange(0, N_LOW_NUM)[:,None,None]*low*tensor_num+\\
                        tl.arange(0, tensor_num)[None,:,None]*low+\\
                        tl.arange(0, low)[None, None,:]
                    if core_idx !=core_used -1:
                        for repeat_idx in range(TASK_PER_CORE//N_LOW_NUM):
                            repeat_offset = in_offset + repeat_idx*N_LOW_NUM*low
                            load_trans_store(low, tensor_num, {tensors},repeat_offset, buffer_repeat,\\
                                buffer_repeat_offset,output,out_repeat_offset)
                        if (TASK_PER_CORE%N_LOW_NUM) > 0:
                            normal_remain_offset = in_offset + (TASK_PER_CORE//N_LOW_NUM)*N_LOW_NUM*low
                            if low >64:
                                buffer_normal_remain = tl.empty(shape=[TASK_PER_CORE%N_LOW_NUM,tensor_num, low], \\
                                    dtype=output.dtype.element_ty)
                            else:
                                buffer_normal_remain = tl.empty(shape=[tensor_num,TASK_PER_CORE%N_LOW_NUM, low], \\
                                    dtype=output.dtype.element_ty)
                            buffer_normal_remain_offset = tl.arange(0, TASK_PER_CORE%N_LOW_NUM)[:, None]*low + \\
                                tl.arange(0, low)[None,:]
                            out_normal_remain_offset= \\
                                tl.arange(0, TASK_PER_CORE%N_LOW_NUM)[:,None,None]*low*tensor_num+\\
                                tl.arange(0, tensor_num)[None,:,None]*low+\\
                                tl.arange(0, low)[None, None,:]
                            load_trans_store(low, tensor_num, {tensors},normal_remain_offset, buffer_normal_remain,\\
                                buffer_normal_remain_offset,output,out_normal_remain_offset)
                    else:
                        for repeat_idx in range(TASK_LAST_CORE_REPEAT//N_LOW_NUM):
                            repeat_offset = in_offset + repeat_idx*N_LOW_NUM*low
                            load_trans_store(low, tensor_num, {tensors},repeat_offset, buffer_repeat,\\
                                buffer_repeat_offset,output,out_repeat_offset)
                        if (TASK_LAST_CORE_REPEAT%N_LOW_NUM) >0 :
                            last_core_remain_offset = in_offset + (TASK_LAST_CORE_REPEAT//N_LOW_NUM)*N_LOW_NUM*low
                            if low >64:
                                buffer_last_core_remain = \\
                                    tl.empty(shape=[TASK_LAST_CORE_REPEAT%N_LOW_NUM,tensor_num, low], \\
                                        dtype=output.dtype.element_ty)
                            else:
                                buffer_last_core_remain = \\
                                    tl.empty(shape=[tensor_num,TASK_LAST_CORE_REPEAT%N_LOW_NUM, low], \\
                                    dtype=output.dtype.element_ty)
                            buffer_last_core_remain_offset = \\
                                tl.arange(0, TASK_LAST_CORE_REPEAT%N_LOW_NUM)[:, None]*low + \\
                                tl.arange(0, low)[None,:]
                            out_last_core_remain_offset= \\
                                tl.arange(0, TASK_LAST_CORE_REPEAT%N_LOW_NUM)[:,None,None]*low*tensor_num+\\
                                tl.arange(0, tensor_num)[None,:,None]*low+\\
                                tl.arange(0, low)[None, None,:]
                            load_trans_store(low, tensor_num, {tensors},last_core_remain_offset, \\
                                buffer_last_core_remain, buffer_last_core_remain_offset, \\
                                output,out_last_core_remain_offset)
                elif LOW_NUM>0:
                    # The memory space is sufficient to hold at least one set of "low * type_bytes"
                    core_idx = tl.program_id(0)
                    core_used = tl.num_programs(0)
                    if core_idx>=core_used:
                        return
                    dtype = output.dtype.element_ty
                    buffer = tl.empty(shape=[LOW_NUM,low], dtype=dtype)
                    buffer_offset = tl.arange(0, LOW_NUM)[:,None]*low+tl.arange(0, low)[None,:]
                    LOW_OFFSET = tl.arange(0, low)
                    if core_idx != core_used-1:
                        for cycles_idx in range(TASK_PER_CORE):
                            task_id = core_idx*TASK_PER_CORE*LOW_NUM+cycles_idx*LOW_NUM
                            out_ptr = output + task_id*low
                            load_and_store(
                                out_ptr,
                                buffer,
                                buffer_offset,
                                task_id,
                                LOW_NUM,
                                low,
                                LOW_OFFSET,
                                tensor_num,
                                {tensors}
                            )
                    else:
                        base_task_id = core_idx*TASK_PER_CORE*LOW_NUM
                        for cycles_idx in range(TASK_LAST_CORE_REPEAT):
                            task_id= base_task_id+cycles_idx*LOW_NUM
                            out_ptr = output + task_id*low
                            load_and_store(
                                out_ptr,
                                buffer,
                                buffer_offset,
                                task_id,
                                LOW_NUM,
                                low,
                                LOW_OFFSET,
                                tensor_num,
                                {tensors}
                            )
                        task_id = base_task_id+TASK_LAST_CORE_REPEAT*LOW_NUM
                        output_ptr = output + task_id*low
                        for low_idx in tl.range(TASK_LAST_CORE_REMAIN):
                            cur_low_id = task_id + low_idx
                            tensor_idx = cur_low_id%tensor_num
                            high_idx = cur_low_id//tensor_num
                            load_start = high_idx *low
            {low_num_gt_0_last_core_code}
                        tl.store(output_ptr+buffer_offset, buffer, mask=buffer_offset<TASK_LAST_CORE_REMAIN*low)
                elif BLOCK_SIZE>0:
                    # Insufficient memory space to hold a set of "low* type_bytes"
                    high_idx = tl.program_id(0)
                    tensor_idx = tl.program_id(1)
                    output_ptr = output + high_idx*(low*tensor_num)+tensor_idx*low
                    offset_in_loop = tl.program_id(2)*BLOCK_SIZE+tl.arange(0, BLOCK_SIZE)
                    x = tl.empty(shape=[BLOCK_SIZE,],dtype=output.dtype.element_ty)
            {block_size_gt_0_code}
                    tl.store(output_ptr+offset_in_loop, x, mask=offset_in_loop<low)
            """

        def add_indent(cleaned_str, indent_size):
            return "\n".join(
                [f"{' ' * indent_size}{line}" for line in cleaned_str.split("\n")]
            )

        tensors = ", ".join([f"in_{idx}" for idx in range(tensor_num)])
        load_form_inputs = textwrap.dedent(
            """\
            if tensor_idx == 0:
                buffer[low_idx,:] = tl.load(in_0+load_start+LOW_OFFSET)\n"""
            + "\n".join(
                [
                    f"""\
            elif tensor_idx == {idx}:
                buffer[low_idx,:] = tl.load(in_{idx}+load_start+LOW_OFFSET)"""
                    for idx in range(1, tensor_num - 1)
                ]
            )
            + "\n"
            + f"""\
            else:
                buffer[low_idx,:] = tl.load(in_{tensor_num - 1}+load_start+LOW_OFFSET)"""
        )
        self.tpl(
            textwrap.dedent(tpl),
            kernel_name=self.kernel_name,
            tensors=tensors,
            low_gt_64_code="\n".join(
                [
                    f"{' ' * 8}buffer[:,{idx},:]=tl.load(in_{idx}+offset+buffer_offset)"
                    for idx in range(tensor_num)
                ]
            ),
            low_le_64_code="\n".join(
                [
                    f"{' ' * 8}buffer[{idx},:,:]=tl.load(in_{idx}+offset+buffer_offset)"
                    for idx in range(tensor_num)
                ]
            ),
            load_and_store_code=add_indent(load_form_inputs, 8),
            low_num_gt_0_last_core_code=add_indent(load_form_inputs, 16),
            block_size_gt_0_code=add_indent(
                textwrap.dedent(
                    """\
                if tensor_idx == 0:
                    x = tl.load(in_0+high_idx *low+offset_in_loop,mask=offset_in_loop<low)\n"""
                    + "\n".join(
                        [
                            f"""\
                elif tensor_idx == {idx}:
                    x = tl.load(in_{idx}+high_idx *low+offset_in_loop,mask=offset_in_loop<low)"""
                            for idx in range(1, tensor_num - 1)
                        ]
                    )
                    + "\n"
                    + f"""\
                else:
                    x = tl.load(in_{tensor_num - 1}+high_idx *low+offset_in_loop,mask=offset_in_loop<low)"""
                ),
                8,
            ),
        )

    def __gen_code(self, tensor_num, high, low, dtype):
        """Entry point for code generation of stack."""
        # generate imports.
        self.__imports()
        # generate config.
        self.__config(tensor_num, high, low, dtype)
        # generate kernel.
        self.__kernel(tensor_num)
        # generate wrapper function.
        self.__wrapper()

    def __call__(
        self, tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
    ) -> torch.Tensor:
        assert dim != 0, "StackKernel template does not optimize `dim=0`."
        tensor_num = len(tensors)
        inp0_shape = list(tensors[0].shape)
        out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
        high = int(math.prod(out_shape[:dim]))
        low = int(math.prod(out_shape[dim + 1 :]))
        dtype = tensors[0].dtype
        self.kernel_name = f"{self.kernel_name}_num_{tensor_num}"
        key = f"num_{tensor_num}_high_{high}_low_{low}_dtype_{dtype}"
        for tensor in tensors[1:]:
            assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        if key not in self.cache:
            # generate code and cache.
            self.__gen_code(tensor_num, high, low, dtype)
            file_name = f"{cache_dir()}/stack_{key}_pid_{self.pid}.py"
            with open(file_name, "wt", encoding="utf-8") as f:
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
        return overload(tensors, dim)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON STACK")

    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp_shapes = [list(_.shape) for _ in tensors]
    inp0_shape = inp_shapes[0]
    for i, s in enumerate(inp_shapes[1:]):
        if (dim < -tensors[i + 1].dim() - 1) or (dim > tensors[i + 1].dim()):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -tensors[i + 1].dim() - 1, tensors[i + 1].dim(), dim
                )
            )
        if s != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {s} at entry {i + 1}"
            )

    if dim < 0:
        dim = dim + len(inp0_shape) + 1
    out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
    if dim == 0:
        return vstack(tensors).view(out_shape)
    tensors = [
        tensor if tensor.is_contiguous() else tensor.contiguous() for tensor in tensors
    ]
    return StackKernelCode()(tensors, dim)
