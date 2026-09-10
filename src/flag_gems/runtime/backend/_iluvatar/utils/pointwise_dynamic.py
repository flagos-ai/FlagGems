# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flag_gems.utils.pointwise_dynamic import KernelGenerator as BaseKernelGenerator
from flag_gems.utils.pointwise_dynamic import ModuleGenerator as BaseModuleGenerator
from flag_gems.utils.pointwise_dynamic import WrapperGenerator, _cs, _type_name


class KernelGenerator(BaseKernelGenerator):
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
                ann = "" if ndim == 1 else ": int"

                # strides for inputs
                for i in range(schema.num_input_tensors()):
                    stride_args = _cs(f"in{i}_stride{j}{ann}" for j in range(ndim))
                    code.writeline(f"{stride_args}, # strides for in{i}")

                # strides for outputs
                for i in range(schema.num_output_tensors()):
                    stride_args = _cs(f"out{i}_stride{j}{ann}" for j in range(ndim))
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


class ModuleGenerator(BaseModuleGenerator):
    def __init__(
        self, function_schema, scalar_fn, ndim, jit_fn_name, wrapper_name, config
    ):
        self.config = config
        self.scalar_fn = scalar_fn
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )
