import flag_gems
from flag_gems.utils.pointwise_dynamic import KernelGenerator as BaseKernelGenerator
from flag_gems.utils.pointwise_dynamic import ModuleGenerator as BaseModuleGenerator
from flag_gems.utils.pointwise_dynamic import WrapperGenerator, _cs, _tuple_content


class KernelGenerator(BaseKernelGenerator):
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

        code.writeline("if offset0 < s0:")
        with code.indent():
            # loads
            code.writeline("# loads")
            for i in range(schema.num_input_tensors()):
                strides = _tuple_content(tuple(f"in{i}_stride{j}" for j in range(ndim)))
                if flag_gems.vendor_name == "spacemit":
                    order = _tuple_content(
                        tuple(f"{ndim - j - 1}" for j in range(ndim))
                    )
                else:
                    order = _tuple_content(
                        tuple(f"in{i}_stride_order{j}" for j in range(ndim))
                    )
                code.writeline(
                    f"in{i}_bptr = tl.make_block_ptr("
                    f"in{i}_ptr, ({shape}), ({strides}), ({offsets}), ({tile_sizes}), order=({order}))"
                )
                code.writeline(
                    f"in{i} = tl.load(in{i}_bptr, boundary_check=({order}))"
                    f".to(in{i}_ptr.type.element_ty) "
                    "# workaround: use original ptr dtype instead of block ptr's"
                )
            code.newline()

            # compute
            # TODO: sepearate this part
            inputs_to_scalar_fn = [
                self.input_name(i) for i in range(schema.num_inputs())
            ]
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
                "# stores, block ptr store does not auto-cast"
                " the value to the pointer's dtype"
            )
            for i in range(schema.num_output_tensors()):
                strides = _tuple_content(
                    tuple(f"out{i}_stride{j}" for j in range(ndim))
                )
                if flag_gems.vendor_name == "spacemit":
                    order = _tuple_content(
                        tuple(f"{ndim - j - 1}" for j in range(ndim))
                    )
                else:
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
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_with_bptr(code)

    def gen_body_gsl_without_bptr(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_without_bptr(code)

    def gen_body_gsl_1d_tile(self, code):
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("for j in smt.parallel(0, tiles_per_cta):")
        with code.indent():
            code.writeline("tile_id = pid + j * num_ctas")
            self.gen_body_one_tile_per_cta_1d_tile(code)


class SpacemitModuleGenerator(BaseModuleGenerator):
    def __init__(
        self, function_schema, scalar_fn, ndim, jit_fn_name, wrapper_name, config
    ):
        self.config = config
        self.wrapper_gen = WrapperGenerator(
            function_schema, jit_fn_name, ndim, wrapper_name, config
        )
        self.kernel_gen = KernelGenerator(
            function_schema, scalar_fn, ndim, jit_fn_name, config
        )
        self.jit_fn_name = jit_fn_name

    @staticmethod
    def generate_imports(code):
        BaseModuleGenerator.generate_imports(code)
        code.writeline("import triton.language.extra.smt as smt")
        code.newline()
        return code


ModuleGenerator = SpacemitModuleGenerator
