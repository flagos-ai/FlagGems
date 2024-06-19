import triton


class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.kernel_cache = dict()
        fn = self.fn
        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        self.jit_function: triton.runtime.JITFunction = fn
        self.kernel_arg_indices = []
        for p in self.jit_function.params:
            if not p.is_constexpr:
                self.kernel_arg_indices.append(p.num)

    def run(self, *args, **kwargs):
        key = []
        for arg in args:
            if hasattr(arg, "data_ptr"):
                key.append(arg.dtype)
                key.append(arg.data_ptr() % self.divisibility == 0)
            elif isinstance(arg, int):
                key.append(arg)
        entry_key = tuple(key)
        if entry_key not in self.kernel_cache:
            kernel = self.fn.run(*args, **kwargs)
            self.kernel_cache[entry_key] = kernel
        else:
            kernel = self.kernel_cache[entry_key]

        # collect all the arguments to the kernel, all non-constexpr arguments
        k_args = [arg for i, arg in enumerate(args) if i in self.kernel_arg_indices]
        if len(k_args) < len(self.kernel_arg_indices):
            for p in self.jit_function.params[len(args) :]:
                if not p.is_constexpr:
                    if p.name in kwargs:
                        k_args.append(kwargs[p.name])
                    else:
                        k_args.append(p.default)

        grid = kwargs["grid"]
        if callable(grid):
            # collect all arguments to the grid fn，ie:
            # 1. args,
            # 2. kwargs,
            # 3. all all other captured arguments in CompiledKernel from Autotunner & Heuristics
            # when kwargs & captured args conflict, captured args have higher priority
            for k, v in kernel.constants.items():
                arg_name = self.arg_names[int(k)]
                kwargs[arg_name] = v
            meta = {**dict(zip(self.arg_names, args)), **kwargs}
            grid = grid(meta)
        grid = grid + (1, 1)

        kernel[grid[0:3]](*k_args)
        return


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator
