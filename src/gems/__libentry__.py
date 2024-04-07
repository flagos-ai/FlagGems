import triton


class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.config_cache = dict()
        self.kernel_cache = dict()

    def get_key(self, all_args):
        key = []
        for name in self.arg_names:
            if name in all_args:
                arg = all_args[name]
                if hasattr(arg, "data_ptr"):
                    key.append(arg.dtype)
                    key.append(arg.data_ptr() % self.divisibility == 0)
                elif isinstance(arg, int):
                    key.append(arg)
        return tuple(key)

    def run(self, *args, **kwargs):
        nargs = dict(zip(self.arg_names, args))
        all_args = {**nargs, **kwargs}
        entry_key = self.get_key(all_args)
        config = {}
        # Autotuner
        if isinstance(self.fn, triton.runtime.Autotuner):
            if entry_key not in self.config_cache:
                # tune
                kernel = self.fn.run(*args, **kwargs)
                config = self.fn.best_config.kwargs
                self.config_cache[entry_key] = config
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # tuned
                config = self.config_cache[entry_key]
                kernel = self.kernel_cache[entry_key]
        # Heuristics
        elif isinstance(self.fn, triton.runtime.Heuristics):
            if entry_key not in self.kernel_cache:
                # compile
                kernel = self.fn.run(*args, **kwargs)
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # compiled
                for v, heur in self.fn.values.items():
                    config[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
                kernel = self.kernel_cache[entry_key]
        # JitFunction
        else:
            if entry_key not in self.kernel_cache:
                # compile
                kernel = self.fn.run(*args, **kwargs)
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # compiled
                args = [
                    arg
                    for i, arg in enumerate(args)
                    if not self.fn.params[i].is_constexpr
                ]
                kernel = self.kernel_cache[entry_key]
        grid = kwargs["grid"]
        if isinstance(grid, type(lambda: None)):
            # grid_fn
            current = dict(**kwargs, **config)
            meta = {**dict(zip(self.arg_names, args)), **current}
            grid = grid(meta)

        # allow grid len < 3
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        grid = (grid_0, grid_1, grid_2)

        kernel[grid](*args)
        return


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator
