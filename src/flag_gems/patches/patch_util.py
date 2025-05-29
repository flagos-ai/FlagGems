def patch_module_method(cls, method_name: str, new_method: callable, verbose=True):
    old_method = getattr(cls, method_name, None)
    setattr(cls, method_name, new_method)
    if verbose:
        print(
            f"Patched VLLM {cls.__name__}.{method_name} with FLAGGEMS {new_method.__name__}"
        )
    return old_method  # incase we need to revert the patch later
