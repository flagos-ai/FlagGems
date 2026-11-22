import triton.language as tl


@triton.jit
def div_mode(a, b):
    # TODO: 后续后端统一清理依赖
    return a / b


@triton.jit
def div_mode_(a, b):
    # TODO: 后续后端统一清理依赖
    return a / b
