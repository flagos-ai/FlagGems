import torch
import triton

cluster_num = 12
core_num = 64
thread_num = core_num * cluster_num
buf_len_per_core = 2048


def get_block_size_1d(n: int, element_size: int) -> int:
    return min(
        triton.next_power_of_2(triton.cdiv(n, cluster_num)),
        triton.cdiv(buf_len_per_core * core_num, element_size),
    )
    # if triton.cdiv(n, block_size) > 256:
    #     return triton.next_power_of_2(triton.cdiv(n, 256))
    # else:
    #     return block_size
    # return min(
    #     triton.next_power_of_2(triton.cdiv(n, cluster_num)),
    #     triton.cdiv(buf_len_per_core * core_num, element_size),
    # )


def heur_m_block_size(args):
    # if triton.next_power_of_2(triton.cdiv(args["M"], cluster_num)) < core_num:
    #     return triton.next_power_of_2(triton.cdiv(args["M"], cluster_num))
    # else:
    return (
        triton.cdiv(triton.cdiv(buf_len_per_core, args["ELEMENT_SIZE"]), args["N"])
        * core_num
    )


def heur_n_block_size(args):
    return min(args["N"], triton.cdiv(buf_len_per_core, args["ELEMENT_SIZE"]))


TORCH_DTYPE_ITEMSIZE = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int32: 4,
    torch.int16: 2,
    torch.int64: 8,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
}


def get_element_size(dtype):
    return TORCH_DTYPE_ITEMSIZE[dtype]
