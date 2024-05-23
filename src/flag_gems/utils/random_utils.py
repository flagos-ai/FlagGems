import torch

# This function is roughly a python wrapper of CUDAGeneratorImpl::philox_cuda_state in Pytorch.
# https://github.com/pytorch/pytorch/blob/8a4597980c2692b73f35fb3c7145eaeaf2273e77/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L452
# It returns the current state of the default Philox RNG in seed and offset and
# updates the next offset by adding `increment`.
def philox_cuda_seed_offset(increment, device=None):
    device = device or torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state_copy = gen.get_state()
    c0, c1 = state_copy.view(torch.int64)
    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset


def per_thread_offset(N, num_blocks, num_warps, warp_threads=32):
    block_threads = num_warps * warp_threads
    max_threads = num_blocks * block_threads
    offset = (N + max_threads - 1) // max_threads
    return offset
