#pragma once
#include <torch/torch.h>

inline cudaStream_t get_flag_stream() {
    return at::cuda::getCurrentCUDAStream();
}

// FlagOS 内置异步DMA预取原语声明
__device__ void flag_async_copy(void* dst, const void* src, size_t bytes);

template<typename T>
__device__ __forceinline__ T rsqrt(T x) {
    return rsqrt((float)x);
}
