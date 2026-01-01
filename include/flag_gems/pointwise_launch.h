#pragma once

#include <c10/cuda/CUDAStream.h>
#include <string_view>
#include "torch/torch.h"

namespace flag_gems {

// Launches pointwise_dynamic-generated unary kernels (1 input + 1 output)
// for rank 0 to rank 2 tensors. The kernel_prefix should be the common
// prefix of the Triton entry point names (e.g. "_copy_kernel_kernel").
bool launch_pointwise_unary_rank0_to2(const at::Tensor &src,
                                      at::Tensor &dst,
                                      const int64_t rank,
                                      CUstream raw_stream,
                                      std::string_view triton_module_path,
                                      std::string_view kernel_name_prefix);

}  // namespace flag_gems
