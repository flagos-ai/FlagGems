#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor contiguous(const at::Tensor &self, at::MemoryFormat memory_format) {
  TORCH_CHECK(memory_format == at::MemoryFormat::Contiguous);
  if (self.is_contiguous(memory_format = memory_format)) {
    return self;
  }
  at::Tensor out = at::empty_like(self, memory_format = memory_format);

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "contiguous.py"),
                                      "copy_kernel");

  int64_t tile_size = 1024;
  const int num_warps = 4;
  const int num_stages = 1;
  int64_t n = out.numel();
  int64_t ndim = out.dim();
  auto options = torch::TensorOptions().device(self.device()).dtype(torch::kInt64);
  at::Tensor input_sizes = torch::tensor(self.sizes(), options);
  at::Tensor input_strides = torch::tensor(self.strides(), options);
  at::Tensor out_strides = torch::tensor(out.strides(), options);
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    self,
    out,
    input_strides,
    out_strides,
    input_sizes,
    ndim,
    n,
    tile_size);
  return out;
}
}  // namespace flag_gems
