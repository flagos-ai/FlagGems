#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor zeros(at::IntArrayRef size,
                 c10::optional<at::ScalarType> dtype,
                 c10::optional<at::Layout> layout,
                 c10::optional<at::Device> device,
                 c10::optional<bool> pin_memory) {
  int64_t n_elements = 1;
  for (auto dim : size) {
    n_elements *= dim;
  }

  auto options =
      at::TensorOptions()
          .dtype(dtype.value_or(at::typeMetaToScalarType(at::get_default_dtype())))
          .layout(layout.value_or(at::kStrided))
          .device(device.value_or(torch::cuda::is_available() ? at::Device(at::kCUDA) : at::Device(at::kCPU)))
          .pinned_memory(pin_memory.value_or(false));

  TORCH_CHECK(n_elements >= 0, "Total elements must be non-negative");

  if (n_elements == 0) {
    return at::empty(size, options);
  }

  at::Tensor out = at::empty(size, options);

  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;

  const uint64_t num_blocks = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "zeros.py"), "zeros_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(raw_stream,
    num_blocks,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    /* num_warps = */ num_warps,
    /* num_stages = */ num_stages,
    out,
    n_elements,
    tile_size);

  return out;
}
}  // namespace flag_gems
