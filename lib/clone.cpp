#include <ATen/ops/clone_ops.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <vector>
#include "c10/core/DispatchKeySet.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

  constexpr int kCloneBlockSize = 1024;

  const c10::DispatchKeySet &fallback_dispatch_keyset() {
    static const c10::DispatchKeySet keys = c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd);
    return keys;
  }

  bool can_use_flag_gems_clone(const at::Tensor &self) {
    if (self.layout() != at::kStrided) {
      return false;
    }
    if (self.device().type() != c10::DeviceType::CUDA) {
      return false;
    }
    if (self.is_quantized()) {
      return false;
    }
    if (self.is_complex()) {
      return false;
    }
    return true;
  }

}  // namespace

at::Tensor clone(const at::Tensor &self, c10::optional<at::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(at::MemoryFormat::Preserve);

  if (!can_use_flag_gems_clone(self)) {
    return at::_ops::clone::redispatch(fallback_dispatch_keyset(), self, optional_memory_format);
  }

  auto tensor_options = self.options();
  auto allocate_output = [&]() {
    if (memory_format == at::MemoryFormat::Preserve && self.is_non_overlapping_and_dense()) {
      return at::empty_strided(self.sizes(), self.strides(), tensor_options);
    }
    return at::empty_like(self, tensor_options, memory_format);
  };

  at::Tensor out = allocate_output();

  if (self._is_zerotensor()) {
    out.zero_();
    return out;
  }

  const int64_t numel = self.numel();
  if (numel == 0) {
    return out;
  }

  const unsigned int grid_x = (numel + kCloneBlockSize - 1) / kCloneBlockSize;

  c10::DeviceGuard guard(self.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  if (self.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
    const TritonJITFunction &kernel_linear =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, self, out, numel, kCloneBlockSize);
    return out;
  }

  std::vector<int64_t> task_shape(self.sizes().begin(), self.sizes().end());
  int NDIMS = task_shape.size();

  std::vector<int64_t> src_stride(self.strides().begin(), self.strides().end());
  std::vector<int64_t> dst_stride(out.strides().begin(), out.strides().end());

  const TritonJITFunction &kernel_nd =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(), "copy_kernel_nd");
  kernel_nd(raw_stream,
            grid_x,
            1,
            1,
            4,
            0,
            self,
            out,
            torch::tensor(task_shape, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            torch::tensor(src_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            torch::tensor(dst_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            numel,
            NDIMS,
            kCloneBlockSize);

  return out;
}

}  // namespace flag_gems
