#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <vector>
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

  constexpr int kToBlockSize = 1024;

  bool can_use_flag_gems_to_tensor(const at::Tensor &tensor) {
    if (tensor.layout() != at::kStrided) {
      return false;
    }
    if (tensor.device().type() != c10::DeviceType::CUDA) {
      return false;
    }
    if (tensor.is_quantized()) {
      return false;
    }
    if (tensor.is_complex()) {
      return false;
    }
    return true;
  }

  at::Tensor allocate_like(const at::Tensor &self,
                           const at::TensorOptions &options,
                           c10::optional<at::MemoryFormat> memory_format) {
    if (memory_format.has_value()) {
      return at::empty_like(self, options, memory_format);
    }
    return at::empty_like(self, options);
  }

  void launch_copy_kernels(const at::Tensor &self, at::Tensor &out) {
    if (self._is_zerotensor()) {
      out.zero_();
      return;
    }

    const int64_t numel = self.numel();
    if (numel == 0) {
      return;
    }

    const unsigned int grid_x = (numel + kToBlockSize - 1) / kToBlockSize;

    c10::DeviceGuard guard(self.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    if (self.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
      const TritonJITFunction &kernel_linear =
          TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                          "copy_kernel_linear");
      kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, self, out, numel, kToBlockSize);
      return;
    }

    std::vector<int64_t> shape(self.sizes().begin(), self.sizes().end());
    int NDIMS = shape.size();
    std::vector<int64_t> src_stride(self.strides().begin(), self.strides().end());
    std::vector<int64_t> dst_stride(out.strides().begin(), out.strides().end());

    const TritonJITFunction &kernel_nd =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_nd");
    kernel_nd(raw_stream,
              grid_x,
              1,
              1,
              4,
              0,
              self,
              out,
              torch::tensor(shape, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
              torch::tensor(src_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
              torch::tensor(dst_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
              numel,
              NDIMS,
              kToBlockSize);
  }

}  // namespace

at::Tensor to_dtype(const at::Tensor &self,
                    at::ScalarType dtype,
                    bool /*non_blocking*/,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
  // std::cout << "[flag_gems][to_dtype] gems::to_dtype" << std::endl;
  if (!copy && self.scalar_type() == dtype) {
    return self;
  }

  TORCH_CHECK(can_use_flag_gems_to_tensor(self),
              "FlagGems to.dtype currently supports CUDA strided, non-quantized, real tensors only.");

  auto options = self.options().dtype(dtype);
  at::Tensor out = allocate_like(self, options, memory_format);
  launch_copy_kernels(self, out);
  return out;
}

at::Tensor to_other(const at::Tensor &self,
                    const at::Tensor &other,
                    bool /*non_blocking*/,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
  const bool same_dtype = self.scalar_type() == other.scalar_type();
  const bool same_device = self.device() == other.device();
  if (!copy && same_dtype && same_device) {
    return self;
  }

  TORCH_CHECK(can_use_flag_gems_to_tensor(self),
              "FlagGems to.other currently supports CUDA strided, non-quantized, real tensors only (self).");
  TORCH_CHECK(can_use_flag_gems_to_tensor(other),
              "FlagGems to.other currently supports CUDA strided, non-quantized, real tensors only (other).");
  TORCH_CHECK(same_device,
              "FlagGems to.other currently supports tensors that reside on the same CUDA device.");

  auto options = self.options().dtype(other.scalar_type()).device(other.device());
  at::Tensor out = allocate_like(self, options, memory_format);
  launch_copy_kernels(self, out);
  return out;
}

}  // namespace flag_gems
