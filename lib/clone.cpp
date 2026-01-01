#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <vector>
#include "c10/core/DispatchKeySet.h"
#include "flag_gems/operators.h"
#include "flag_gems/pointwise_launch.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

namespace {

  constexpr int kCloneBlockSize = 1024;

}  // namespace

// clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
at::Tensor clone(const at::Tensor &self, c10::optional<at::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(at::MemoryFormat::Preserve);
  at::Tensor out;
  if (memory_format == at::MemoryFormat::Preserve && self.is_non_overlapping_and_dense()) {
    out = at::empty_strided(self.sizes(), self.strides(), self.options());
  } else {
    out = at::empty_like(self, self.options(), memory_format);
  }

  /***
   *
   * the original clone.cpp logic is
   *   if (src._is_zerotensor()) {
   *      out.zero_();
   *   } else {
   *      out.copy_(self);
   * }
   * but we replace out.copy_ with our triton copy kernel launch
   * ***/

  c10::DeviceGuard guard(self.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  const std::string copy_kernel_path = (utils::get_triton_src_path() / "copy.py").string();

  const int64_t ndim = out.dim();
  if (ndim >= 0 && ndim <= 2) {
    if (launch_pointwise_unary_rank0_to2(self,
                                         out,
                                         ndim,
                                         raw_stream,
                                         copy_kernel_path,
                                         "_copy_kernel_kernel")) {
      return out;
    }
  }

  const int64_t numel = self.numel();
  const unsigned int grid_x = (numel + kCloneBlockSize - 1) / kCloneBlockSize;
  if (self.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
    const TritonJITFunction &kernel_linear =
        TritonJITFunction::get_instance(copy_kernel_path, "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, self, out, numel, kCloneBlockSize);
    return out;
  }

  std::vector<int64_t> task_shape(self.sizes().begin(), self.sizes().end());
  int NDIMS = task_shape.size();

  std::vector<int64_t> src_stride(self.strides().begin(), self.strides().end());
  std::vector<int64_t> dst_stride(out.strides().begin(), out.strides().end());

  const TritonJITFunction &kernel_nd = TritonJITFunction::get_instance(copy_kernel_path, "copy_kernel_nd");
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
