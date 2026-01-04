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

  const int64_t rank = self.sizes().size();
  if (rank >= 0 && rank <= 4) {
    if (launch_pointwise_low_rank(self, out, rank, raw_stream, copy_kernel_path, "_copy_kernel_cppwrapper")) {
      return out;
    }
  }

  // fallback to original implementation
  out.copy_(self);
  return out;
}

}  // namespace flag_gems
