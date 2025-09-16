#include <torch/torch.h>
#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

void reshape_and_cache_flash(const at::Tensor& key,
                             const at::Tensor& value,
                             at::Tensor& key_cache,
                             at::Tensor& value_cache,
                             const at::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             const c10::optional<at::Tensor>& k_scale,
                             const c10::optional<at::Tensor>& v_scale) {
  const auto num_tokens = slot_mapping.size(0);
  const auto num_heads = key.size(1);
  const auto head_size = key.size(2);
  const auto block_size = key_cache.size(1);

  const auto key_stride = key.stride(0);
  const auto value_stride = value.stride(0);
  const auto block_stride = key_cache.stride(0);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0), "Key and Value cache strides must be equal");

  unsigned int grid_x = num_tokens;
  unsigned int grid_y = 1;
  unsigned int grid_z = 1;
  const int num_warps = 4;
  const int num_stages = 2;

  c10::DeviceGuard guard(key.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = stream.stream();

  const TritonJITFunction& kernel = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "reshape_and_cache_flash.py"),
      "reshape_and_cache_flash_kernel");

  // 1.0f is mean not scaling
  // float k_scale_val = k_scale.has_value() ? k_scale.value().item<float>() : 1.0f;
  // float v_scale_val = v_scale.has_value() ? v_scale.value().item<float>() : 1.0f;

  kernel(raw_stream,
         grid_x,
         grid_y,
         grid_z,
         num_warps,
         num_stages,
         key,
         value,
         key_cache,
         value_cache,
         slot_mapping,
         block_stride,
         key_stride,
         value_stride,
         num_heads,
         head_size,
         block_size,
         std::nullopt,
         std::nullopt,
         num_heads * head_size);
}
}  // namespace flag_gems
