#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor rwkv_mm_sparsity(const at::Tensor &k, const at::Tensor &v) {
  at::IntArrayRef k_sizes = k.sizes();
  at::IntArrayRef v_sizes = v.sizes();

  at::Tensor out = at::empty({v_sizes[1]}, k.options());

  const TritonJITFunction &f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "rwkv_mm_sparsity.py"),
      "rwkv_mm_sparsity_kernel");

  // add utility to build this automatically
  int64_t blk_size = 512;
  int64_t block_size = 16;
  const int num_warps = 4;
  const int num_stages = 8;
  int64_t k_size = utils::next_power_of_2(k_sizes[0]);

  const unsigned int num_blocks = (v_sizes[1] + block_size - 1) / block_size;

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
  f(raw_stream, num_blocks, 1, 1, num_warps, num_stages, k, v, out, v_sizes[1], blk_size, k_size, block_size);
  return out;
}

}  // namespace flag_gems
