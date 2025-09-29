#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor, at::Tensor> rwkv_ka_fusion(const at::Tensor &k,
                                                              const at::Tensor &kk,
                                                              const at::Tensor &a,
                                                              const at::Tensor &ka,
                                                              int64_t H,
                                                              int64_t N) {
  int64_t T = 0, C = 0;
  at::IntArrayRef k_sizes = k.sizes();
  at::Tensor o_k, o_kk, o_kka;
  if (k.dim() == 1) {
    T = 1;
    C = k_sizes[0];
  } else {
    T = k.sizes()[0];
    C = k.sizes()[1];
  }
  o_k = at::empty_like(k, k.options());
  o_kk = at::empty_like(k, k.options());
  o_kka = at::empty_like(k, k.options());

  const TritonJITFunction &f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "rwkv_ka_fusion.py"),
      "rwkv_ka_fusion_kernel");

  // add utility to build this automatically
  int64_t block_size = C;
  const int num_warps = 4;
  const int num_stages = 8;
  int64_t N_size = utils::next_power_of_2(N);

  const unsigned int num_blocks = (T * C + block_size - 1) / block_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(o_k.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    k,
    kk,
    a,
    ka,
    o_k,
    o_kk,
    o_kka,
    T,
    C,
    H,
    N,
    N_size,
    block_size);
  return std::make_tuple(o_k, o_kk, o_kka);
}

}  // namespace flag_gems
