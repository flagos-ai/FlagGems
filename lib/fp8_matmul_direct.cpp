#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

#include <cuda.h>
#include <string>

namespace flag_gems {
using namespace triton_jit;

at::Tensor fp8_matmul_direct(const at::Tensor& a,
                             const at::Tensor& a_s,
                             const at::Tensor& b,
                             const at::Tensor& b_s) {
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "a and b must be contiguous");
  TORCH_CHECK(a_s.is_contiguous() && b_s.is_contiguous(),
              "a_s and b_s must be contiguous");

  int64_t K = a.size(-1);
  int64_t M = a.numel() / K;
  int64_t N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "a and b must have the same K dimension");

  auto out_shape = a.sizes().vec();
  out_shape.back() = N;

  at::Tensor a_2d = a.view({M, K});
  at::Tensor a_s_2d = a_s.view({M, -1});
  at::Tensor C = at::empty({M, N}, a.options().dtype(at::kBFloat16));

  const int BLOCK_M = 64;
  const int BLOCK_N = 64;

  // Get JIT function instance (cached singleton)
  static const std::string kernel_path =
      std::string(utils::get_flag_gems_src_path() / "ops" / "fp8_matmul.py");
  const TritonJITFunction& jit =
      TritonJITFunction::get_instance(kernel_path, "_fp8_matmul_kernel");

  // Build signature and args manually, matching ArgHandle behavior
  // but without variadic template overhead.
  //
  // Kernel params: A, B, C, As, Bs, M, N, K,
  //   stride_am, stride_ak, stride_bn, stride_bk,
  //   stride_cm, stride_cn, stride_as_m, stride_as_k,
  //   stride_bs_n, stride_bs_k,
  //   GROUP_K(constexpr), BLOCK_M(ce), BLOCK_N(ce), BLOCK_K(ce), GROUP_SIZE_M(ce)

  // Storage (stack-allocated, must outlive cuLaunchKernel)
  void* tp[5];  // tensor data pointers
  int32_t iv[16]; // int values (scalars + strides that aren't :1)
  void* scratch[2] = {nullptr, nullptr};

  void* arg_ptrs[25];
  int na = 0;  // arg count
  int ni = 0;  // int storage index

  std::string sig;
  sig.reserve(256);
  bool first = true;

  // Append separator
  auto sep = [&]() { if (!first) sig += ','; first = false; };

  // Add tensor (always SPECIALIZED for this kernel)
  auto add_t = [&](const at::Tensor& t, int ti) {
    tp[ti] = t.data_ptr();
    arg_ptrs[na++] = &tp[ti];
    sep();
    sig += '*';
    sig += to_triton_typename(t.scalar_type());
    sig += spec(reinterpret_cast<std::uintptr_t>(tp[ti]));
  };

  // Add specialized int32
  auto add_i = [&](int32_t val) {
    const char* s = spec(val);
    sep();
    sig += "i32";
    sig += s;
    if (s[0] == '\0' || (s[0] == ':' && s[1] != '1')) {
      // Not :1 → push arg
      iv[ni] = val;
      arg_ptrs[na++] = &iv[ni];
      ni++;
    } else if (s[0] == ':' && s[1] == '1' && s[2] == '\0') {
      // :1 → specialized away, not passed
    } else {
      // :16 or other → push arg
      iv[ni] = val;
      arg_ptrs[na++] = &iv[ni];
      ni++;
    }
  };

  // Add constexpr
  auto add_ce = [&](int32_t val) {
    sep();
    sig += std::to_string(val);
  };

  // --- Pack all args ---
  // 5 tensors
  add_t(a_2d, 0);
  add_t(b, 1);
  add_t(C, 2);
  add_t(a_s_2d, 3);
  add_t(b_s, 4);

  // 3 scalars
  add_i(static_cast<int32_t>(M));
  add_i(static_cast<int32_t>(N));
  add_i(static_cast<int32_t>(K));

  // 10 strides
  add_i(static_cast<int32_t>(a_2d.stride(0)));   // stride_am
  add_i(static_cast<int32_t>(a_2d.stride(1)));   // stride_ak (1 for contiguous)
  add_i(static_cast<int32_t>(b.stride(0)));       // stride_bn
  add_i(static_cast<int32_t>(b.stride(1)));       // stride_bk (1 for contiguous)
  add_i(static_cast<int32_t>(C.stride(0)));       // stride_cm
  add_i(static_cast<int32_t>(C.stride(1)));       // stride_cn (1 for contiguous)
  add_i(static_cast<int32_t>(a_s_2d.stride(0))); // stride_as_m
  add_i(static_cast<int32_t>(a_s_2d.stride(1))); // stride_as_k (1 when K/128==1)
  add_i(static_cast<int32_t>(b_s.stride(0)));     // stride_bs_n
  add_i(static_cast<int32_t>(b_s.stride(1)));     // stride_bs_k

  // 5 constexprs
  add_ce(128);  // GROUP_K
  add_ce(64);   // BLOCK_M
  add_ce(64);   // BLOCK_N
  add_ce(128);  // BLOCK_K
  add_ce(4);    // GROUP_SIZE_M

  // Global scratch (2x null void*)
  arg_ptrs[na++] = &scratch[0];
  arg_ptrs[na++] = &scratch[1];

  // Launch
  unsigned int grid_x = utils::cdiv(static_cast<int>(M), BLOCK_M) *
                        utils::cdiv(static_cast<int>(N), BLOCK_N);

  c10::DeviceGuard guard(C.device());
  backend::StreamType stream = backend::getCurrentStream();

  jit.launch_with_raw_args(
      stream,
      grid_x, 1, 1,  // grid
      4,              // num_warps
      3,              // num_stages
      std::move(sig),
      arg_ptrs,
      na);

  return C.view(out_shape);
}

}  // namespace flag_gems
