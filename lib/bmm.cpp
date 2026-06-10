#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"

namespace flag_gems {
using namespace triton_jit;

static inline int64_t cdiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

at::Tensor bmm(const at::Tensor& A_in, const at::Tensor& B_in) {
  TORCH_CHECK(A_in.dim() == 3 && B_in.dim() == 3, "both the tensors must be 3-D");
  TORCH_CHECK(A_in.dtype() == B_in.dtype(),
              "expected a and b to have the same dtype, but got: ",
              A_in.dtype(),
              " != ",
              B_in.dtype());

  at::Tensor A = A_in.contiguous();
  at::Tensor B = B_in.contiguous();

  at::IntArrayRef A_sizes = A.sizes();
  at::IntArrayRef B_sizes = B.sizes();

  const int64_t batch = A_sizes[0];
  const int64_t M = A_sizes[1];
  const int64_t N = B_sizes[2];
  const int64_t K = A_sizes[2];

  at::Tensor out = at::empty({batch, M, N}, A.options());

#if defined(FLAGGEMS_USE_IX)
  // On IX (CoreX/iLuvatar) backend, the batched bmm kernel triggers a
  // "divergent base ptr" compiler error due to Triton compiler limitations.
  // Work around by dispatching per-batch 2D mm kernels instead.
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "mm.py"),
                                      "mm_kernel_general");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const int BLOCK_M = 64;
  const int BLOCK_N = 128;
  const int BLOCK_K = 64;
  const int num_stages = 2;
  const int num_warps = 4;
  const int GROUP_M = 8;

  unsigned int grid_x = static_cast<unsigned int>(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N));

  for (int64_t b = 0; b < batch; ++b) {
    at::Tensor a_slice = A[b];
    at::Tensor b_slice = B[b];
    at::Tensor o_slice = out[b];

    f(/* stream = */ raw_stream,
      /* grid_x = */ grid_x,
      /* grid_y = */ 1u,
      /* grid_z = */ 1u,
      num_warps,
      num_stages,
      a_slice,
      b_slice,
      o_slice,
      (int64_t)M,
      (int64_t)N,
      (int64_t)K,
      a_slice.stride(0),
      a_slice.stride(1),
      b_slice.stride(0),
      b_slice.stride(1),
      o_slice.stride(0),
      o_slice.stride(1),
      /* BLOCK_M = */ BLOCK_M,
      /* BLOCK_N = */ BLOCK_N,
      /* BLOCK_K = */ BLOCK_K,
      /* GROUP_M = */ GROUP_M);
  }
#else
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "bmm.py"),
                                      "bmm_kernel");

  // bmm_kernel @libtuner(key=["M", "N", "K", "stride_am", "stride_bk"])
  // -- IS_FP64 (constexpr, default False) is not in V0 user-kwargs path:
  // test_bmm.py covers fp16/fp32/bf16 only, so the JITFunction default
  // wins via autotune_bridge defaults fallback. fp64 reserved for V1 (Q13).
  static AutotunedCall ac(std::string(utils::get_flag_gems_src_path() / "ops" / "bmm.py"),
                          "bmm_kernel",
                          {"M", "N", "K", "stride_am", "stride_bk"});

  // Grid lambda for the SQL-miss bench path (Q12). `batch` is captured by
  // value: it is not a kernel arg (only M/N/TILE_M/TILE_N come through the
  // Triton meta dict), so grid_fn synthesizes the z dim from C++ state.
  auto grid_fn = [batch](const triton_jit::Config& c) -> std::tuple<unsigned, unsigned, unsigned> {
    int64_t Mv = get_int_kwarg(c, "M");
    int64_t Nv = get_int_kwarg(c, "N");
    int64_t tm = get_int_kwarg(c, "TILE_M");
    int64_t tn = get_int_kwarg(c, "TILE_N");
    return {static_cast<unsigned>((Mv + tm - 1) / tm),
            static_cast<unsigned>((Nv + tn - 1) / tn),
            static_cast<unsigned>(batch)};
  };

  const triton_jit::Config& cfg = ac.lookup(TuneKey {M, N, K, A.stride(1), B.stride(1)},
                                            grid_fn,
                                            A,
                                            B,
                                            out,
                                            (int)M,
                                            (int)N,
                                            (int)K,
                                            A.stride(0),
                                            A.stride(1),
                                            A.stride(2),
                                            B.stride(0),
                                            B.stride(1),
                                            B.stride(2),
                                            out.stride(0),
                                            out.stride(1),
                                            out.stride(2));

  const int64_t tile_m = get_int_kwarg(cfg, "TILE_M");
  const int64_t tile_n = get_int_kwarg(cfg, "TILE_N");
  unsigned int grid_x = static_cast<unsigned int>((M + tile_m - 1) / tile_m);
  unsigned int grid_y = static_cast<unsigned int>((N + tile_n - 1) / tile_n);

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  f.autotuned_call(raw_stream,
                   grid_x,
                   grid_y,
                   (unsigned int)batch,
                   cfg,
                   A,
                   B,
                   out,
                   (int)M,
                   (int)N,
                   (int)K,
                   A.stride(0),
                   A.stride(1),
                   A.stride(2),
                   B.stride(0),
                   B.stride(1),
                   B.stride(2),
                   out.stride(0),
                   out.stride(1),
                   out.stride(2));
#endif
  return out;
}

}  // namespace flag_gems
