#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <ATen/cuda/CUDAContext.h>  // for getCurrentDevice, getDeviceProperties
#include <cuda_runtime.h>           // for cudaDeviceProp
#include <iostream>
#include <tuple>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

struct DeviceInfo {
  int device_id;
  size_t l2_cache_size;
  int sm_count;
  int major;
};

inline const DeviceInfo &get_device_info() {
  static const DeviceInfo info = []() {
    DeviceInfo dev_info {};
    if (cudaGetDevice(&dev_info.device_id) != cudaSuccess) {
      dev_info.device_id = 0;  // fallback
    }

    cudaDeviceProp props {};
    if (cudaGetDeviceProperties(&props, dev_info.device_id) == cudaSuccess) {
#if CUDART_VERSION >= 11020
      dev_info.l2_cache_size = props.l2CacheSize;
#else
      dev_info.l2_cache_size = 40ull * 1024 * 1024;  // fallback
#endif
      dev_info.sm_count = props.multiProcessorCount;
      dev_info.major = props.major;
    } else {
      // fallback for A100 默认值
      dev_info.l2_cache_size = 40ull * 1024 * 1024;
      dev_info.sm_count = 108;
      dev_info.major = 8;  // A100 compute capability major
    }
    return dev_info;
  }();
  return info;
}

inline int get_device_id() {
  return get_device_info().device_id;
}
inline size_t get_l2_cache_size() {
  return get_device_info().l2_cache_size;
}
inline int get_sm_count() {
  return get_device_info().sm_count;
}
inline int get_major() {
  return get_device_info().major;
}

static inline int64_t cdiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

bool streamk_scenario(const at::Tensor &a, const at::Tensor &b, int64_t M, int64_t N, int64_t K) {
  bool a_is_half_or_bf16 = (a.scalar_type() == at::kHalf) || (a.scalar_type() == at::kBFloat16);
  bool b_is_half_or_bf16 = (b.scalar_type() == at::kHalf) || (b.scalar_type() == at::kBFloat16);
  return (a_is_half_or_bf16 && b_is_half_or_bf16 && get_major() == 8 && K > M * 5 && K > N * 5);
}

void streamk_mm_tensor(const at::Tensor &a,
                       const at::Tensor &b,
                       at::Tensor &c,
                       int64_t M,
                       int64_t N,
                       int64_t K,
                       int sm_count = 108) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(a.dtype() == b.dtype(), "expected a and b to have the same dtype");

  // config (same as python defaults)
  const int BLOCK_M = 128;
  const int BLOCK_N = 128;
  const int BLOCK_K = 128;
  const int num_stages = 3;
  const int num_warps = 8;
  const int GROUP_M = 8;

  const int64_t number_blocks_m = cdiv(M, BLOCK_M);
  const int64_t number_blocks_n = cdiv(N, BLOCK_N);

  const int64_t total_tiles = number_blocks_m * number_blocks_n;
  const int64_t iters_per_tile = cdiv(K, BLOCK_K);
  const int64_t tiles_per_wave = sm_count;

  int64_t number_cooperative_tiles = total_tiles % tiles_per_wave;
  int64_t number_other_tiles = total_tiles - number_cooperative_tiles;

  if (number_other_tiles > 0 && number_cooperative_tiles < (int64_t)(sm_count * 0.5)) {
    number_cooperative_tiles += tiles_per_wave;
  } else if (number_other_tiles > 0 && number_cooperative_tiles > (int64_t)(sm_count * 0.8)) {
    number_cooperative_tiles = 0;
  }

  // Prepare Triton kernels
  const auto triton_src = (utils::get_flag_gems_src_path() / "ops" / "mm_streamk.py").string();
  const TritonJITFunction &first_wave = TritonJITFunction::get_instance(triton_src, "first_wave");
  const TritonJITFunction &first_wave_for_bf16 =
      TritonJITFunction::get_instance(triton_src, "first_wave_for_bf16");
  const TritonJITFunction &classic_mm = TritonJITFunction::get_instance(triton_src, "classic_mm");

  // device / stream
  c10::DeviceGuard guard(c.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  if (number_cooperative_tiles > 0) {
    // mini wave handling
    int64_t total_iters_streamk = number_cooperative_tiles * iters_per_tile;
    int64_t iters_per_pid = total_iters_streamk / tiles_per_wave;  // integer division
    int64_t iters_remaining = total_iters_streamk % tiles_per_wave;
    bool even_k = (K % BLOCK_K) == 0;

    if (a.dtype() == at::kBFloat16) {
      // create locks and P (float32)
      auto locks = at::zeros({(int64_t)tiles_per_wave}, a.options().dtype(at::kInt));
      auto P = at::empty({(int64_t)tiles_per_wave, BLOCK_M, BLOCK_N}, a.options().dtype(at::kFloat));

      // call first_wave_for_bf16 kernel: set grid_x = tiles_per_wave
      // The argument order follows the python call:
      // a, b, c, P, M, N, K, locks, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
      // c.stride(0), c.stride(1), iters_per_pid=iters_per_pid, iters_remaining=iters_remaining,
      // iters_per_tile=iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, even_k
      first_wave_for_bf16(
          /* CUstream */ raw_stream,
          /* grid_x */ (int)tiles_per_wave,
          /* grid_y */ 1,
          /* grid_z */ 1,
          /* num_warps */ num_warps,
          /* num_stages */ num_stages,
          a,
          b,
          c,
          P,
          (int64_t)M,
          (int64_t)N,
          (int64_t)K,
          locks,
          (int64_t)a.stride(0),
          (int64_t)a.stride(1),
          (int64_t)b.stride(0),
          (int64_t)b.stride(1),
          (int64_t)c.stride(0),
          (int64_t)c.stride(1),
          (int64_t)iters_per_pid,
          (int64_t)iters_remaining,
          (int64_t)iters_per_tile,
          /* BLOCK_M */ BLOCK_M,
          /* BLOCK_N */ BLOCK_N,
          /* BLOCK_K */ BLOCK_K,
          /* GROUP_M */ GROUP_M,
          even_k);

    } else {
      // non-bf16 path
      auto locks = at::zeros({(int64_t)number_cooperative_tiles}, a.options().dtype(at::kInt));
      // call first_wave kernel: grid_x = tiles_per_wave
      // arg order follows python call (positional):
      // a, b, c, M, N, K, locks, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
      // c.stride(0), c.stride(1), iters_per_pid=iters_per_pid, iters_remaining=iters_remaining,
      // iters_per_tile = iters_per_tile, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, even_k

      first_wave(
          /* CUstream = */ raw_stream,
          /* grid_x = */ (int)tiles_per_wave,
          /* grid_y = */ 1,
          /* grid_z = */ 1,
          /* num_warps = */ num_warps,
          /* num_stages = */ num_stages,
          a,
          b,
          c,
          (int64_t)M,
          (int64_t)N,
          (int64_t)K,
          locks,
          (int64_t)a.stride(0),
          (int64_t)a.stride(1),
          (int64_t)b.stride(0),
          (int64_t)b.stride(1),
          (int64_t)c.stride(0),
          (int64_t)c.stride(1),
          (int64_t)iters_per_pid,
          (int64_t)iters_remaining,
          (int64_t)iters_per_tile,
          BLOCK_M,
          BLOCK_N,
          BLOCK_K,
          GROUP_M,
          even_k);
    }
  }

  // classic_mm for the rest tiles
  int64_t classic_grid = total_tiles - number_cooperative_tiles;
  if (classic_grid > 0) {
    // call classic_mm with grid_x = classic_grid
    // order (a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),
    // total_tiles_streamk=number_cooperative_tiles, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages,
    // num_warps)
    classic_mm(
        /* CUstream = */ raw_stream,
        /* grid_x = */ (int)classic_grid,
        /* grid_y = */ 1,
        /* grid_z = */ 1,
        /* num_warps = */ num_warps,
        /* num_stages = */ num_stages,
        a,
        b,
        c,
        (int64_t)M,
        (int64_t)N,
        (int64_t)K,
        (int64_t)a.stride(0),
        (int64_t)a.stride(1),
        (int64_t)b.stride(0),
        (int64_t)b.stride(1),
        (int64_t)c.stride(0),
        (int64_t)c.stride(1),
        (int64_t)number_cooperative_tiles,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M);
  }
  return;
}

void general_mm_tensor(
    const at::Tensor &a, const at::Tensor &b, at::Tensor &c, int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(a.dtype() == b.dtype(), "expected a and b to have the same dtype");

  const int BLOCK_M = 64;
  const int BLOCK_N = 128;
  const int BLOCK_K = 64;
  const int num_stages = 2;
  const int num_warps = 4;
  const int GROUP_M = 8;

  // general situation
  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "mm.py"),
                                      "mm_kernel_general");

  c10::DeviceGuard guard(c.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  unsigned int grid_x = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    num_warps,
    num_stages,
    a,
    b,
    c,
    M,
    N,
    K,
    a.stride(0),
    a.stride(1),
    b.stride(0),
    b.stride(1),
    c.stride(0),
    c.stride(1),
    /* BLOCK_M = */ BLOCK_M,
    /* BLOCK_N = */ BLOCK_N,
    /* BLOCK_K = */ BLOCK_K,
    /* GROUP_M = */ GROUP_M);
  return;
}

at::Tensor mm_tensor(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected a and b to have the same dtype, but got: ",
              mat1.dtype(),
              " != ",
              mat2.dtype())

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(1);

  at::Tensor out = at::empty({M, N}, mat1.options());

  int sm_count = get_sm_count();

  if (streamk_scenario(mat1, mat2, M, N, K)) {
    streamk_mm_tensor(mat1, mat2, out, M, N, K, sm_count);
    return out;
  } else {
    general_mm_tensor(mat1, mat2, out, M, N, K);
    return out;
  }
}

at::Tensor &mm_out_tensor(const at::Tensor &mat1, const at::Tensor &mat2, at::Tensor &out) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected a and b to have the same dtype, but got: ",
              mat1.dtype(),
              " != ",
              mat2.dtype())

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N = mat2.size(1);

  int sm_count = get_sm_count();

  if (streamk_scenario(mat1, mat2, M, N, K)) {
    streamk_mm_tensor(mat1, mat2, out, M, N, K, sm_count);
    return out;
  } else {
    general_mm_tensor(mat1, mat2, out, M, N, K);
    return out;
  }
}

}  // namespace flag_gems
