#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <cmath>
#include "flag_gems/backend_utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include <filesystem>
#include "ATen/WrapDimUtils.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  // Path to the Python module that contains all fused Triton kernels.
  inline std::string kpath() {
    return utils::get_flag_gems_src_path() / "ops" / "linalg_matrix_norm.py";
  }

  // ============================================================================
  // ord1_norm  — 1-norm (ord=1 / ord=-1).  Mirrors Python _ord1_norm.
  //
  // Uses the unified _abs_norm_kernel (TILED + BATCHED switches).
  //   ord=1  → max_j Σ_i |A[i,j]|
  //   ord=-1 → min_j Σ_i |A[i,j]|
  // ============================================================================

  at::Tensor ord1_norm(const at::Tensor& A, int64_t M, int64_t N, bool is_min) {
    // Tile size selection: keep total blocks ~1K-16K for GPU occupancy.
    int64_t BM, BN;
    const int num_warps = 8;
    if (M <= 1024 && N <= 1024) {
      BM = 32;
      BN = 32;
    } else if (N >= 8 * M || M >= 8 * N) {
      BM = std::min(M, (int64_t)128);
      BN = std::min(N, (int64_t)128);
    } else {
      BM = 128;
      BN = 32;
    }
    // Cap tiles to actual matrix dims (avoids loading masked-out elements).
    // tl.arange requires power-of-2 sizes; ensure BM,BN are pow2 after cap.
    if (BM > M) BM = utils::next_power_of_2(M);
    if (BN > N) BN = utils::next_power_of_2(N);
    int64_t grid_m = utils::cdiv(M, BM);
    int64_t grid_n = utils::cdiv(N, BN);

    c10::DeviceGuard guard(A.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    if (grid_m * grid_n >= 128) {
      // Use matching accumulation dtype: fp64 for fp64 inputs, fp32 otherwise.
      auto acc_dtype = (A.scalar_type() == c10::kDouble) ? c10::kDouble : c10::kFloat;
      auto partial_opts = A.options().dtype(acc_dtype);
      at::Tensor partial = torch::zeros({N}, partial_opts);

      const TritonJITFunction& main_k = TritonJITFunction::get_instance(kpath(), "_abs_norm_kernel");
      int64_t grid_size = grid_m * grid_n;
      int64_t sum_axis = 0;  // SUM_AXIS=0 → 1norm (reduce along rows)
      int64_t is_min_val = is_min ? 1 : 0;
      main_k(rs,
             grid_size,
             1,
             1,
             num_warps,
             2,
             A,
             partial,
             partial,
             M,
             N,
             BM,
             BN,
             grid_n,
             sum_axis,
             is_min_val,
             /*TILED=*/true,
             /*BATCHED=*/false);
      // Host-side reduction (partial.max / partial.min)
      return is_min ? partial.min() : partial.max();
    }

    // Small/medium matrices: single-launch fused kernel.
    // Use fp32 output buffer — tl.atomic_max / tl.atomic_min don't support
    // fp16/bf16.  Convert back to the original dtype after the kernel.
    auto tmp_opts = A.options().dtype(c10::kFloat);
    at::Tensor out = is_min ? torch::full({}, INFINITY, tmp_opts) : torch::zeros({}, tmp_opts);

    const TritonJITFunction& main_k = TritonJITFunction::get_instance(kpath(), "_abs_norm_kernel");

    int64_t sum_axis = 0;  // SUM_AXIS=0 → 1norm
    int64_t is_min_val = is_min ? 1 : 0;
    auto dummy = torch::empty({1}, tmp_opts);
    main_k(rs,
           grid_n,
           1,
           1,
           num_warps,
           2,
           A,
           out,
           dummy,
           M,
           N,
           BM,
           BN,
           1,
           sum_axis,
           is_min_val,
           /*TILED=*/false,
           /*BATCHED=*/false);
    if (out.scalar_type() != A.scalar_type()) out = out.to(A.scalar_type());
    return out;
  }

  // ============================================================================
  // ordinf_norm  — infinity-norm (ord=inf / ord=-inf).  Mirrors Python _ordinf_norm.
  //
  // Uses the unified _abs_norm_kernel (TILED + BATCHED switches).
  //   ord=inf  → max_i Σ_j |A[i,j]|
  //   ord=-inf → min_i Σ_j |A[i,j]|
  // ============================================================================

  at::Tensor ordinf_norm(const at::Tensor& A, int64_t M, int64_t N, bool is_min) {
    int64_t BM, BN;
    if (M <= 1024 && N <= 1024) {
      BM = 32;
      BN = 32;
    } else if (N >= 8 * M || M >= 8 * N) {
      BM = std::min(M, (int64_t)128);
      BN = std::min(N, (int64_t)128);
    } else {
      BM = 128;
      BN = 32;
    }
    if (BM > M) BM = utils::next_power_of_2(M);
    if (BN > N) BN = utils::next_power_of_2(N);
    int64_t grid_m = utils::cdiv(M, BM);
    int64_t grid_n = utils::cdiv(N, BN);
    const int num_warps = 8;

    c10::DeviceGuard guard(A.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    if (grid_m * grid_n >= 512) {
      auto acc_dtype = (A.scalar_type() == c10::kDouble) ? c10::kDouble : c10::kFloat;
      auto partial_opts = A.options().dtype(acc_dtype);
      at::Tensor partial = torch::zeros({M}, partial_opts);

      const TritonJITFunction& main_k = TritonJITFunction::get_instance(kpath(), "_abs_norm_kernel");
      int64_t grid_size = grid_m * grid_n;
      int64_t sum_axis = 1;  // SUM_AXIS=1 → infnorm (reduce along cols)
      int64_t is_min_val = is_min ? 1 : 0;
      main_k(rs,
             grid_size,
             1,
             1,
             num_warps,
             2,
             A,
             partial,
             partial,
             M,
             N,
             BM,
             BN,
             grid_n,
             sum_axis,
             is_min_val,
             /*TILED=*/true,
             /*BATCHED=*/false);
      // Host-side reduction (partial.max / partial.min)
      return is_min ? partial.min() : partial.max();
    }

    // Small matrices: single-launch fused kernel with atomic output.
    auto tmp_opts = A.options().dtype(c10::kFloat);
    at::Tensor out = is_min ? torch::full({}, INFINITY, tmp_opts) : torch::zeros({}, tmp_opts);

    const TritonJITFunction& main_k = TritonJITFunction::get_instance(kpath(), "_abs_norm_kernel");

    int64_t sum_axis = 1;  // SUM_AXIS=1 → infnorm (reduce along cols)
    int64_t is_min_val = is_min ? 1 : 0;
    auto dummy = torch::empty({1}, tmp_opts);
    main_k(rs,
           grid_m,
           1,
           1,
           num_warps,
           2,
           A,
           out,
           dummy,
           M,
           N,
           BM,
           BN,
           1,
           sum_axis,
           is_min_val,
           /*TILED=*/false,
           /*BATCHED=*/false);
    if (out.scalar_type() != A.scalar_type()) out = out.to(A.scalar_type());
    return out;
  }

  // ============================================================================
  // fro_norm  — Frobenius norm (ord="fro").  Mirrors Python _fro_norm.
  //
  // Uses the unified _fro_kernel (TILE_2D switch for large single matrices).
  // ============================================================================

  at::Tensor fro_norm(const at::Tensor& A, int64_t M, int64_t N) {
    int64_t BM, BN;
    if (M <= 1024 && N <= 1024) {
      BM = 32;
      BN = 32;
    } else if (N >= 8 * M || M >= 8 * N) {
      BM = std::min(M, (int64_t)128);
      BN = std::min(N, (int64_t)128);
    } else {
      BM = 128;
      BN = 32;
    }
    if (BM > M) BM = utils::next_power_of_2(M);
    if (BN > N) BN = utils::next_power_of_2(N);
    int64_t grid_m = utils::cdiv(M, BM);
    int64_t grid_n = utils::cdiv(N, BN);
    int64_t grid_size = grid_m * grid_n;
    const int num_warps = 8;
    // Accumulate in fp64 for the tiled atomic_add path so that
    // both the per-tile sum and the cross-tile reduction use fp64.
    // The _fro_kernel internally uses fp64 when TILE_2D=True.
    auto tmp_opts = A.options().dtype(c10::kDouble);
    at::Tensor out = torch::zeros({}, tmp_opts);

    c10::DeviceGuard guard(A.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    const TritonJITFunction& main_k = TritonJITFunction::get_instance(kpath(), "_fro_kernel");
    main_k(rs,
           grid_size,
           1,
           1,
           num_warps,
           2,
           A,
           out,
           M,
           N,
           BM,
           BN,
           grid_n,
           /*TILE_2D=*/true);

    out = torch::sqrt(out);
    if (out.scalar_type() != A.scalar_type()) out = out.to(A.scalar_type());
    return out;
  }

  // ============================================================================
  // svdvals_rank2  — closed-form singular values for k=2.
  // Mirrors Python _svdvals_rank2.  Uses _rank2_svals_kernel.
  // ============================================================================

  at::Tensor svdvals_rank2(const at::Tensor& A, int64_t M, int64_t N) {
    int64_t batch = A.size(0);
    int64_t largest = std::max(M, N);
    bool tall = M >= N;
    int64_t block_r = utils::next_power_of_2(largest);

    auto S = torch::empty({batch, 2}, A.options());
    c10::DeviceGuard guard(A.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    // Tiny matrices with many batches: process multiple per program.
    if (largest <= 16 && batch >= 16) {
      int64_t block_b;
      if (largest <= 2)
        block_b = 2;
      else if (largest == 16)
        block_b = 16;
      else if (tall)
        block_b = 2;
      else
        block_b = 8;

      const TritonJITFunction& k = TritonJITFunction::get_instance(kpath(), "_rank2_svals_kernel");
      int64_t grid = utils::cdiv(batch, block_b);
      k(rs, grid, 1, 1, 1, 2, A, S, batch, M, N, tall, block_b, block_r);
    } else {
      const TritonJITFunction& k = TritonJITFunction::get_instance(kpath(), "_rank2_svals_kernel");
      int num_warps = (block_r <= 64) ? 1 : 4;
      // BLOCK_B=1: single matrix per program (regular path).
      k(rs, batch, 1, 1, num_warps, 2, A, S, batch, M, N, tall, 1, block_r);
    }
    return S;
  }

  // ============================================================================
  // svdvals_hybrid  — hybrid SVD: Jacobi on R, DBDSQR for k=3.
  // Mirrors Python _svdvals_hybrid.
  //   1. fp64 QR → triangular R (k×k)
  //   2. k≥4: Parallel Brent-Luk Jacobi on R.
  //      fp64: 60/50 sweeps,  fp32: 30/40 sweeps.
  //   3. k=3: QR + bidiag + DBDSQR
  // ============================================================================

  at::Tensor svdvals_hybrid(
      const at::Tensor& A_batch, int64_t batch, int64_t M, int64_t N, at::ScalarType out_dtype) {
    int64_t k = std::min(M, N);
    bool use_tall = M >= N;  // QR kernel requires M ≥ N

    c10::DeviceGuard guard(A_batch.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    // === Compute dtype selection ==============================================
    // Match PyTorch CUDA gesvdj precision:
    //   fp64 → fp64,  fp32 → fp32,  fp16/bf16 → upcast to fp32.
    // ==========================================================================
    auto in_dtype = A_batch.scalar_type();
    at::Tensor A_compute = A_batch;
    if (in_dtype == at::kHalf || in_dtype == at::kBFloat16) {
      A_compute = A_batch.to(at::kFloat);
      in_dtype = at::kFloat;
    }
    // Always fp64 for SVD compute to match Jacobi path precision.
    auto compute_dtype = at::kDouble;
    double dbdsqr_eps = 2.220446049250313e-16;  // fp64 machine epsilon

    // === Step 1: QR → triangular R (k×k) ======================================
    at::Tensor A_qr;
    int64_t M_qr, N_qr;
    if (use_tall) {
      // Always clone: the QR kernel overwrites A_qr in place with Householder
      // vectors.  Use fp64 for QR accuracy (qr_use_fp64 is always true).
      A_qr = A_compute.to(at::kDouble).clone();
      M_qr = M;
      N_qr = N;
    } else {
      A_qr = A_compute.transpose(-2, -1).contiguous().to(at::kDouble);
      M_qr = N;
      N_qr = M;
    }

    int64_t qr_block_m = utils::next_power_of_2(std::min(M_qr, (int64_t)256));
    int64_t qr_block_n = 32;

    auto Rf = torch::zeros({batch, k, k}, A_qr.options());

    const TritonJITFunction& qr_k = TritonJITFunction::get_instance(kpath(), "_householder_qr_r_kernel");
    qr_k(rs,
         batch,
         1,
         1,
         4,
         2,
         A_qr,
         Rf,
         M_qr,
         N_qr,
         k,
         A_qr.stride(0),
         A_qr.stride(1),
         A_qr.stride(2),
         qr_block_m,
         qr_block_n,
         /*USE_FP64=*/true);

    // === Step 2: Jacobi SVD on R (k≥4) ==========================================
    // Sweeps tuned by dtype:
    //   fp64: 60/50 sweeps — drowns out ULPs-level tl.sum/tl.sqrt differences
    //         across GPU SM versions so final column norms are identical.
    //   fp32: 30/40 sweeps — precision floor is higher, fewer sweeps suffice.
    bool use_jacobi = (k >= 4);

    if (use_jacobi) {
      int64_t JACOBI_SWEEPS;
      if (in_dtype == at::kDouble)
        JACOBI_SWEEPS = (k <= 48) ? 60 : 50;
      else
        JACOBI_SWEEPS = (k <= 48) ? 30 : 40;
      int64_t block_r_j = utils::next_power_of_2(k);

      auto a_work = Rf.transpose(1, 2).contiguous();  // column-major for Jacobi

      const TritonJITFunction& jacobi_step_k =
          TritonJITFunction::get_instance(kpath(), "_parallel_jacobi_step_kernel");
      int num_warps_j = (block_r_j <= 64) ? 1 : 4;

      // Brent-Luk ordering: K-1 steps per sweep, K/2 independent pairs/step.
      for (int64_t sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
        for (int64_t step = 0; step < k - 1; step++) {
          jacobi_step_k(rs, batch, k / 2, 1, num_warps_j, 2, a_work, k, k, step, block_r_j);
        }
      }

      // Column norms = singular values.  We use norm(-1) instead of
      // bmm + diagonal because bmm (cuBLAS) is non-deterministic across
      // GPU architectures — ULPs difference can cause spurious DBDSQR
      // fallback on some devices (~3% of calls).
      auto col_norms = a_work.norm(-1);
      auto S = col_norms.to(out_dtype);
      S = std::get<0>(S.sort(-1, /*descending=*/true));
      return S;
    }

    // === Step 3: DBDSQR — QR + bidiag + Golub-Kahan (k=3 only) ===
    at::Tensor A_qr_fb;
    int64_t M_qr2, N_qr2;
    if (use_tall) {
      A_qr_fb = A_compute.to(compute_dtype).clone();
      M_qr2 = M;
      N_qr2 = N;
    } else {
      A_qr_fb = A_compute.transpose(-2, -1).contiguous().to(compute_dtype);
      M_qr2 = N;
      N_qr2 = M;
    }

    int64_t qr_block_m2 = utils::next_power_of_2(std::min(M_qr2, (int64_t)256));

    auto R_fb = torch::zeros({batch, k, k}, A_qr_fb.options());
    qr_k(rs,
         batch,
         1,
         1,
         4,
         2,
         A_qr_fb,
         R_fb,
         M_qr2,
         N_qr2,
         k,
         A_qr_fb.stride(0),
         A_qr_fb.stride(1),
         A_qr_fb.stride(2),
         qr_block_m2,
         qr_block_n,
         /*USE_FP64=*/true);

    // Bidiagonalization — dtype flows from R_fb.
    int64_t bidiag_block_k = utils::next_power_of_2(k);
    auto d_fb = torch::zeros({batch, k}, R_fb.options());
    auto e_fb = torch::zeros({batch, k - 1}, R_fb.options());

    const TritonJITFunction& bidiag_k = TritonJITFunction::get_instance(kpath(), "_bidiag_kernel");
    bidiag_k(rs, batch, 1, 1, (bidiag_block_k <= 64 ? 1 : 4), 2, R_fb, d_fb, e_fb, k, bidiag_block_k);

    // Fused DBDSQR — adaptive parameters by k.
    int64_t dbdsqr_block_k = utils::next_power_of_2(k);
    int64_t max_iters = (k <= 32) ? 30 : (k <= 64) ? 50 : (k <= 128) ? 100 : 200;
    int64_t num_w = (k <= 64) ? 1 : 4;
    const TritonJITFunction& dbdsqr_k = TritonJITFunction::get_instance(kpath(), "_fused_dbdsqr_kernel");
    dbdsqr_k(rs,
             batch,
             1,
             1,
             num_w,
             2,
             d_fb,
             e_fb,
             k,
             dbdsqr_block_k,
             /*EPS=*/dbdsqr_eps,
             /*MAX_ITERS=*/max_iters,
             /*BLOCK_SWEEPS=*/50);

    auto S = d_fb.abs().to(out_dtype);
    S = std::get<0>(S.sort(-1, /*descending=*/true));
    return S;
  }

  // ============================================================================
  // nuc_norm  — nuclear norm (ord="nuc").  Mirrors Python _nuc_norm.
  // ============================================================================

  at::Tensor nuc_norm(
      const at::Tensor& A, int64_t d0, int64_t d1, bool keepdim, std::optional<at::ScalarType> dtype) {
    at::ScalarType out_dtype = dtype.has_value() ? dtype.value() : A.scalar_type();

    // Permute so that d0, d1 are the last two dimensions.
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < A.dim(); i++)
      if (i != d0 && i != d1) perm.push_back(i);
    perm.push_back(d0);
    perm.push_back(d1);
    auto A_perm = A.permute(perm);

    // Compute batch = product of all non-matrix dimensions.
    int64_t batch = 1;
    for (int64_t i = 0; i < A_perm.dim() - 2; i++) batch *= A_perm.size(i);

    int64_t M = A_perm.size(-2);
    int64_t N = A_perm.size(-1);
    int64_t k = std::min(M, N);
    int64_t rows = std::max(M, N);
    bool tall = M >= N;

    // Reshape to (batch, M, N) — one matrix per SVD kernel program.
    auto A_batch = A_perm.reshape({batch, M, N}).contiguous();

    // fp16/bf16: upcast to fp32 for SVD compute (matching PyTorch CUDA).
    if (A_batch.scalar_type() == at::kHalf || A_batch.scalar_type() == at::kBFloat16) {
      A_batch = A_batch.to(at::kFloat);
    }

    at::Tensor S;
    c10::DeviceGuard guard(A.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);

    // Dispatch: k=1→L2, k=2→rank2, k>2→svdvals_hybrid (matches _svdvals_for_norm)
    if (k == 1) {
      // Single singular value = Frobenius norm via _fro_kernel
      int64_t total_k1 = M * N;
      auto flat_k1 = A_batch.reshape({batch, total_k1});
      S = torch::empty({batch, 1}, flat_k1.options().dtype(c10::kFloat));
      const TritonJITFunction& fro_k1 = TritonJITFunction::get_instance(kpath(), "_fro_kernel");
      int64_t blk_n = utils::next_power_of_2(std::min(total_k1, (int64_t)512));
      fro_k1(rs,
             batch,
             1,
             1,
             8,
             2,
             flat_k1,
             S,
             0,
             total_k1,
             1,
             blk_n,
             1,
             /*TILE_2D=*/false);
    } else if (k == 2 && rows <= 2048) {
      S = svdvals_rank2(A_batch, M, N);
    } else if (k <= 512 && rows <= 2048) {
      S = svdvals_hybrid(A_batch, batch, M, N, out_dtype);
    } else {
      return at::Tensor();  // Unsupported → Python fallback.
    }

    // Check if SVD succeeded (non-empty tensor); fall back to Python if not.
    if (!S.defined() || S.numel() == 0) {
      return at::Tensor();
    }

    // S has shape (batch, k).  Nuclear norm = Σ_k σ_k → (batch,).
    auto result = S.sum(-1);

    if (result.scalar_type() != out_dtype) result = result.to(out_dtype);

    if (keepdim) {
      std::vector<int64_t> out_shape = A.sizes().vec();
      out_shape[d0] = 1;
      out_shape[d1] = 1;
      result = result.reshape(out_shape);
    } else {
      std::vector<int64_t> batch_shape;
      for (int64_t i = 0; i < A.dim(); i++)
        if (i != d0 && i != d1) batch_shape.push_back(A.size(i));
      result = result.reshape(batch_shape);
    }
    return result;
  }

  // ============================================================================
  // flatten_to_2d  — permute + reshape so that d0,d1 become the last two dims
  //                   and everything else is folded into the leading dimension.
  //
  // Returns (A2d, M, N) where A2d has shape (M, N).
  //   M = product of all batch dims
  //   N = size(d0) × size(d1)
  //
  // WARNING: only correct for Frobenius norm (element-wise L2 is invariant
  // under merging batch dims).  Other ords must call with simple 2D matrices.
  // ============================================================================

  static std::tuple<at::Tensor, int64_t, int64_t> flatten_to_2d(const at::Tensor& A, int64_t d0, int64_t d1) {
    if (A.dim() == 2 && d0 == 0 && d1 == 1) return {A, A.size(0), A.size(1)};

    // Build permutation: all dims except d0,d1 come first, then d0, d1 last.
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < A.dim(); i++)
      if (i != d0 && i != d1) perm.push_back(i);
    perm.push_back(d0);
    perm.push_back(d1);

    auto A2d = A.permute(perm);
    int64_t M = 1;
    for (size_t i = 0; i < perm.size() - 2; i++) M *= A2d.size(i);
    int64_t N = A2d.size(perm.size() - 2) * A2d.size(perm.size() - 1);
    return {A2d.reshape({M, N}), M, N};
  }

}  // anonymous namespace

// ============================================================================
// Public entry points  — registered via TORCH_LIBRARY_IMPL(aten, …)
// ============================================================================

// ----------------------------------------------------------------------------
// linalg_matrix_norm_str  — string-valued ord ("fro", "nuc")
// ----------------------------------------------------------------------------

at::Tensor linalg_matrix_norm_str(const at::Tensor& A,
                                  c10::string_view ord,
                                  at::IntArrayRef dim,
                                  bool keepdim,
                                  std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(A.dim() >= 2, "linalg.matrix_norm: A must be at least 2-D");
  TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple");
  int64_t d0 = at::maybe_wrap_dim(dim[0], A.dim());
  int64_t d1 = at::maybe_wrap_dim(dim[1], A.dim());
  TORCH_CHECK(d0 != d1, "linalg.matrix_norm: dims must be different");

  // --- "fro": Frobenius norm -------------------------------------------
  if (ord == "fro") {
    at::ScalarType out_dtype = dtype.has_value() ? dtype.value() : A.scalar_type();

    // Batched: permute to (batch, mat_size), per-row L2 via _fro_kernel
    if (A.dim() > 2) {
      // Permute so target dims are the last two.
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < A.dim(); i++)
        if (i != d0 && i != d1) perm.push_back(i);
      perm.push_back(d0);
      perm.push_back(d1);
      auto A_perm = A.permute(perm);

      int64_t batch = 1;
      for (int64_t i = 0; i < A_perm.dim() - 2; i++) batch *= A_perm.size(i);
      int64_t mat_size = A_perm.size(-2) * A_perm.size(-1);
      auto flat = A_perm.reshape({batch, mat_size}).contiguous();

      auto result = torch::empty({batch}, flat.options().dtype(out_dtype));
      c10::DeviceGuard guard(flat.device());
      backend::StreamType stream = backend::getCurrentStream();
      backend::RawStreamType rs = backend::getRawStream(stream);

      const TritonJITFunction& k = TritonJITFunction::get_instance(kpath(), "_fro_kernel");
      int64_t blk_n = utils::next_power_of_2(std::min(mat_size, (int64_t)512));
      // num_warps must match every other _fro_kernel launch — the
      // TritonJIT overload cache is keyed by signature (not num_warps),
      // so a mismatched num_warps reuses the wrong compiled kernel and
      // the launch fails with "invalid argument".
      k(rs,
        batch,
        1,
        1,
        8,
        2,
        flat,
        result,
        0,
        mat_size,
        1,
        blk_n,
        1,
        /*TILE_2D=*/false);

      if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
      if (keepdim) {
        auto out_shape = A.sizes().vec();
        out_shape[d0] = 1;
        out_shape[d1] = 1;
        result = result.reshape(out_shape);
      } else {
        // Reshape from flat (batch_size,) to original batch dims.
        std::vector<int64_t> batch_shape;
        for (int64_t i = 0; i < A.dim(); i++)
          if (i != d0 && i != d1) batch_shape.push_back(A.size(i));
        result = result.reshape(batch_shape);
      }
      return result;
    }

    // Simple 2D path — flatten to 2D and use existing kernels.
    auto [A2d, M, N] = flatten_to_2d(A, d0, d1);
    int64_t total = M * N;

    if (total <= 65536) {
      // 2D flat array → call _fro_kernel with TILE_2D=false, batch=1.
      at::Tensor flat = A2d.reshape({1, total});
      at::Tensor result_vec = torch::empty({1}, A2d.options().dtype(out_dtype));
      c10::DeviceGuard guard(A2d.device());
      backend::StreamType stream = backend::getCurrentStream();
      backend::RawStreamType rs = backend::getRawStream(stream);
      const TritonJITFunction& k = TritonJITFunction::get_instance(kpath(), "_fro_kernel");
      // num_warps=8: must match all other _fro_kernel launches (TritonJIT
      // overload cache is keyed by signature, not num_warps).
      k(rs,
        1,
        1,
        1,
        8,
        2,
        flat,
        result_vec,
        0,
        total,
        1,
        512,
        1,
        /*TILE_2D=*/false);
      at::Tensor result = result_vec.squeeze(0);  // (1,) → scalar
      if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
      if (keepdim) result = result.reshape(std::vector<int64_t>(A.dim(), 1));
      return result;
    }

    at::Tensor result = fro_norm(A2d, M, N);
    if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
    if (keepdim) result = result.reshape(std::vector<int64_t>(A.dim(), 1));
    return result;
  }

  // --- "nuc": nuclear norm (sum of singular values) --------------------
  if (ord == "nuc") {
    return nuc_norm(A, d0, d1, keepdim, dtype);
  }

  // Not handled — Python fallback.
  return at::Tensor();
}

// ----------------------------------------------------------------------------
// linalg_matrix_norm  — numeric ord (1, -1, 2, -2, inf, -inf)
// ----------------------------------------------------------------------------

at::Tensor linalg_matrix_norm(const at::Tensor& A,
                              const c10::Scalar& ord,
                              at::IntArrayRef dim,
                              bool keepdim,
                              std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(A.dim() >= 2, "linalg.matrix_norm: A must be at least 2-D");
  TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple");
  int64_t d0 = at::maybe_wrap_dim(dim[0], A.dim());
  int64_t d1 = at::maybe_wrap_dim(dim[1], A.dim());
  TORCH_CHECK(d0 != d1, "linalg.matrix_norm: dims must be different");

  double ord_val = ord.toDouble();
  double abs_ord = std::abs(ord_val);

  at::ScalarType out_dtype = dtype.has_value() ? dtype.value() : A.scalar_type();

  // Batched path for ord=1/inf: keep (batch, M, N) for proper per-matrix norm.
  if (A.dim() > 2 && (abs_ord == 1.0 || std::isinf(abs_ord))) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < A.dim(); i++)
      if (i != d0 && i != d1) perm.push_back(i);
    perm.push_back(d0);
    perm.push_back(d1);
    auto A_perm = A.permute(perm);
    int64_t batch = 1;
    for (int64_t i = 0; i < A_perm.dim() - 2; i++) batch *= A_perm.size(i);
    int64_t mat_M = A_perm.size(-2), mat_N = A_perm.size(-1);
    auto Ab = A_perm.reshape({batch, mat_M, mat_N}).contiguous();
    // fp32 buffer for atomic ops; convert to out_dtype after kernels.
    auto result = torch::empty({batch}, Ab.options().dtype(c10::kFloat));
    c10::DeviceGuard guard(Ab.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType rs = backend::getRawStream(stream);
    int64_t blk_n = utils::next_power_of_2(std::min(mat_N, (int64_t)256));
    bool is_min = (ord_val < 0);
    int64_t sum_axis, is_min_val = is_min ? 1 : 0;
    const TritonJITFunction& abs_k = TritonJITFunction::get_instance(kpath(), "_abs_norm_kernel");
    auto dummy = torch::empty({1}, Ab.options().dtype(c10::kFloat));
    if (std::isinf(abs_ord)) {
      // inf/-inf: row-parallel, grid=(batch, grid_m).
      int64_t tile_m = 16;
      int64_t grid_dim = utils::cdiv(mat_M, tile_m);
      auto init_val = is_min ? INFINITY : 0.0;
      result = torch::full({batch}, init_val, Ab.options().dtype(c10::kFloat));
      sum_axis = 1;  // SUM_AXIS=1 → infnorm
      abs_k(rs,
            batch,
            grid_dim,
            1,
            8,
            2,
            Ab,
            result,
            dummy,
            mat_M,
            mat_N,
            tile_m,
            blk_n,
            1,
            sum_axis,
            is_min_val,
            /*TILED=*/false,
            /*BATCHED=*/true);
    } else {
      // 1/-1: column-parallel, grid=(batch, grid_n).
      int64_t tile_n_raw = std::min(mat_N, (int64_t)128);
      int64_t tile_n = utils::next_power_of_2(tile_n_raw);
      int64_t grid_dim = utils::cdiv(mat_N, tile_n_raw);
      int64_t blk_m = utils::next_power_of_2(std::min(mat_M, (int64_t)32));
      auto init_val = is_min ? INFINITY : 0.0;
      result = torch::full({batch}, init_val, Ab.options().dtype(c10::kFloat));
      sum_axis = 0;  // SUM_AXIS=0 → 1norm
      abs_k(rs,
            batch,
            grid_dim,
            1,
            8,
            2,
            Ab,
            result,
            dummy,
            mat_M,
            mat_N,
            blk_m,
            tile_n,
            1,
            sum_axis,
            is_min_val,
            /*TILED=*/false,
            /*BATCHED=*/true);
    }
    if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
    if (keepdim) {
      auto out_shape = A.sizes().vec();
      out_shape[d0] = 1;
      out_shape[d1] = 1;
      result = result.reshape(out_shape);
    } else {
      std::vector<int64_t> batch_shape;
      for (int64_t i = 0; i < A.dim(); i++)
        if (i != d0 && i != d1) batch_shape.push_back(A.size(i));
      result = result.reshape(batch_shape);
    }
    return result;
  }

  // Batched SVD path for ord=2/-2: permute dims, compute per-matrix.
  if (A.dim() > 2 && abs_ord == 2.0) {
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < A.dim(); i++)
      if (i != d0 && i != d1) perm.push_back(i);
    perm.push_back(d0);
    perm.push_back(d1);
    auto A_perm = A.permute(perm);
    int64_t batch = 1;
    for (int64_t i = 0; i < A_perm.dim() - 2; i++) batch *= A_perm.size(i);
    int64_t mM = A_perm.size(-2), mN = A_perm.size(-1);
    auto Ab = A_perm.reshape({batch, mM, mN}).contiguous();
    int64_t k = std::min(mM, mN), rows = std::max(mM, mN);

    // fp16/bf16: upcast to fp32 for SVD compute (matching PyTorch CUDA).
    if (Ab.scalar_type() == at::kHalf || Ab.scalar_type() == at::kBFloat16) {
      Ab = Ab.to(at::kFloat);
    }

    c10::DeviceGuard guard_svd(Ab.device());
    backend::StreamType stream_svd = backend::getCurrentStream();
    backend::RawStreamType rs_svd = backend::getRawStream(stream_svd);

    at::Tensor S;
    if (k == 1 && rows <= 2048) {
      int64_t total_k1 = mM * mN;
      auto flat_k1 = Ab.reshape({batch, total_k1});
      S = torch::empty({batch, 1}, flat_k1.options().dtype(c10::kFloat));
      const TritonJITFunction& fro_k1 = TritonJITFunction::get_instance(kpath(), "_fro_kernel");
      int64_t blk_n = utils::next_power_of_2(std::min(total_k1, (int64_t)512));
      fro_k1(rs_svd,
             batch,
             1,
             1,
             8,
             2,
             flat_k1,
             S,
             0,
             total_k1,
             1,
             blk_n,
             1,
             /*TILE_2D=*/false);
    } else if (k == 2 && rows <= 2048) {
      S = svdvals_rank2(Ab, mM, mN);  // (batch, 2)
    } else if (k <= 512 && rows <= 2048) {
      S = svdvals_hybrid(Ab, batch, mM, mN, out_dtype);
    } else {
      return at::Tensor();
    }

    auto result = (ord_val > 0) ? std::get<0>(S.max(-1)) : std::get<0>(S.min(-1));
    if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
    if (keepdim) {
      auto out_shape = A.sizes().vec();
      out_shape[d0] = 1;
      out_shape[d1] = 1;
      result = result.reshape(out_shape);
    } else {
      std::vector<int64_t> batch_shape_vec;
      for (int64_t i = 0; i < A.dim(); i++)
        if (i != d0 && i != d1) batch_shape_vec.push_back(A.size(i));
      result = result.reshape(batch_shape_vec);
    }
    return result;
  }

  auto [A2d, M, N] = flatten_to_2d(A, d0, d1);
  at::Tensor result;

  // --- ord=2 / ord=-2: spectral norm (max / min singular value) ----------
  if (abs_ord == 2.0) {
    // Dispatch: k=1→L2, k=2→rank2, k>2→svdvals_hybrid
    int64_t M_val = A2d.size(0);
    int64_t N_val = A2d.size(1);
    int64_t k = std::min(M_val, N_val);
    int64_t rows = std::max(M_val, N_val);

    // fp16/bf16: upcast to fp32 for SVD compute (matching PyTorch CUDA).
    auto A2d_svd =
        (A2d.scalar_type() == at::kHalf || A2d.scalar_type() == at::kBFloat16) ? A2d.to(at::kFloat) : A2d;

    at::Tensor S;
    if (k == 1 && rows <= 2048) {
      // k=1: single singular value = Frobenius norm of the matrix.
      // Delegate to fro_norm which handles large/small via _fro_kernel.
      auto s_val = fro_norm(A2d_svd, M_val, N_val);
      S = s_val.reshape({1, 1}).to(out_dtype);
    } else if (k == 2 && rows <= 2048) {
      auto A_batch = A2d_svd.reshape({1, M_val, N_val});
      S = svdvals_rank2(A_batch, M_val, N_val);
    } else if (k <= 512 && rows <= 2048) {
      auto A_batch_2d = A2d_svd.reshape({1, M_val, N_val});
      S = svdvals_hybrid(A_batch_2d, 1, M_val, N_val, A2d_svd.scalar_type());
    } else {
      return at::Tensor();
    }

    if (ord_val > 0)
      result = S.max();  // ord=2
    else
      result = S.min();  // ord=-2
  }
  // --- ord=1 / ord=-1: max/min absolute column sum -----------------------
  // --- ord=inf / ord=-inf: max/min absolute row sum ----------------------
  else if (abs_ord == 1.0 || std::isinf(abs_ord)) {
    bool is_min = (ord_val < 0);
    if (std::isinf(abs_ord))
      result = ordinf_norm(A2d, M, N, is_min);  // ord=±inf
    else
      result = ord1_norm(A2d, M, N, is_min);  // ord=±1
  } else {
    TORCH_CHECK(false, "C++ wrapper: unsupported ord");
  }

  if (result.scalar_type() != out_dtype) result = result.to(out_dtype);
  if (keepdim) result = result.reshape(std::vector<int64_t>(A.dim(), 1));
  return result;
}

}  // namespace flag_gems
