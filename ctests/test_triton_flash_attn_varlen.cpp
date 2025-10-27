#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>
#include "flag_gems/operators.h"
#include "torch/torch.h"

at::Tensor ref_paged_attn_cpp(const at::Tensor& query,
                              const at::Tensor& key_cache,
                              const at::Tensor& value_cache,
                              const std::vector<int64_t>& query_lens,
                              const std::vector<int64_t>& kv_lens,
                              const at::Tensor& block_tables,
                              double scale,
                              std::optional<at::Tensor> attn_bias_opt,
                              std::optional<int64_t> sliding_window_opt,
                              std::optional<double> soft_cap_opt) {
  TORCH_CHECK(query.device() == key_cache.device() && query.device() == value_cache.device(),
              "All tensors must be on the same device");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2-D");

  const auto device = query.device();
  const auto dtype = query.dtype();  // query dtype (float16/bfloat16)
  const auto v_dtype = value_cache.dtype();

  int64_t num_seqs = static_cast<int64_t>(query_lens.size());
  int64_t start_idx = 0;

  int64_t num_blocks = key_cache.size(0);
  int64_t block_size = key_cache.size(1);
  int64_t num_kv_heads = key_cache.size(2);
  int64_t head_size = key_cache.size(3);

  std::vector<at::Tensor> outputs;
  outputs.reserve(num_seqs);

  for (int64_t i = 0; i < num_seqs; ++i) {
    int64_t query_len = query_lens[i];
    int64_t kv_len = kv_lens[i];

    at::Tensor q = query.narrow(0, start_idx, query_len).clone();
    // q *= scale
    q.mul_(scale);

    // number of kv blocks and indices
    int64_t num_kv_blocks = (kv_len + block_size - 1) / block_size;

    at::Tensor block_idx_row = block_tables.index({i});  // shape (max_num_blocks_per_seq,)
    at::Tensor block_indices = block_idx_row.narrow(0, 0, num_kv_blocks).to(torch::kLong);

    at::Tensor k_sel = key_cache.index_select(0, block_indices).contiguous();
    k_sel = k_sel.view({-1, num_kv_heads, head_size});
    if (k_sel.size(0) > kv_len) {
      k_sel = k_sel.narrow(0, 0, kv_len);
    }

    at::Tensor v_sel = value_cache.index_select(0, block_indices).contiguous();
    v_sel = v_sel.view({-1, num_kv_heads, head_size});
    if (v_sel.size(0) > kv_len) {
      v_sel = v_sel.narrow(0, 0, kv_len);
    }

    int64_t q_heads = q.size(1);
    int64_t k_heads = k_sel.size(1);
    if (q_heads != k_heads) {
      TORCH_CHECK(q_heads % k_heads == 0, "Number of query heads must be a multiple of number of kv heads");
      int64_t repeats = q_heads / k_heads;
      k_sel = at::repeat_interleave(k_sel, repeats, /*dim=*/1);
      v_sel = at::repeat_interleave(v_sel, repeats, /*dim=*/1);
      k_heads = k_sel.size(1);
    }

    at::Tensor q_per = q.permute({1, 0, 2}).contiguous();
    at::Tensor k_per = k_sel.permute({1, 0, 2}).contiguous();

    at::Tensor attn = at::bmm(q_per, k_per.transpose(1, 2));

    at::Tensor empty_mask = at::ones({query_len, kv_len}, query.options().dtype(torch::kUInt8)).to(device);
    int64_t diag = kv_len - query_len + 1;
    at::Tensor mask = at::triu(empty_mask, diag).to(at::kBool);

    // sliding window: compute sliding mask and OR it
    if (sliding_window_opt.has_value()) {
      int64_t sliding = sliding_window_opt.value();
      int64_t diag_sw = kv_len - (query_len + sliding) + 1;
      at::Tensor sw_mask = at::triu(empty_mask, diag_sw).to(at::kBool);
      // invert (logical_not)
      sw_mask = sw_mask.logical_not();
      mask = at::logical_or(mask, sw_mask);
    }

    if (soft_cap_opt.has_value()) {
      double soft_cap = soft_cap_opt.value();
      at::Tensor attn_fp32 = attn.to(at::kFloat);
      attn_fp32 = soft_cap * at::tanh(attn_fp32 / static_cast<float>(soft_cap));
      attn = attn_fp32.to(attn.dtype());
    }

    at::Tensor mask_b = mask.unsqueeze(0);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    attn.masked_fill_(mask_b, neg_inf);

    if (attn_bias_opt.has_value()) {
      at::Tensor bias_i = attn_bias_opt.value().index({i});
      if (bias_i.size(2) > kv_len) {
        bias_i = bias_i.narrow(2, 0, kv_len);
      }
      // Adjust q dim: broadcast if qb == 1 and query_len > 1
      if (bias_i.size(1) == 1 && query_len > 1) {
        bias_i = bias_i.expand({bias_i.size(0), query_len, bias_i.size(2)});
      } else if (bias_i.size(1) > query_len) {
        bias_i = bias_i.narrow(1, 0, query_len);
      }
      attn = attn + bias_i;
    }

    attn = at::softmax(attn, -1);
    attn = attn.to(v_sel.dtype());

    at::Tensor v_per = v_sel.permute({1, 0, 2}).contiguous();
    at::Tensor out_per = at::bmm(attn, v_per);
    at::Tensor out = out_per.permute({1, 0, 2}).contiguous();

    outputs.push_back(out);
    start_idx += query_len;
  }
  at::Tensor result = at::cat(outputs, 0);
  return result;
}

at::Tensor attn_bias_from_alibi_slopes_cpp(const at::Tensor& slopes,  // (batch, nheads), float32
                                           int64_t seqlen_q,
                                           int64_t seqlen_k,
                                           bool causal) {
  at::Tensor s = slopes.unsqueeze(-1).unsqueeze(-1);
  if (causal) {
    at::Tensor ar = at::arange(-seqlen_k + 1, 1, slopes.options());
    ar = ar.view({1, 1, 1, seqlen_k});
    return s * ar;
  } else {
    at::Tensor row_idx = at::arange(seqlen_q, slopes.options().dtype(at::kLong)).unsqueeze(-1);
    at::Tensor col_idx = at::arange(seqlen_k, slopes.options().dtype(at::kLong));
    at::Tensor relative_pos = (row_idx + (seqlen_k - seqlen_q) - col_idx).abs().to(at::kFloat);
    relative_pos = relative_pos.view({1, 1, seqlen_q, seqlen_k});
    return -s * relative_pos;
  }
}

using VarlenParams = std::tuple<std::pair<int, int>,  // (num_query_heads, num_kv_heads)
                                int,                  // head_size
                                at::ScalarType,       // dtype
                                bool,                 // alibi
                                int,                  // soft_cap_code: 0->None, 1->10.0, 2->50.0
                                int                   // num_blocks
                                >;

class FlashAttnVarlenParamTest : public ::testing::TestWithParam<VarlenParams> {};

TEST_P(FlashAttnVarlenParamTest, MatchesReference) {
  torch::manual_seed(1234567890);
  const torch::Device device(torch::kCUDA, 0);

  const std::vector<std::pair<int64_t, int64_t>> seq_lens = {
      {  1, 1328},
      {  5,   18},
      {129,  463}
  };
  const int64_t num_seqs = static_cast<int64_t>(seq_lens.size());
  std::vector<int64_t> query_lens, kv_lens;
  query_lens.reserve(num_seqs);
  kv_lens.reserve(num_seqs);
  for (auto& p : seq_lens) {
    query_lens.push_back(p.first);
    kv_lens.push_back(p.second);
  }
  const int64_t max_query_len = *std::max_element(query_lens.begin(), query_lens.end());
  const int64_t max_kv_len = *std::max_element(kv_lens.begin(), kv_lens.end());

  auto [heads, head_size, dtype, alibi, softcap_code, num_blocks] = GetParam();
  const int64_t num_query_heads = heads.first;
  const int64_t num_kv_heads = heads.second;
  ASSERT_TRUE(num_query_heads % num_kv_heads == 0);

  const int64_t block_size = 32;
  const std::optional<int64_t> sliding_window_opt = std::nullopt;

  std::optional<double> soft_cap_opt;
  if (softcap_code == 1)
    soft_cap_opt = 10.0;
  else if (softcap_code == 2)
    soft_cap_opt = 50.0;
  // if alibi is True and soft_cap is not None
  if (alibi && soft_cap_opt.has_value()) {
    GTEST_SKIP() << "Skip (alibi + soft_cap)";
  }

  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);

  const int64_t total_q_tokens = std::accumulate(query_lens.begin(), query_lens.end(), 0LL);
  at::Tensor q = torch::randn({total_q_tokens, num_query_heads, head_size}, opts);
  at::Tensor k_cache = torch::randn({num_blocks, block_size, num_kv_heads, head_size}, opts);
  at::Tensor v_cache = torch::randn_like(k_cache);

  // cu_seqlens_q
  std::vector<int32_t> cu_q_lens_vec;
  cu_q_lens_vec.reserve(num_seqs + 1);
  cu_q_lens_vec.push_back(0);
  for (auto ql : query_lens) cu_q_lens_vec.push_back(static_cast<int32_t>(cu_q_lens_vec.back() + ql));
  at::Tensor cu_seqlens_q = torch::tensor(cu_q_lens_vec, opts_int);

  // seqused_k
  at::Tensor seqused_k = torch::tensor(std::vector<int32_t>(kv_lens.begin(), kv_lens.end()), opts_int);

  // block table
  const int64_t max_num_blocks_per_seq = (max_kv_len + block_size - 1) / block_size;
  at::Tensor block_table = torch::randint(0, num_blocks, {num_seqs, max_num_blocks_per_seq}, opts_int);

  // alibi slopes / bias
  std::optional<at::Tensor> alibi_slopes_opt = std::nullopt;
  std::optional<at::Tensor> attn_bias_opt = std::nullopt;
  if (alibi) {
    at::Tensor alibi_slopes = torch::ones({num_seqs, num_query_heads},
                                          torch::TensorOptions().dtype(torch::kFloat32).device(device)) *
                              0.3;
    alibi_slopes_opt = alibi_slopes;
    at::Tensor attn_bias =
        attn_bias_from_alibi_slopes_cpp(alibi_slopes, max_query_len, max_kv_len, /*causal=*/true);
    attn_bias_opt = attn_bias;
  }

  auto out_lse = flag_gems::flash_attn_varlen_func(q,
                                                   k_cache,
                                                   v_cache,
                                                   max_query_len,
                                                   cu_seqlens_q,
                                                   max_kv_len,
                                                   /*cu_seqlens_k*/ std::nullopt,
                                                   /*seqused_k*/ seqused_k,
                                                   /*q_v*/ std::nullopt,
                                                   /*dropout_p*/ 0.0,
                                                   /*softmax_scale*/ scale,
                                                   /*causal*/ true,
                                                   /*window_size*/ c10::nullopt,
                                                   /*softcap*/ soft_cap_opt.value_or(0.0),
                                                   /*alibi_slopes*/ alibi_slopes_opt,
                                                   /*deterministic*/ false,
                                                   /*return_attn_probs*/ false,
                                                   /*block_table*/ block_table,
                                                   /*return_softmax_lse*/ false);
  at::Tensor op_output = std::get<0>(out_lse);

  at::Tensor ref_output = ref_paged_attn_cpp(q,
                                             k_cache,
                                             v_cache,
                                             std::vector<int64_t>(query_lens.begin(), query_lens.end()),
                                             std::vector<int64_t>(kv_lens.begin(), kv_lens.end()),
                                             block_table,
                                             scale,
                                             attn_bias_opt,
                                             sliding_window_opt,
                                             soft_cap_opt);

  EXPECT_TRUE(torch::allclose(op_output, ref_output, /*rtol=*/1e-2, /*atol=*/2e-2));
}

INSTANTIATE_TEST_SUITE_P(
    FlashAttnVarlenPytestParity_Param,
    FlashAttnVarlenParamTest,
    ::testing::Combine(::testing::Values(std::make_pair(4, 4), std::make_pair(8, 2), std::make_pair(16, 2)),
                       ::testing::Values(128, 192, 256),
                       ::testing::Values(at::kHalf, at::kBFloat16),
                       ::testing::Values(false, true),
                       ::testing::Values(0, 1, 2),  // soft_cap: None / 10.0 / 50.0
                       ::testing::Values(32768, 2048)));

using SwapParams = std::tuple<at::ScalarType,  // dtype
                              int              // soft_cap_code: 0->None, 1->10.0
                              >;

class FlashAttnVarlenSwapQGParamTest : public ::testing::TestWithParam<SwapParams> {};

TEST_P(FlashAttnVarlenSwapQGParamTest, MatchesReference) {
  torch::manual_seed(1234567890);
  const torch::Device device(torch::kCUDA, 0);

  const std::vector<std::pair<int64_t, int64_t>> seq_lens = {
      {1, 1328},
      {1,   18},
      {1,  463}
  };
  const int64_t num_seqs = static_cast<int64_t>(seq_lens.size());
  std::vector<int64_t> query_lens, kv_lens;
  query_lens.reserve(num_seqs);
  kv_lens.reserve(num_seqs);
  for (auto& p : seq_lens) {
    query_lens.push_back(p.first);
    kv_lens.push_back(p.second);
  }
  const int64_t max_query_len = *std::max_element(query_lens.begin(), query_lens.end());
  const int64_t max_kv_len = *std::max_element(kv_lens.begin(), kv_lens.end());

  const int64_t num_query_heads = 8;
  const int64_t num_kv_heads = 2;
  const int64_t head_size = 128;
  const int64_t block_size = 32;
  const int64_t num_blocks = 2048;

  auto [dtype, softcap_code] = GetParam();
  std::optional<double> soft_cap_opt;
  if (softcap_code == 1) soft_cap_opt = 10.0;

  const double scale = 1.0 / std::sqrt(static_cast<double>(head_size));

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);

  const int64_t total_q_tokens = std::accumulate(query_lens.begin(), query_lens.end(), 0LL);
  at::Tensor q = torch::randn({total_q_tokens, num_query_heads, head_size}, opts);
  at::Tensor k_cache = torch::randn({num_blocks, block_size, num_kv_heads, head_size}, opts);
  at::Tensor v_cache = torch::randn_like(k_cache);

  std::vector<int32_t> cu_q_lens_vec;
  cu_q_lens_vec.reserve(num_seqs + 1);
  cu_q_lens_vec.push_back(0);
  for (auto ql : query_lens) cu_q_lens_vec.push_back(static_cast<int32_t>(cu_q_lens_vec.back() + ql));
  at::Tensor cu_seqlens_q = torch::tensor(cu_q_lens_vec, opts_int);

  at::Tensor seqused_k = torch::tensor(std::vector<int32_t>(kv_lens.begin(), kv_lens.end()), opts_int);
  const int64_t max_num_blocks_per_seq = (max_kv_len + block_size - 1) / block_size;
  at::Tensor block_table = torch::randint(0, num_blocks, {num_seqs, max_num_blocks_per_seq}, opts_int);

  auto out_lse = flag_gems::flash_attn_varlen_func(q,
                                                   k_cache,
                                                   v_cache,
                                                   max_query_len,
                                                   cu_seqlens_q,
                                                   max_kv_len,
                                                   /*cu_seqlens_k*/ std::nullopt,
                                                   /*seqused_k*/ seqused_k,
                                                   /*q_v*/ std::nullopt,
                                                   /*dropout_p*/ 0.0,
                                                   /*softmax_scale*/ scale,
                                                   /*causal*/ true,
                                                   /*window_size*/ c10::nullopt,
                                                   /*softcap*/ soft_cap_opt.value_or(0.0),
                                                   /*alibi_slopes*/ std::nullopt,
                                                   /*deterministic*/ false,
                                                   /*return_attn_probs*/ false,
                                                   /*block_table*/ block_table,
                                                   /*return_softmax_lse*/ false);
  at::Tensor op_output = std::get<0>(out_lse);

  at::Tensor ref_output = ref_paged_attn_cpp(q,
                                             k_cache,
                                             v_cache,
                                             std::vector<int64_t>(query_lens.begin(), query_lens.end()),
                                             std::vector<int64_t>(kv_lens.begin(), kv_lens.end()),
                                             block_table,
                                             scale,
                                             /*attn_bias*/ std::nullopt,
                                             /*sliding_window*/ std::nullopt,
                                             soft_cap_opt);

  EXPECT_TRUE(torch::allclose(op_output, ref_output, /*rtol=*/1e-2, /*atol=*/2e-2));
}

INSTANTIATE_TEST_SUITE_P(FlashAttnVarlenPytestParity_Param_Swap,
                         FlashAttnVarlenSwapQGParamTest,
                         ::testing::Combine(::testing::Values(at::kHalf, at::kBFloat16),
                                            ::testing::Values(0, 1)  // soft_cap: None / 10.0
                                            ));
