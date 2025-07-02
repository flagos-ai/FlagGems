#include <gtest/gtest.h>
#include <cmath>
#include <optional>
#include <tuple>
#include "flag_gems/operators.h"
#include "torch/torch.h"

std::tuple<at::Tensor, at::Tensor> get_rope_cos_sin(int64_t max_seq_len,
                                                    int64_t dim,
                                                    c10::ScalarType dtype,
                                                    double base = 10000.0,
                                                    c10::Device device = at::kCUDA) {
  auto arange_dtype = at::kFloat;
  at::Tensor inv_freq = at::arange(0, dim, 2, at::TensorOptions().dtype(arange_dtype).device(device));
  inv_freq = inv_freq.div(dim).to(at::kFloat);
  inv_freq = 1.0 / at::pow(base, inv_freq);

  at::Tensor t = at::arange(0, max_seq_len, at::TensorOptions().dtype(inv_freq.scalar_type()).device(device));

  at::Tensor freqs = at::ger(t, inv_freq);  // ger = outer product
  // at::Tensor freqs = torch::outer(t, inv_freq);

  at::Tensor cos = freqs.cos().to(dtype);
  at::Tensor sin = freqs.sin().to(dtype);

  return std::make_tuple(cos, sin);
}

torch::Tensor rotate_half(const torch::Tensor& x) {
  auto dim = x.size(-1) / 2;
  auto x1 = x.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, dim)});
  auto x2 = x.index({torch::indexing::Ellipsis, torch::indexing::Slice(dim)});
  return torch::cat({-x2, x1}, -1);
}

torch::Tensor rotate_interleave(const torch::Tensor& x) {
  auto x1 = x.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, torch::indexing::None, 2)});  // ::2
  auto x2 =
      x.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None, 2)});  // 1::2
  return torch::stack({-x2, x1}, -1).flatten(-2);
}

std::tuple<torch::Tensor, torch::Tensor> torch_apply_rotary_pos_emb_cpp(
    torch::Tensor q,    // [batch_size, seq_len, q_heads, head_dim] or [num_tokens, q_heads, head_dim]
    torch::Tensor k,    // [batch_size, seq_len, k_heads, head_dim] or [num_tokens, k_heads, head_dim]
    torch::Tensor cos,  // [max_seq_len, head_dim // 2]
    torch::Tensor sin,  // [max_seq_len, head_dim // 2]
    std::optional<torch::Tensor> position_ids,  // [batch_size, seq_len] or None
    bool rotary_interleaved) {
  q = q.to(torch::kFloat);
  k = k.to(torch::kFloat);

  if (!position_ids.has_value()) {
    auto seq_len = q.size(-3);
    {
      cos = cos.unsqueeze(0);
      cos = cos.index({torch::indexing::Slice(), torch::indexing::Slice(0, seq_len)});
      cos = cos.unsqueeze(-2);
    }  // same as cos = cos[None, : q.size(-3), None, :]
    {
      sin = sin.unsqueeze(0);
      sin = sin.index({torch::indexing::Slice(), torch::indexing::Slice(0, seq_len)});
      sin = sin.unsqueeze(-2);
    }                                   // same as sin = sin[None, : q.size(-3), None, :]
  } else {                              // default case
    auto pos = position_ids.value();    // [batch_size, seq_len]
    auto pos_flat = pos.reshape({-1});  // [batch_size * seq_len]
    cos = cos.index_select(0, pos_flat)
              .view({pos.size(0), pos.size(1), cos.size(-1)})
              .unsqueeze(-2);  // [batch_size, seq_len, 1, head_dim // 2]
    sin = sin.index_select(0, pos_flat)
              .view({pos.size(0), pos.size(1), sin.size(-1)})
              .unsqueeze(-2);  // [batch_size, seq_len, 1, head_dim // 2]
  }

  torch::Tensor cos_full, sin_full;
  if (rotary_interleaved) {
    cos_full = torch::repeat_interleave(cos, 2, -1);
    sin_full = torch::repeat_interleave(sin, 2, -1);
  } else {
    cos_full = torch::cat({cos, cos}, -1);
    sin_full = torch::cat({sin, sin}, -1);
  }

  auto rotate_fn = rotary_interleaved ? rotate_interleave : rotate_half;

  auto q_embed = (q * cos_full) + (rotate_fn(q) * sin_full);
  auto k_embed = (k * cos_full) + (rotate_fn(k) * sin_full);

  return {q_embed, k_embed};
}

class RotaryEmbeddingTest
    : public ::testing::TestWithParam<std::tuple<int, int, int, int, torch::ScalarType, bool, bool>> {};

TEST_P(RotaryEmbeddingTest, CompareWithReference) {
  auto [batch_size, seq_len, q_heads, head_dim, dtype, rotary_interleaved, has_pos_id] = GetParam();

  auto max_seq_len = std::max(seq_len, 2048);  // Ensure max_seq_len is at least seq_len or 2048

  torch::manual_seed(0);
  torch::Device device(torch::kCUDA, 0);

  int k_heads = std::max(1, q_heads / 2);  // 随便设的一个可变 k_heads

  torch::Tensor q = torch::randn({batch_size, seq_len, q_heads, head_dim},
                                 torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor k = torch::randn({batch_size, seq_len, k_heads, head_dim},
                                 torch::TensorOptions().device(device).dtype(dtype));

  c10::optional<torch::Tensor> position_ids;
  if (has_pos_id) {
    position_ids = torch::randint(0,
                                  max_seq_len,
                                  {batch_size, seq_len},
                                  torch::TensorOptions().device(device).dtype(torch::kLong));
  }

  auto [cos, sin] = get_rope_cos_sin(max_seq_len, head_dim, dtype, 10000.0, device);

  auto [q_ref, k_ref] = torch_apply_rotary_pos_emb_cpp(q, k, cos, sin, position_ids, rotary_interleaved);
  auto [q_out, k_out] = flag_gems::rotary_embedding(q, k, cos, sin, position_ids, rotary_interleaved);

  double atol = (dtype == torch::kFloat16) ? 1e-2 : 1e-5;
  double rtol = (dtype == torch::kFloat16) ? 1e-2 : 1e-3;

  ASSERT_TRUE(torch::allclose(q_out, q_ref.to(dtype), rtol, atol));
  ASSERT_TRUE(torch::allclose(k_out, k_ref.to(dtype), rtol, atol));
}

INSTANTIATE_TEST_SUITE_P(RotaryEmbeddingTests,
                         RotaryEmbeddingTest,
                         ::testing::Values(
                             // batch_size, seq_len, q_heads, head_dim, dtype, rotary_interleaved, has_pos_id
                             std::make_tuple(1, 16, 8, 64, torch::kFloat32, true, true),
                             std::make_tuple(2, 512, 4, 64, torch::kFloat32, false, true),
                             std::make_tuple(4, 1024, 8, 128, torch::kFloat16, true, true),
                             std::make_tuple(8, 2048, 128, 128, torch::kBFloat16, false, true),
                             std::make_tuple(8, 2048, 32, 64, torch::kFloat16, true, false),
                             std::make_tuple(8, 2048, 16, 32, torch::kBFloat16, false, false),
                             std::make_tuple(8, 1024, 64, 128, torch::kFloat32, true, false),
                             std::make_tuple(8, 2048, 128, 256, torch::kFloat32, false, false)));
