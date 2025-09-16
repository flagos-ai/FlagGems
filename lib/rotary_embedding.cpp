#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <optional>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

void check_rotary_embedding_inputs(
    const at::Tensor& q,    // [batch_size, seq_len, q_heads, head_dim] or [num_tokens, q_heads, head_dim]
    const at::Tensor& k,    // [batch_size, seq_len, k_heads, head_dim] or [num_tokens, k_heads, head_dim]
    const at::Tensor& cos,  // [max_seq_len, head_dim // 2]
    const at::Tensor& sin,  // [max_seq_len, head_dim // 2]
    const std::optional<at::Tensor>& position_ids) {  // None or [..., seq_len]
  // 1. Check that q and k have the same head dimension
  TORCH_CHECK(k.size(-1) == q.size(-1),
              "q and k must have the same last dimension, got ",
              q.sizes(),
              " and ",
              k.sizes());

  // 2. Check that cos and sin have the same last dimension
  TORCH_CHECK(cos.size(-1) == sin.size(-1),
              "cos and sin must have the same last dimension, got ",
              cos.sizes(),
              " and ",
              sin.sizes());

  // 3. Check that cos/sin dimension matches q/k head_dim // 2
  TORCH_CHECK(cos.size(-1) * 2 == q.size(-1),
              "cos/sin dim must be half of q/k dim, got ",
              cos.sizes(),
              " and ",
              q.sizes());

  // 4. Check that cos and sin are contiguous at the last dimension
  TORCH_CHECK(cos.stride(-1) == 1,
              "cos must be contiguous at the last dimension, got stride ",
              cos.stride(-1));
  TORCH_CHECK(sin.stride(-1) == 1,
              "sin must be contiguous at the last dimension, got stride ",
              sin.stride(-1));

  auto q_sizes = q.sizes();
  auto k_sizes = k.sizes();

  // 5. Check that q and k have the same number of dimensions
  TORCH_CHECK(q_sizes.size() == k_sizes.size(),
              "q and k must have the same number of dimensions, got ",
              q_sizes.size(),
              " and ",
              k_sizes.size());

  // 6. Check that all dimensions except the last two match between q and k
  for (int i = 0; i < q_sizes.size() - 2; ++i) {
    TORCH_CHECK(q_sizes[i] == k_sizes[i],
                "Mismatch in q and k shape at dim ",
                i,
                ": got ",
                q_sizes[i],
                " and ",
                k_sizes[i]);
  }

  // 7. If position_ids is not provided, q must have 4 dimensions
  if (!position_ids.has_value()) {
    TORCH_CHECK(q_sizes.size() == 4,
                "q must have 4 dimensions if position_ids is not provided, got ",
                q_sizes.size());
  } else {
    auto pos_sizes = position_ids.value().sizes();

    // 8. Check that position_ids has the same number of dims as q.shape[:-2]
    TORCH_CHECK(pos_sizes.size() == q_sizes.size() - 2,
                "position_ids must have the same number of dims as q.shape[:-2], got ",
                pos_sizes.size(),
                " and ",
                q_sizes.size() - 2);

    // 9. Check that position_ids shape matches q.shape[:-2] on each dimension
    for (int i = 0; i < pos_sizes.size(); ++i) {
      TORCH_CHECK(pos_sizes[i] == q_sizes[i],
                  "Mismatch in position_ids and q shape at dim ",
                  i,
                  ": got ",
                  pos_sizes[i],
                  " and ",
                  q_sizes[i]);
    }
  }
}

void rotary_embedding_inplace(
    at::Tensor& q,          // [batch_size, seq_len, q_heads, head_dim] or [num_tokens, q_heads, head_dim]
    at::Tensor& k,          // [batch_size, seq_len, k_heads, head_dim] or [num_tokens, k_heads, head_dim]
    const at::Tensor& cos,  // [max_seq_len, head_dim // 2]
    const at::Tensor& sin,  // [max_seq_len, head_dim // 2]
    const std::optional<at::Tensor>& position_ids,  // None or [..., seq_len]
    bool rotary_interleaved) {                      // default false

  check_rotary_embedding_inputs(q, k, cos, sin, position_ids);

  auto q_sizes = q.sizes();
  auto k_sizes = k.sizes();

  std::optional<int64_t> seq_len = std::nullopt;
  std::optional<at::Tensor> flat_position_ids = std::nullopt;
  if (!position_ids.has_value()) {
    seq_len = q_sizes[1];
  } else {                                                // default case
    flat_position_ids = position_ids.value().view({-1});  // flatten the position_ids tensor
  }

  q = q.view({-1, q.size(-2), q.size(-1)});  // [num_tokens, q_heads, head_dim]
  k = k.view({-1, k.size(-2), k.size(-1)});  // [num_tokens, k_heads, head_dim]

  int64_t n_tokens = q.size(0);
  int64_t q_heads = q.size(1);
  int64_t head_dim = q.size(2);

  int64_t padded_head_dim = std::max(utils::next_power_of_2(head_dim), int64_t(16));

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "rotary_embedding.py"),
      "apply_rotary_pos_emb_inplace_kernel");

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(q.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  /* signature info
def apply_rotary_pos_emb_inplace_kernel(
    q_ptr,   # (n_tokens, q_heads, head_dim)
    k_ptr,   # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
  ) */
  f(raw_stream,
    n_tokens,
    1,
    1,
    /* num_warps */ 8,
    /* num_stages */ 1,
    q,
    k,
    cos,
    sin,
    flat_position_ids,  // std::optional<at::Tensor>
    q.stride(0),
    q.stride(1),
    q.stride(2),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    flat_position_ids.has_value() ? flat_position_ids.value().stride(0)
                                  : 0,  // 0 if flat_position_ids is not defined
    cos.stride(0),
    sin.stride(0),
    seq_len,     // std::optional<long int>
    q.size(-2),  // q_heads
    k.size(-2),  // k_heads
    head_dim,
    padded_head_dim,
    rotary_interleaved,
    cos.size(0)  // max_seq_len
  );

  // Reshape back to original shapes
  q = q.view(q_sizes.vec());
  k = k.view(k_sizes.vec());

  return;
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding(const at::Tensor& q,
                                                    const at::Tensor& k,
                                                    const at::Tensor& cos,
                                                    const at::Tensor& sin,
                                                    const std::optional<at::Tensor>& position_ids,
                                                    bool rotary_interleaved) {
  // Check inputs
  check_rotary_embedding_inputs(q, k, cos, sin, position_ids);

  auto q_sizes = q.sizes();
  auto k_sizes = k.sizes();
  std::optional<int64_t> seq_len = std::nullopt;
  std::optional<at::Tensor> flat_position_ids = std::nullopt;

  if (!position_ids.has_value()) {
    seq_len = q_sizes[1];
  } else {                                                // default case
    flat_position_ids = position_ids.value().view({-1});  // flatten the position_ids tensor
  }

  auto q_view = q.view({-1, q.size(-2), q.size(-1)});  // [num_tokens, q_heads, head_dim]
  auto k_view = k.view({-1, k.size(-2), k.size(-1)});  // [num_tokens, k_heads, head_dim]

  int64_t n_tokens = q_view.size(0);
  int64_t q_heads = q_view.size(1);
  int64_t head_dim = q_view.size(2);

  int64_t padded_head_dim = std::max(utils::next_power_of_2(head_dim), int64_t(16));

  auto q_embed = at::empty_like(q_view);
  auto k_embed = at::empty_like(k_view);
  auto q_embed_stride = q_embed.strides();
  auto k_embed_stride = k_embed.strides();

  const TritonJITFunction& f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "fused" / "rotary_embedding.py"),
      "apply_rotary_pos_emb_kernel");
  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(q.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(raw_stream,
    n_tokens,
    1,
    1,
    /* num_warps */ 8,
    /* num_stages */ 1,
    q_embed,
    k_embed,
    q_view,
    k_view,
    cos,
    sin,
    flat_position_ids,  // std::optional<at::Tensor>
    q_view.stride(0),
    q_view.stride(1),
    q_view.stride(2),
    k_view.stride(0),
    k_view.stride(1),
    k_view.stride(2),
    q_embed_stride[0],
    q_embed_stride[1],
    q_embed_stride[2],
    k_embed_stride[0],
    k_embed_stride[1],
    k_embed_stride[2],
    flat_position_ids.has_value() ? flat_position_ids.value().stride(0)
                                  : 0,  // 0 if flat_position_ids is not defined
    cos.stride(0),
    sin.stride(0),
    seq_len,          // std::optional<long int>
    q_view.size(-2),  // q_heads
    k_view.size(-2),  // k_heads
    head_dim,
    padded_head_dim,
    rotary_interleaved,
    cos.size(0)  // max_seq_len
  );

  // Reshape back to original shapes
  q_embed = q_embed.view(q_sizes.vec());
  k_embed = k_embed.view(k_sizes.vec());
  return std::make_tuple(q_embed, k_embed);
}
}  // namespace flag_gems
