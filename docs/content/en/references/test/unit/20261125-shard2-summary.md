|                       filepath                       |                             function                              | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------- | ----------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_im2col.py                                 | test_im2col                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_cholesky.py                        | test_linalg_cholesky                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_cholesky.py                        | test_linalg_cholesky_upper                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_cholesky.py                        | test_linalg_cholesky_batch                                        |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                            | test_multinomial_with_replacement                                 |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                            | test_multinomial_with_replacement_1                               |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                            | test_multinomial_does_not_mutate_input                            |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                            | test_multinomial_without_replacement                              |      1 |      0 |       0 |        1 |
| tests/test_rshift.py                                 | test_rshift_tensor                                                |      1 |      0 |       0 |        1 |
| tests/test_rshift.py                                 | test_rshift_scalar                                                |      1 |      0 |       0 |        1 |
| tests/test_rshift.py                                 | test_rshift_output_overloads                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm                                                        |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_beta_zero_ignores_bias                                 |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_vector_bias_shapes                                     |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_scalar_bias                                            |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_padded_vector_bias                                     |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_beta_zero_ignores_bias                             |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_vector_bias_shapes                                 |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_noncontiguous                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_broadcast_bias                                         |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_fp32_accum                                       |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_beta_zero_ignores_bias                           |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_out_fp32_accum                                   |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_out_beta_zero_ignores_bias                       |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_shape                       |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_unsupported_dtype               |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_padding                     |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_stride                      |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_groups                      |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_input_size                  |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_none_operand                    |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_dense_operand                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_2d                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_3d                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_broadcast                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_zero_sized                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_mixed_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_complex                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_grad                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                          | test_linalg_matmul_double_grad                                    |      1 |      0 |       0 |        1 |
| tests/test_is_any_true.py                            | test_is_any_true                                                  |      1 |      0 |       0 |        1 |
| tests/test_is_any_true.py                            | test_is_any_true_arbitrary_rank                                   |      1 |      0 |       0 |        1 |
| tests/test_is_any_true.py                            | test_is_any_true_scalar_and_expand                                |      1 |      0 |       0 |        1 |
| tests/test_is_any_true.py                            | test_is_any_true_invalid_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_hardshrink_backward.py                    | test_hardshrink_backward_grad_input                               |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u                       |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u_scalar_n              |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u_                      |      1 |      0 |       0 |        1 |
| tests/test_unique2.py                                | test_unique2                                                      |      1 |      0 |       0 |        1 |
| tests/test_unique2.py                                | test_unique2_empty_input                                          |      1 |      0 |       0 |        1 |
| tests/test_unique2.py                                | test_unique2_large_reduce                                         |      1 |      0 |       0 |        1 |
| tests/test_unique2.py                                | test_unique2_edge_cases                                           |      1 |      0 |       0 |        1 |
| tests/test_unique2.py                                | test_unique2_float_special_values                                 |      1 |      0 |       0 |        1 |
| tests/test_DSA/test_indexer_k_tiled.py               | test_lighting_indexer_forward                                     |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices                                                 |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_bsr                                             |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_batched                                         |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_int32_buffer                                    |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_alias_and_mutation                              |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_dispatch_stability                              |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_dispatch_sentinel                               |      1 |      0 |       0 |        1 |
| tests/test_crow_indices.py                           | test_crow_indices_unsupported_layout_raises                       |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims_                                                    |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__negative_dims                                      |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__non_contiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__scalar_and_empty                                   |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__aliasing                                           |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__version_counter                                    |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__version_counter_identity                           |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__invalid_dims                                       |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__expanded_overlap                                   |      1 |      0 |       0 |        1 |
| tests/test_swapdims_.py                              | test_swapdims__int_dtypes                                         |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d_backward.py              | test_reflection_pad3d_backward                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                       | test_linalg_ldl_solve                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                       | test_linalg_ldl_solve_batched                                     |      1 |      0 |       0 |        1 |
| tests/test_addmv_.py                                 | test_addmv_                                                       |      1 |      0 |       0 |        1 |
| tests/test_cauchy.py                                 | test_cauchy_accuracy                                              |      1 |      0 |       0 |        1 |
| tests/test_cauchy.py                                 | test_cauchy_out_accuracy                                          |      1 |      0 |       0 |        1 |
| tests/test_cauchy.py                                 | test_cauchy_reproducibility                                       |      1 |      0 |       0 |        1 |
| tests/test_cauchy.py                                 | test_cauchy_large_tensors                                         |      1 |      0 |       0 |        1 |
| tests/test_matrix_exp.py                             | test_linalg_matrix_exp                                            |      1 |      0 |       0 |        1 |
| tests/test_matrix_exp.py                             | test_linalg_matrix_exp_batch                                      |      1 |      0 |       0 |        1 |
| tests/test_matrix_exp.py                             | test_linalg_matrix_exp_identity                                   |      1 |      0 |       0 |        1 |
| tests/test_matrix_exp.py                             | test_linalg_matrix_exp_zero                                       |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured                                    |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_strided                            |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_special_values                     |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_rejects_rank                       |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_rejects_dtype                      |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_rejects_geometry                   |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_rejects_pattern                    |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py              | test_to_sparse_semi_structured_rejects_non_tensor                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_reduce                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_dense_noncontiguous                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_rejects_dense_sparse                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_rejects_shape_mismatch                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_reduce_rejects_non_csr                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_sparse_mm_reduce_rejects_unknown_reduce                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_dispatch_reaches_gems_impl                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_cpu_inputs_fall_back_without_recursing                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_requires_grad_rejected                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm.py                              | test_reduce_against_aten_cpu                                      |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist                                                        |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist_p1                                                     |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist_pinf                                                   |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist_p_general                                              |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist_p0                                                     |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                  | test_pdist_p_large                                                |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention                                  |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_self_attention                   |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_no_average                       |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_src_mask                         |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_key_padding_mask                 |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_generic_mask                     |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount                                            |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_with_weights                               |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_with_minlength                             |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_empty                                      |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_single                                     |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_all_zeros                                  |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                               | test_accuracy_bincount_weights_edge_cases                         |      1 |      0 |       0 |        1 |
| tests/test__resize_output_.py                        | test_resize_output_                                               |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                 | test_hsplit_int                                                   |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                 | test_hsplit_array                                                 |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                 | test_hsplit_array_out_of_range                                    |      1 |      0 |       0 |        1 |
| tests/test_less_equal_.py                            | test_less_equal_                                                  |      1 |      0 |       0 |        1 |
| tests/test_less_equal_.py                            | test_less_equal_scalar_                                           |      1 |      0 |       0 |        1 |
| tests/test_less_equal_.py                            | test_less_equal__byte                                             |      1 |      0 |       0 |        1 |
| tests/test_less_equal_.py                            | test_less_equal_scalar__byte                                      |      1 |      0 |       0 |        1 |
| tests/test_unfold.py                                 | test_unfold                                                       |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes_                                                    |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__negative_dims                                      |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__non_contiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__scalar_and_empty                                   |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__aliasing                                           |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__version_counter                                    |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__version_counter_identity                           |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__invalid_dims                                       |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__expanded_overlap                                   |      1 |      0 |       0 |        1 |
| tests/test_swapaxes_.py                              | test_swapaxes__int_dtypes                                         |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                             | test_native_batch_norm                                            |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                             | test_batch_norm                                                   |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                             | test_batch_norm_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_blackman_window.py                        | test_blackman_window                                              |      1 |      0 |       0 |        1 |
| tests/test_blackman_window.py                        | test_blackman_window_periodic                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_out                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_out_invalid                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_non_contiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_invalid_mode                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_input_not_mutated                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_zero_column                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_rank_deficient                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_empty                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_transposed_input                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_unsupported_dtype                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                              | test_linalg_qr_dim_lt_2                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_out                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_extreme                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_non_contiguous          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_out_non_contiguous      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_unit_dim                |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py   | test_cudnn_affine_grid_generator_backward_out_dispatch            |      1 |      0 |       0 |        1 |
| tests/test_embedding_sparse_backward.py              | test_embedding_sparse_backward                                    |      1 |      0 |       0 |        1 |
| tests/test_and_scalar.py                             | test_and_scalar                                                   |      1 |      0 |       0 |        1 |
| tests/test_mT.py                                     | test_mT                                                           |      1 |      0 |       0 |        1 |
| tests/test_mT.py                                     | test_mT_noncontiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_mT.py                                     | test_mT_shares_storage                                            |      1 |      0 |       0 |        1 |
| tests/test_mT.py                                     | test_mT_autograd                                                  |      1 |      0 |       0 |        1 |
| tests/test_mT.py                                     | test_mT_rank1_raises                                              |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d                                                  |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_kernel_padding_stride                            |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_without_bias                                     |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_default_bias                                     |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_int_kernel_and_defaults                          |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_noncontiguous_input                              |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_does_not_mutate_operands                         |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_rejects_unsupported_dtype                        |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_rejects_non_tensor_weight                        |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_rejects_rank3_weight                             |      1 |      0 |       0 |        1 |
| tests/test_thnn_conv2d.py                            | test_thnn_conv2d_rejects_cin_mismatch                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex_check_errors                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex_singular                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex_zero_pivot                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex_check_errors_raises                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor_ex.py                    | test_linalg_lu_factor_ex_out                                      |      1 |      0 |       0 |        1 |
| tests/test_conj.py                                   | test_conj                                                         |      1 |      0 |       0 |        1 |
| tests/test_conj.py                                   | test__conj                                                        |      1 |      0 |       0 |        1 |
| tests/test_rad2deg.py                                | test_rad2deg                                                      |      1 |      0 |       0 |        1 |
| tests/test_rad2deg.py                                | test_rad2deg_                                                     |      1 |      0 |       0 |        1 |
| tests/test_special_entr.py                           | test_special_entr                                                 |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d.py                             | test_max_pool3d                                                   |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d.py                             | test_max_pool3d_autograd                                          |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_registers_operators_to_torch_ops_namespace                   |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_idempotent_multiple_calls                                    |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_silu_and_mul_callable                                        |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_silu_and_mul_with_clamp_callable                             |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_rms_norm_callable                                            |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_silu_and_mul_dtypes                                          |      1 |      0 |       0 |        1 |
| tests/test_patch_empyt_vllm.py                       | test_silu_and_mul_shapes                                          |      1 |      0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py       | test_accuracy_has_compatible_shallow_copy_type                    |      1 |      0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py       | test_accuracy_has_compatible_shallow_copy_type_dense_family       |      1 |      0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py       | test_accuracy_has_compatible_shallow_copy_type_sparse_family      |      1 |      0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py       | test_accuracy_has_compatible_shallow_copy_type_opaque             |      1 |      0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py       | test_accuracy_has_compatible_shallow_copy_type_cross_family       |      1 |      0 |       0 |        1 |
| tests/test_expand_copy.py                            | test_expand_copy                                                  |      1 |      0 |       0 |        1 |
| tests/test_expand_copy.py                            | test_expand_copy_same_shape                                       |      1 |      0 |       0 |        1 |
| tests/test_softshrink_backward.py                    | test_softshrink_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_atan2_.py                                 | test_atan2_                                                       |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                              | test_bucketize                                                    |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                              | test_bucketize_right                                              |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                              | test_bucketize_int32                                              |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                              | test_bucketize_boundary_cases                                     |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                              | test_bucketize_noncontiguous_input                                |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                | test_unflatten_dense_tensors                                      |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                | test_unflatten_dense_tensors_metadata                             |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                | test_unflatten_dense_tensors_strided_flat                         |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                | test_unflatten_dense_tensors_exceeds_flat                         |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa                                        |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_uint8_and_layout                       |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_transposed_input                       |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_out                                    |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_vec                                    |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_vec_requires_one_size                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_explicit_size                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_infer_size                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_rejects_bad_layout                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap  |      1 |      0 |       0 |        1 |
| tests/test_reshape_and_cache_flash.py                | test_reshape_and_cache_flash                                      |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter                                                    |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_                                                   |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_out                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_out_none                                           |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_inplace_is_self                                    |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_rejects_out                                        |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_edge_values                                        |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_nan                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_infinity                                           |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                              | test_nextafter_small_values                                       |      1 |      0 |       0 |        1 |
| tests/test_lt_.py                                    | test_lt_                                                          |      1 |      0 |       0 |        1 |
| tests/test_lt_.py                                    | test_lt_scalar_                                                   |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_value_range                |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_dim                        |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_structure                  |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_special_values             |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_out                        |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_rejects_dtype              |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_rejects_dim                |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_rejects_dense_input        |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test__sparse_log_softmax_backward_data_rejects_mixed_dtype        |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax_backward_data.py      | test_sparse_log_softmax_backward_data_dispatch                    |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_gems_route                                            |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out_gems_route                                        |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm                                                       |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_int32_overflow                                        |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out_int32_overflow                                    |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_observes_mat2_inplace_updates                         |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_observes_inference_mat2_inplace_updates               |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_does_not_reuse_packed_mat2_for_distinct_alias         |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out_noncontiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out_k_zero                                            |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_invalid_inputs                                        |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_mixed_input_devices                                   |      1 |      0 |       0 |        1 |
| tests/test_int_mm.py                                 | test_int_mm_out_validation                                        |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_                                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw__tensor_lr                                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_amp                                              |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_mixed_state                                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_channels_last                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_many_tensors                                     |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_invalid                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_invalid_hyperparameter                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_native                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_graph                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_launch_boundaries                                |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_bias_correction                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_invalid_lr_dtype                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_bulk_validation                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_storage_offsets                                  |      1 |      0 |       0 |        1 |
| tests/test_cos.py                                    | test_cos                                                          |      1 |      0 |       0 |        1 |
| tests/test_cos.py                                    | test_cos_                                                         |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward                            |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward_with_scale                 |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward_grad_input                 |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward_empty                      |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward_uint8                      |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py      | test_upsample_nearest_exact1d_backward_invalid_size               |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_broadcast                                       |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_out                                             |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_tensor_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_tensor_scalar_out                               |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_scalar_tensor                                   |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_scalar_tensor_out                               |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_special_values                                  |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_numerical_regimes                               |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_negative_noninteger_q                           |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_integer_promotion                               |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_half_bfloat16_promotion                         |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_rejects_native_low_precision                    |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_out_exact_alias                                 |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_out_partial_overlap_rejected                    |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_out_casts_after_float_compute                   |      1 |      0 |       0 |        1 |
| tests/test_selu.py                                   | test_selu                                                         |      1 |      0 |       0 |        1 |
| tests/test_selu.py                                   | test_selu_                                                        |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_partition                                      |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_partition_rejects_nonpositive_tiles            |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_partition_rejects_nonpositive_max_grid         |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_codegen_config_balance_grid_defaults_off                     |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_codegen_config_accepts_balance_grid_opt_in                   |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_partition_covers_each_tile_once                |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_generated_wrapper_source_and_runtime_partition |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_cache_isolation                                |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py         | test_balanced_grid_real_kernel_covers_non_power_of_two_tail       |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward                                              |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_autograd_chain                               |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_non_contiguous                               |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_middle_and_negative_dim                      |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_empty                                        |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_scalar_input                                 |      1 |      0 |       0 |        1 |
| tests/test_cumsum_backward.py                        | test_cumsum_backward_use_gems_entry                               |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                              | test_bernoulli_                                                   |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                              | test_bernoulli_various_p                                          |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                              | test_bernoulli                                                    |      1 |      0 |       0 |        1 |
| tests/test_dropout_.py                               | test_dropout_                                                     |      1 |      0 |       0 |        1 |
| tests/test_dropout_.py                               | test_dropout__noop                                                |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices                                                      |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_hybrid                                               |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_alias_and_mutation                                   |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_uncoalesced_rejected                                 |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_unsupported_layout_raises                            |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test_indices_dense_rejected                                       |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices                                                     |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_uncoalesced_keeps_coordinates                       |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_empty                                               |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_hybrid                                              |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_alias_and_mutation                                  |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_unsupported_layout_raises                           |      1 |      0 |       0 |        1 |
| tests/test_indices.py                                | test__indices_dense_rejected                                      |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell                                           |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_non_contiguous                            |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_zero_sized                                |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_invalid_input                             |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_complex_rejected                          |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_autograd_graph                            |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                              | test_accuracy_lstm_cell_keyword_arguments                         |      1 |      0 |       0 |        1 |
| tests/test_tan.py                                    | test_tan                                                          |      1 |      0 |       0 |        1 |
| tests/test_tan.py                                    | test_tan_                                                         |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_vector_rhs                                           |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_identity                                             |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_rejects_matrix_rhs                                   |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_rejects_left_false                                   |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_rejects_device_mismatch                              |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_requires_csr                                         |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                | test_spsolve_requires_square                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm                                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_channels_last                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_empty                                   |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_zero_output_scale                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_bad_size                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_bad_dtype                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_mismatched_qparams                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_resizes                             |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_storage_offset                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_empty                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_wrong_dtype                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                   | test_quantized_batch_norm_out_wrong_device                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_reduction_large.py                | test_large_foreach_reduction_on_device                            |      1 |      0 |       0 |        1 |
| tests/test_unfold_backward.py                        | test_unfold_backward                                              |      1 |      0 |       0 |        1 |
| tests/test_glu.py                                    | test_glu                                                          |      1 |      0 |       0 |        1 |
| tests/test_glu.py                                    | test_glu_backward                                                 |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py          | test_binary_cross_entropy_backward                                |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py          | test_binary_cross_entropy_backward_weighted                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims_dtypes                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims_default_dtype                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims_out                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims_rejects_bad_layout               |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py            | test_sparse_coo_tensor_with_dims_rejects_size_mismatch            |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors                                 |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_weighted                        |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_density                         |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_density_weighted                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_non_uniform_bins                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_many_bins                       |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_many_bins_multidim              |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_many_bins_density_weighted      |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_higher_dim_input                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_outliers                        |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_boundary_edges                  |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_dims                            |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_out                             |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_out_weighted                    |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_bin_edge_precision              |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_density_zero_total              |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_density_signed_weights          |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_weight_dtype_mismatch           |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_out_resize                      |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_out_dtype_mismatch              |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_tensors.py           | test_histogramdd_from_bin_tensors_out_non_contiguous              |      1 |      0 |       0 |        1 |
| tests/test_masked_scale.py                           | test_masked_scale                                                 |      1 |      0 |       0 |        1 |
| tests/test_sort.py                                   | test_sort                                                         |      1 |      0 |       0 |        1 |
| tests/test_sort.py                                   | test_sort_stable                                                  |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half                                                    |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half_strided                                            |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half_special_values                                     |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half_complex_layout                                     |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half_complex_noncontiguous_grad                         |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                              | test_cast_Half_autograd                                           |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_div                                                    |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_divide_                                                |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_divide_scalar_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_divide_tensor_int                                      |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_divide_tensor_scalar_int                               |      1 |      0 |       0 |        1 |
| tests/test_trunc_divide.py                           | test_trunc_div_scalar_tensor_int                                  |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                         | test_pdist_backward                                               |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                         | test_pdist_backward_p1                                            |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                         | test_pdist_backward_pinf                                          |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                         | test_pdist_backward_general                                       |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_coo_all_sparse                                          |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_coo_hybrid                                              |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_matches_indices_formula                                 |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_empty                                                   |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_single_entry                                            |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_uncoalesced                                             |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_nan_inf_values_ignored                                  |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_dense_raises                                            |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_compressed_layouts_raise                                |      1 |      0 |       0 |        1 |
| tests/test_dimI.py                                   | test_dimI_dispatch_stability                                      |      1 |      0 |       0 |        1 |
| tests/test_amax.py                                   | test_amax                                                         |      1 |      0 |       0 |        1 |
| tests/test_enable_api.py                             | test_enable_with_exclude                                          |      1 |      0 |       0 |        1 |
| tests/test_enable_api.py                             | test_only_enable                                                  |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy                                                        |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_special_values                                         |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_tensor_scalar                                          |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_tensor_scalar_out                                      |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_scalar_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                  | test_xlogy_scalar_tensor_out                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_mm.py              | test_sparse_semi_structured_mm                                    |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                   | test_rsub_tensor                                                  |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                   | test_rsub_scalar                                                  |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                | test_max_pool3d_with_indices                                      |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                | test_max_pool3d_with_indices_ties                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_common                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_large                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_out_parameter                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_linalg_matrix_power_out                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_out_missing_raises                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                    | test_invalid_shape_rejected                                       |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_.py                        | test_atanh_backward_                                              |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_.py                        | test_atanh_backward__leaf_error_preserves_input                   |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal                                                |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal_scalar                                         |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal_scalar_byte                                    |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal_                                               |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal__byte                                          |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal__byte_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal_scalar_                                        |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                          | test_greater_equal_scalar__byte                                   |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                 | test_copy_sparse_to_sparse_                                       |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                 | test_copy_sparse_to_sparse_self_alias                             |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                 | test_copy_sparse_to_sparse_dtype_conversion                       |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                 | test_copy_sparse_to_sparse_nonempty_shrink_raises                 |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                 | test_copy_sparse_to_sparse_nonempty_grow                          |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward_impl.py     | test_thnn_fused_lstm_cell_backward_impl                           |      1 |      0 |       0 |        1 |
| tests/test_special_i0e.py                            | test_special_i0e                                                  |      1 |      0 |       0 |        1 |
| tests/test_special_i0e.py                            | test_special_i0e_out                                              |      1 |      0 |       0 |        1 |
| tests/test_addcmul.py                                | test_addcmul                                                      |      1 |      0 |       0 |        1 |
| tests/test_addcmul.py                                | test_addcmul_out_broadcast                                        |      1 |      0 |       0 |        1 |
| tests/test_nonzero.py                                | test_nonzero                                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_large_matrix                                           |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_large_matrix_alpha_beta                                |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_small_square_large                                     |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_large_matrix                                       |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_bounded_workspace                                      |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_fallback_autotuning                                    |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_beta_cache_transitions                          |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_empty                                    |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_out_resize                               |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_beta_zero                                |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_out_validation                           |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight                                 |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_size_inferred                   |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_params                          |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_out                             |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_special_values                  |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_rejects_backward                |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                    | test_hardshrink_backward                                          |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                    | test_hardshrink_backward_special_values                           |      0 |      0 |       1 |        1 |
| tests/test_act_quant.py                              | test_act_quant_triton                                             |      0 |      0 |       1 |        1 |
| tests/test_fp8_paged_mqa_logits_ops.py               | test_accuracy_fp8_paged_mqa_logits                                |      0 |      0 |       1 |        1 |
| tests/test_fp8_paged_mqa_logits_ops.py               | test_accuracy_fp8_paged_mqa_logits_param                          |      0 |      0 |       1 |        1 |
| tests/test_cutlass_scaled_mm.py                      | test_cutlass_scaled_mm                                            |      0 |      0 |       1 |        1 |
| tests/test_upsample_lanczos2d_aa.py                  | test_upsample_lanczos2d_aa_dispatch                               |      0 |      0 |       1 |        1 |
| tests/test_reshape_and_cache_flash.py                | test_reshape_and_cache_flash_vllm_accuracy                        |      0 |      0 |       1 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_alignment_classes                                |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_public_linear_rejects_mismatched_k_before_launch             |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_public_linear_valid_dimensions_match_reference               |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_public_linear_matches_fp64_and_keeps_default_configs         |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_public_linear_registered_aten_path                           |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_tuned_linear_capture_and_two_replays                         |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_parameter_metadata_and_values_preserved                      |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_public_selection_exclusion_branches                          |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_selection_metadata_rejections                                |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_tuned_numerical_boundaries                                   |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                           | test_tuned_nonfinite_inputs                                       |      0 |      0 |       1 |        1 |
| tests/test_native_multi_head_attention.py            | test_native_multi_head_attention_no_weights                       |      0 |      1 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_dispatch                                         |      0 |      1 |       0 |        1 |
| tests/test_fused_adamw_.py                           | test_fused_adamw_optimizer                                        |      0 |      1 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_q_gradient                                      |      0 |      1 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_x_gradient_is_not_implemented                   |      0 |      1 |       0 |        1 |
| tests/test_enable_api.py                             | test_enable                                                       |      0 |      1 |       0 |        1 |
| tests/test_enable_api.py                             | test_only_enable_with_yaml                                        |      0 |      1 |       0 |        1 |
| tests/test_enable_api.py                             | test_only_enable_default                                          |      0 |      1 |       0 |        1 |
| TOTAL                                                |                                                                   |    479 |      8 |      36 |      523 |
|                             filepath                              |                                function                                 | passed | skipped | SUBTOTAL |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_metax_index_offset.py                                  | test_metax_index_large_state_stride                                     |      0 |       1 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_ascend_paths                                        |      0 |       1 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_ascend_sparse_group_fallback                        |      0 |       1 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_ascend_bfloat16_special_values                      |      0 |       1 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_device_index                                     |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_gate_engages                                         |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_gate_defers                                          |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_gate_limits_follow_element_width                     |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax                                                      |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_out                                                  |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_half_to_float                                        |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_neg_inf                                              |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_at_row_cap                                           |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                       | test_rdna4_softmax_non_contiguous                                       |      0 |       1 |        1 |
| tests/flagtune/test_status.py                                     | test_model_loaded_once_per_version                                      |      0 |       1 |        1 |
| tests/flagtune/test_status.py                                     | test_proposer_cache_separates_outer_package_versions                    |      0 |       1 |        1 |
| tests/test_inverse.py                                             | test_inverse_singular                                                   |      0 |       1 |        1 |
| tests/test_where_self_out.py                                      | test_where_uniform_condition_overlapping_output_rejected                |      0 |       1 |        1 |
| tests/test_where_self_out.py                                      | test_where_uniform_condition_graph_reloads_inputs                       |      0 |       1 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_fp8                                 |      0 |       1 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args                                    |      1 |       0 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args_invalid_metadata                   |      1 |       0 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args_invalid_indices                    |      1 |       0 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args_blocks_and_long_rows               |      1 |       0 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args_metadata_parity                    |      1 |       0 |        1 |
| tests/test_validate_sparse_bsr_tensor_args.py                     | test_validate_sparse_bsr_tensor_args_dispatch                           |      1 |       0 |        1 |
| tests/test_gather_block_quantized.py                              | test_gather_block_quantized                                             |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_runs_candidate_once_per_selected_case                    |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_without_override_preserves_gems_dispatch                 |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_failure_cannot_be_recorded_as_passed                     |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_rejects_legacy_and_empty_case_plans                      |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_report_replaces_stale_data                               |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_unknown_or_unexecuted_case_and_all_skip_cannot_pass                |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_preflight_rejects_conflicting_cli_modes                            |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_profile_still_uses_shared_candidate_resolution                     |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_pytest_profile_plugin_dispatch                                     |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_benchmark_times_reference_and_live_override_separately             |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_profile_invokes_kernel_returns_false_for_cpu_only                  |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_profile_invokes_kernel_returns_true_for_device_event               |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_base_without_device_kernel_is_rejected                             |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_base_with_device_kernel_is_accepted                                |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_verify_base_invokes_kernel_caches_result                           |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_verify_base_invokes_kernel_caches_failure                          |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_benchmark_cannot_claim_an_unused_override                          |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_case_invokes_kernel_passes                            |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_case_no_kernel_records_false                          |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_all_cases_run_by_default                              |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_failure_records_error                                 |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_legacy_fallback_invokes_kernel                        |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_legacy_fallback_no_kernel                             |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_rejects_empty_case_plan                               |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_session_passes_if_all_operators_have_kernel           |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_session_fails_if_any_operator_has_no_kernel           |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_session_fails_if_no_kernel                            |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_session_fails_if_no_records                           |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_does_not_downgrade_interrupt                          |      1 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                            | test_kernel_check_rejects_conflicting_cli_modes                         |      1 |       0 |        1 |
| tests/test_arctan2.py                                             | test_arctan2                                                            |      1 |       0 |        1 |
| tests/test_arctan2.py                                             | test_arctan2_                                                           |      1 |       0 |        1 |
| tests/test_arctan2.py                                             | test_arctan2__integer_rejects                                           |      1 |       0 |        1 |
| tests/test_arctan2.py                                             | test_arctan2_special_values                                             |      1 |       0 |        1 |
| tests/test_arctan2.py                                             | test_arctan2_broadcast                                                  |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution                                                        |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_empty_batch                                            |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_noncontiguous_input                                    |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_noncontiguous_weight                                   |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_spatial_rank                                           |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_depthwise                                              |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_transposed                                             |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_bias_is_optional                                       |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_non_positive_groups                                    |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_input_channel_mismatch                                 |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_groups_do_not_divide_channels                          |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_empty_weight                                           |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_bias_shape_mismatch                                    |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_rank_mismatch                                          |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_non_positive_stride                                    |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_non_positive_dilation                                  |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_negative_padding                                       |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_output_padding_too_large                               |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_kernel_larger_than_padded_input                        |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_dilated_kernel_larger_than_input                       |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_rejects_complex                                        |      1 |       0 |        1 |
| tests/test_convolution.py                                         | test_convolution_autograd_input_is_detached                             |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_dtypes                                              |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_cases                                               |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_non_contiguous                                      |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_argument_errors                                     |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_rejects_bool_arguments                              |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_out                                                 |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_cuda_complex                                        |      1 |       0 |        1 |
| tests/test_nonzero_static.py                                      | test_nonzero_static_cuda_paths                                          |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_tile.py                         | test_sparse_semi_structured_tile                                        |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor                                  |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor_float_zero_points                |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor_noncontiguous                    |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor_empty                            |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor_float_input                      |      1 |       0 |        1 |
| tests/test_make_per_channel_quantized_tensor.py                   | test_make_per_channel_quantized_tensor_unsupported_dtype                |      1 |       0 |        1 |
| tests/test__transformer_encoder_layer_fwd.py                      | test__transformer_encoder_layer_fwd                                     |      1 |       0 |        1 |
| tests/test_softmax.py                                             | test_softmax_wide_inner_row                                             |      1 |       0 |        1 |
| tests/test_softmax.py                                             | test_softmax                                                            |      1 |       0 |        1 |
| tests/test_softmax.py                                             | test_softmax_out                                                        |      1 |       0 |        1 |
| tests/test_softmax.py                                             | test_softmax_backward_out                                               |      1 |       0 |        1 |
| tests/test_softmax.py                                             | test_softmax_backward                                                   |      1 |       0 |        1 |
| tests/test_bitwise_and_tensor.py                                  | test_bitwise_and_tensor                                                 |      1 |       0 |        1 |
| tests/test_bitwise_and_tensor.py                                  | test_bitwise_and_tensor_                                                |      1 |       0 |        1 |
| tests/test_narrow_copy_backward.py                                | test_backward                                                           |      1 |       0 |        1 |
| tests/test_narrow_copy_backward.py                                | test_dispatch_and_compute                                               |      1 |       0 |        1 |
| tests/test_narrow_copy_backward.py                                | test_saved_hooks_and_fresh_data                                         |      1 |       0 |        1 |
| tests/test_narrow_copy_backward.py                                | test_nondefault_stream                                                  |      1 |       0 |        1 |
| tests/test_matmuladd.py                                           | test_matmuladd                                                          |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_strided                                                        |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_strided_views                                                  |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_sparse_coo                                                     |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_sparse_compressed                                              |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_nested                                                         |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_matches_reference_table                                        |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_aliases_agree                                                  |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_is_dispatch_free                                               |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_dtype_and_device_independence                                  |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_requires_grad_and_meta                                         |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_arg_validation                                                 |      1 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_with_registration_active                                       |      1 |       0 |        1 |
| tests/test_exponential.py                                         | test_exponential                                                        |      1 |       0 |        1 |
| tests/test_exponential.py                                         | test_exponential_moments                                                |      1 |       0 |        1 |
| tests/test_exponential.py                                         | test_exponential_far_left_tail                                          |      1 |       0 |        1 |
| tests/test_concat_and_cache_mla.py                                | test_concat_and_cache_mla                                               |      1 |       0 |        1 |
| tests/test_relu6.py                                               | test_relu6                                                              |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack                                                          |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_zero_sized_trailing_dims                                 |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_memory_format                                            |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_mixed_dtype_promotion                                    |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out                                                      |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_memory_format                                        |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_non_contiguous                                       |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_resize                                               |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_dtype_cast                                           |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_non_castable_dtype_raises                            |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_device_mismatch_raises                               |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_out_overlapping_input_raises                             |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_shape_mismatch_raises                                    |      1 |       0 |        1 |
| tests/test_row_stack.py                                           | test_row_stack_empty_list_raises                                        |      1 |       0 |        1 |
| tests/test_rwkv_mm_sparsity.py                                    | test_rwkv_mmsparsity                                                    |      1 |       0 |        1 |
| tests/test_silu_and_mul.py                                        | test_silu_and_mul                                                       |      1 |       0 |        1 |
| tests/test_silu_and_mul.py                                        | test_silu_and_mul_out                                                   |      1 |       0 |        1 |
| tests/test_threshold.py                                           | test_threshold                                                          |      1 |       0 |        1 |
| tests/test_threshold.py                                           | test_threshold_backward                                                 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test_linalg_solve_ex                                                    |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test_linalg_solve_ex_batched                                            |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex                                                   |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_batched                                           |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_left_false                                        |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_vector_rhs                                        |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_broadcast_batches                                 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_row_pivoting                                      |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_singular                                          |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_complex                                           |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_non_contiguous                                    |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_dtype_mismatch                                    |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                     | test__linalg_solve_ex_aux_outputs_usable                                |      1 |       0 |        1 |
| tests/test_heaviside.py                                           | test_heaviside_tensor                                                   |      1 |       0 |        1 |
| tests/test_heaviside.py                                           | test_heaviside_out                                                      |      1 |       0 |        1 |
| tests/test_randn_like.py                                          | test_randn_like                                                         |      1 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe                                           |      1 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_hybrid                                    |      1 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_dtype                                     |      1 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_skips_index_validation                    |      1 |       0 |        1 |
| tests/test_unbind_copy.py                                         | test_unbind_copy                                                        |      1 |       0 |        1 |
| tests/test_unbind_copy.py                                         | test_unbind_copy_default_dim                                            |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_infer_size                                       |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_conversion_and_strides                           |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_scalar_values                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_invalid_inputs                                   |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_coalesced                                        |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_dispatch                                         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_registration_scope                               |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_mismatched_devices                               |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_autograd                                         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_pin_memory                                       |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_pin_memory_rejection                             |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_large_reduction                                  |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_size_overflow                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_cpu_transfer                                     |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_invariant_check                                  |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_explicit_size                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_explicit_validation                              |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                   | test_sparse_coo_tensor_explicit_pin_memory                              |      1 |       0 |        1 |
| tests/test_replication_pad2d_backward.py                          | test_replication_pad2d_backward                                         |      1 |       0 |        1 |
| tests/test_replication_pad2d_backward.py                          | test_replication_pad2d_backward_grad_input                              |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack                                                          |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_unpack_data_false                                        |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_unpack_pivots_false                                      |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_batched                                                  |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_out                                                      |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_backward                                                 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_backward_entry                                           |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_backward_batched                                         |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_backward_nondifferentiable_outputs                       |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                           | test_lu_unpack_backward_dispatch                                        |      1 |       0 |        1 |
| tests/test_alpha_dropout_.py                                      | test_alpha_dropout_                                                     |      1 |       0 |        1 |
| tests/test_dequantize.py                                          | test_dequantize                                                         |      1 |       0 |        1 |
| tests/test_kl_div.py                                              | test_kl_div                                                             |      1 |       0 |        1 |
| tests/test_kl_div.py                                              | test_kl_div_zero_target                                                 |      1 |       0 |        1 |
| tests/test_kl_div.py                                              | test_kl_div_broadcast                                                   |      1 |       0 |        1 |
| tests/test_kl_div.py                                              | test_kl_div_functional                                                  |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take                                                               |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_out                                                           |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_negative_index                                                |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_out_of_bounds                                                 |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_noncontiguous                                                 |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_out_noncontiguous                                             |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_out_resizes_empty_tensor                                      |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_out_rejects_mismatched_dtype                                  |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_empty                                                         |      1 |       0 |        1 |
| tests/test_take.py                                                | test_take_multidim_index                                                |      1 |       0 |        1 |
| tests/test_maximum.py                                             | test_maximum                                                            |      1 |       0 |        1 |
| tests/test_maximum.py                                             | test_maximum_cpu_scalar_operand                                         |      1 |       0 |        1 |
| tests/test_maximum.py                                             | test_maximum_cpu_scalar_precision                                       |      1 |       0 |        1 |
| tests/test_maximum.py                                             | test_maximum_rejects_non_scalar_cpu_operand                             |      1 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_status_uses_stderr_and_flush                                       |      1 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_exception_chain_redacts_url_secrets                                |      1 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_status_closed_stream_does_not_fail                                 |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                                    | test_accuracy_linalg_multi_dot                                          |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                                    | test_accuracy_linalg_multi_dot_out                                      |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                                    | test_accuracy_linalg_multi_dot_out_noncontiguous                        |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                                    | test_error_linalg_multi_dot_requires_two_tensors                        |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv                                                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_empty                                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_round_trip                                              |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_row_swap_pivoting                                       |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_batch_independence                                      |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_binds_single_argument                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_invalid_shape                                           |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_rejects_low_precision                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_singular_does_not_hang_or_crash                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_rejects_complex                                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_fp64_preserved                                          |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                          | test_linalg_inv_autograd_input_is_detached                              |      1 |       0 |        1 |
| tests/test_isposinf.py                                            | test_isposinf                                                           |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse                                                            |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_batch                                                      |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_requires_pivoting                                          |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_identity_product                                           |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_noncontiguous                                              |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_empty                                                      |      1 |       0 |        1 |
| tests/test_inverse.py                                             | test_inverse_unsupported_dtype                                          |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward                |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_mask_semantics |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_noncontiguous  |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_empty          |      1 |       0 |        1 |
| tests/test_cummin_helper.py                                       | test_cummin_helper                                                      |      1 |       0 |        1 |
| tests/test_cummin_helper.py                                       | test_cummin_helper_with_nan                                             |      1 |       0 |        1 |
| tests/test_cummin_helper.py                                       | test_cummin_helper_dim                                                  |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                   | test_special_scaled_modified_bessel_k1                                  |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                   | test_special_scaled_modified_bessel_k1_out                              |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                   | test_special_scaled_modified_bessel_k1_edge_cases                       |      1 |       0 |        1 |
| tests/test_eq.py                                                  | test_eq                                                                 |      1 |       0 |        1 |
| tests/test_eq.py                                                  | test_eq_scalar                                                          |      1 |       0 |        1 |
| tests/test_eq.py                                                  | test_eq_integer_precision                                               |      1 |       0 |        1 |
| tests/test_eq.py                                                  | test_eq_scalar_integer_precision                                        |      1 |       0 |        1 |
| tests/test_eq.py                                                  | test_eq_scalar_float_scalar                                             |      1 |       0 |        1 |
| tests/test_cudnn_rnn_backward.py                                  | test_cudnn_rnn_backward                                                 |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py                             | test_native_batch_norm_legit                                            |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py                             | test_native_batch_norm_legit_no_stats                                   |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py                             | test_native_batch_norm_legit_out                                        |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_csc                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_bsc                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_batched                                           |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_wrong_layout                                      |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_out                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_out_non_contiguous                                |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_out_resize                                        |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                                    | test_row_indices_copy_out_dtype_mismatch_raises                         |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_self                                                         |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_self_scalar                                                  |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_self_scalar_other                                            |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_self_out                                                     |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_self_out_cross_device                                        |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_uniform_condition_strided_storage                            |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_uniform_condition_boundaries                                 |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_uniform_condition_scalar                                     |      1 |       0 |        1 |
| tests/test_where_self_out.py                                      | test_where_invalid_broadcast_raises_runtime_error                       |      1 |       0 |        1 |
| tests/test_clamp.py                                               | test_clamp                                                              |      1 |       0 |        1 |
| tests/test_clamp.py                                               | test_clamp_                                                             |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors                            |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_nnz                        |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_aliases_payload            |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_default_uncoalesced        |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_forwards_coalesced         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_out                        |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_registration_names         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py             | test_sparse_coo_tensor_with_dims_and_tensors_rejects_bad_layout         |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_value_range                                 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_wide_values                                 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_accumulate_matches                          |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_stored_layout                               |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_hybrid_coo                                  |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_strided_storage                             |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_special_values                              |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_out                                         |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_out_stored_layout                           |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_dense_self                          |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_dense_mask                          |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_size_mismatch                       |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_sparse_dim_mismatch                 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_invalid_accumulate                  |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_rejects_dense_out                           |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                             | test_sparse_mask_projection_dispatch                                    |      1 |       0 |        1 |
| tests/test_resolve_neg.py                                         | test_accuracy_resolve_neg                                               |      1 |       0 |        1 |
| tests/test_new_ones.py                                            | test_new_ones                                                           |      1 |       0 |        1 |
| tests/test_new_ones.py                                            | test_new_ones_different_size                                            |      1 |       0 |        1 |
| TOTAL                                                             |                                                                         |    293 |      20 |      313 |
