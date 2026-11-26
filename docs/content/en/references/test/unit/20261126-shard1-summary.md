|                             filepath                              |                                 function                                  | passed | skipped | SUBTOTAL |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_xlogy_.py                                              | test_xlogy_                                                               |      1 |       0 |        1 |
| tests/test_xlogy_.py                                              | test_xlogy_special_values_                                                |      1 |       0 |        1 |
| tests/test_xlogy_.py                                              | test_xlogy_tensor_scalar_                                                 |      1 |       0 |        1 |
| tests/test_linalg_cholesky.py                                     | test_linalg_cholesky                                                      |      1 |       0 |        1 |
| tests/test_linalg_cholesky.py                                     | test_linalg_cholesky_upper                                                |      1 |       0 |        1 |
| tests/test_linalg_cholesky.py                                     | test_linalg_cholesky_batch                                                |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward                                       |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_empty_batch                           |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_noncontiguous                         |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_nan_and_inf                           |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_watchdog_path             |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_full_target               |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_duplicate                 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_duplicate_tail            |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_empty_prefix              |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_large_class_bucket_boundaries         |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_many_rows_target_length_reduce        |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_invalid_target_values_raise           |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_invalid_after_distant_sentinel_raises |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_invalid_target_metadata_raises        |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_invalid_input_shape_raises            |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_backward                              |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_backward_dtypes                       |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                      | test_multilabel_margin_loss_forward_backward_single_kernel                |      1 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable.py     | test_scaled_dot_product_fused_attention_overrideable                      |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_strided                                                   |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_sparse_coo                                                |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_sparse_compressed                                         |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_matches_reference_table                                   |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_disjoint_from_native_neighbors                            |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_dispatch_sentinel                                         |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_coexistence                                               |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_bsr_bsc_coverage                                          |      1 |       0 |        1 |
| tests/test_sparse_dim.py                                          | test_sparse_dim_unsupported_layout_parity                                 |      1 |       0 |        1 |
| tests/test_max_pool1d.py                                          | test_max_pool1d                                                           |      1 |       0 |        1 |
| tests/test_max_pool1d.py                                          | test_max_pool1d_default_stride                                            |      1 |       0 |        1 |
| tests/test_linalg_solve.py                                        | test_linalg_solve                                                         |      1 |       0 |        1 |
| tests/test_linalg_solve.py                                        | test_linalg_solve_batched                                                 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_formula_matches_aten_cpu             |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_strided_grad_output                  |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_odd_class_count                      |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_hinge_uses_input_dtype               |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward                                      |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_target_patterns                      |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_noncontiguous                        |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_class_count                          |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_extra_reduction_values               |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_special_values                       |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_empty_batch                          |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_scalar_grad_output                   |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_half_dtypes                          |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_invalid_target_dtype                 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_invalid_is_target_dtype              |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_invalid_target_shape                 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                     | test_multilabel_margin_loss_backward_invalid_reduction                    |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_src                                                          |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_reduce_add                                                   |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_add_                                                         |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_add_2d_noncontiguous                                         |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_add                                                          |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_add_2d                                                       |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_reduce_multiply                                              |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_src_                                                         |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_reduce_add_                                                  |      1 |       0 |        1 |
| tests/test_scatter.py                                             | test_scatter_reduce_multiply_                                             |      1 |       0 |        1 |
| tests/test_triu.py                                                | test_triu                                                                 |      1 |       0 |        1 |
| tests/test_triu.py                                                | test_triu_noncontiguous                                                   |      1 |       0 |        1 |
| tests/test_triu.py                                                | test_triu_                                                                |      1 |       0 |        1 |
| tests/test_triu.py                                                | test_triu_inplace_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_expand.py                                              | test_expand                                                               |      1 |       0 |        1 |
| tests/test_expand.py                                              | test_expand_                                                              |      1 |       0 |        1 |
| tests/test_expand.py                                              | test_expand_invalid_sizes                                                 |      1 |       0 |        1 |
| tests/test_expand.py                                              | test_expand_zero_size_singleton_stride                                    |      1 |       0 |        1 |
| tests/test_cummin.py                                              | test_cummin                                                               |      1 |       0 |        1 |
| tests/test_cummin.py                                              | test_cummin_with_nan                                                      |      1 |       0 |        1 |
| tests/test_acos_.py                                               | test_acos_                                                                |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward                                                 |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_layout                                          |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_empty_batch                                     |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_empty_channel                                   |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_invalid                                         |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_deterministic                                   |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_out                                             |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_large_reduction                                 |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_fp64_precision                                  |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_64bit_address                                   |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_out_strided                                     |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_out_input_alias                                 |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_invalid_tensors                                 |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_spatial_boundaries                              |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_repeated_input_alias                            |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_channel_tiles                                   |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_large_dilation                                  |      1 |       0 |        1 |
| tests/test_convolution_backward.py                                | test_convolution_backward_fresh_buffers                                   |      1 |       0 |        1 |
| tests/test_tensor_wrapper.py                                      | test_typed_pointer                                                        |      1 |       0 |        1 |
| tests/test_tensor_wrapper.py                                      | test_typed_pointer_reinterpret_with_offset                                |      1 |       0 |        1 |
| tests/test_tensor_wrapper.py                                      | test_typed_pointer_as_is                                                  |      1 |       0 |        1 |
| tests/test_tensor_wrapper.py                                      | test_strided_buffer_slice                                                 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                       | test_rowwise_prune                                                        |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                       | test_rowwise_prune_all_kept                                               |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                       | test_rowwise_prune_none_kept                                              |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p                                                                |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_                                                               |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward                                                       |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward_expanded_grad                                         |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward_special_values                                        |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward_matches_aten_rounding                                 |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward_via_torch_dispatch                                    |      1 |       0 |        1 |
| tests/test_log1p.py                                               | test_log1p_backward_fused_kernel_count                                    |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add                                                                  |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_complex                                                          |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_                                                                 |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_tensor_scalar                                                    |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_tensor_scalar_                                                   |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_scalar_tensor                                                    |      1 |       0 |        1 |
| tests/test_add.py                                                 | test_add_scalar_scalar                                                    |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_accuracy_fake_quantize_per_tensor_affine_cachemask                   |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_accuracy_fake_quantize_per_tensor_affine_cachemask_boundaries        |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_accuracy_fake_quantize_per_tensor_affine_cachemask_out               |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_accuracy_fake_quantize_per_tensor_affine_cachemask_noncontiguous     |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_accuracy_fake_quantize_per_tensor_affine_cachemask_empty             |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask.py           | test_fake_quantize_per_tensor_affine_cachemask_invalid_qparams            |      1 |       0 |        1 |
| tests/test_dense_dim.py                                           | test_dense_dim_strided                                                    |      1 |       0 |        1 |
| tests/test_dense_dim.py                                           | test_dense_dim_sparse_coo                                                 |      1 |       0 |        1 |
| tests/test_dense_dim.py                                           | test_dense_dim_sparse_compressed                                          |      1 |       0 |        1 |
| tests/test_dense_dim.py                                           | test_dense_dim_matches_reference_table                                    |      1 |       0 |        1 |
| tests/test_dense_dim.py                                           | test_dense_dim_disjoint_from_native_neighbors                             |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide                                                          |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_                                                         |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_out                                                      |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_tensor_scalar                                            |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_tensor_scalar_                                           |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_scalar_tensor                                            |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_tensor_dispatch                                          |      1 |       0 |        1 |
| tests/test_true_divide.py                                         | test_true_divide_tensor_inplace_dispatch                                  |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default                                                   |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default_empty                                             |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default_uncoalesced                                       |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dtype                                                     |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim                                                       |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_all                                                   |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_dtype                                                 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_out                                                   |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_negative                                              |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_keeps_structural_zeros                                |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_hybrid_dense_dims                                         |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_uncoalesced_input                                         |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_zero_size_dims                                            |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_integer_promotion                                         |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_duplicate_dim_raises                                      |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_out_of_range_dim_raises                                   |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dense_input_raises                                        |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_csr_input_raises                                          |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_out_dtype_mismatch_raises                             |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dtype_accumulator_precision                               |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_complex                                                   |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_dispatch_reaches_gems_impl                                           |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_dispatch_covers_every_overload                                       |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_uncoalesced_matches_aten                                  |      1 |       0 |        1 |
| tests/test_rand_like.py                                           | test_rand_like                                                            |      1 |       0 |        1 |
| tests/test_nll_loss_backward.py                                   | test_nll_loss_backward                                                    |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float                                                           |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_strided                                                   |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_special_values                                            |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_integer_limits                                            |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_autograd                                                  |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_complex                                                   |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_negative_view                                             |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_cpu                                                       |      1 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_sparse                                                    |      1 |       0 |        1 |
| tests/test_all.py                                                 | test_all                                                                  |      1 |       0 |        1 |
| tests/test_all.py                                                 | test_all_dim                                                              |      1 |       0 |        1 |
| tests/test_all.py                                                 | test_all_dims                                                             |      1 |       0 |        1 |
| tests/test_eye.py                                                 | test_eye                                                                  |      1 |       0 |        1 |
| tests/test_eye.py                                                 | test_eye_m                                                                |      1 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                     | test_grid_sampler_2d                                                      |      1 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                     | test_grid_sampler_2d_oh_ne_iw                                             |      1 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                     | test_grid_sampler_2d_nearest_half_integer                                 |      1 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                     | test_grid_sampler_2d_dim_one_reflection                                   |      1 |       0 |        1 |
| tests/test_isin.py                                                | test_accuracy_isin                                                        |      1 |       0 |        1 |
| tests/test_isin.py                                                | test_accuracy_isin_scalar_tensor                                          |      1 |       0 |        1 |
| tests/test_isin.py                                                | test_accuracy_isin_tensor_scalar                                          |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_posdim                                                 |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_posdim_with_nan                                        |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_posdim_bool                                            |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_negdim                                                 |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_negdim_with_nan                                        |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_negdim_bool                                            |      1 |       0 |        1 |
| tests/test_cummax_helper.py                                       | test_cummax_helper_dim0                                                   |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell                                                    |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_non_contiguous                                     |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_zero_sized                                         |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_invalid_input                                      |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_complex_rejected                                   |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_autograd_graph                                     |      1 |       0 |        1 |
| tests/test_gru_cell.py                                            | test_accuracy_gru_cell_keyword_arguments                                  |      1 |       0 |        1 |
| tests/test_slice_scatter.py                                       | test_slice_scatter                                                        |      1 |       0 |        1 |
| tests/test_slice_scatter.py                                       | testslice_scatter_with_self_overlapping_input                             |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py | test_fake_quantize_learnable_per_channel_affine_backward                  |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py | test_fake_quantize_learnable_per_channel_affine_backward_extreme          |      1 |       0 |        1 |
| tests/test_lstm.py                                                | test_lstm                                                                 |      1 |       0 |        1 |
| tests/test_copysign.py                                            | test_copysign                                                             |      1 |       0 |        1 |
| tests/test_copysign.py                                            | test_copysign_out                                                         |      1 |       0 |        1 |
| tests/test_copysign.py                                            | test_copysign_byte                                                        |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce                                                             |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_large_key_space                                             |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_coalesced_input                                             |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_value_ranges                                                |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_nan_inf                                                     |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_rejects_dense_input                                         |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_rejects_csr_input                                           |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test_coalesce_rejects_fp8_input                                           |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce                                                            |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_value_ranges                                               |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_nan_inf                                                    |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_out                                                        |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_dense_dims                                                 |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_out_dense_dims                                             |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_rejects_coalesced_input                                    |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_out_rejects_wrong_dtype_and_shape                          |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_rejects_dense_input                                        |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_rejects_csr_input                                          |      1 |       0 |        1 |
| tests/test_coalesce.py                                            | test__coalesce_rejects_fp8_input                                          |      1 |       0 |        1 |
| tests/test_atanh_.py                                              | test_atanh_                                                               |      1 |       0 |        1 |
| tests/test_linalg_householder_product.py                          | test_linalg_householder_product                                           |      1 |       0 |        1 |
| tests/test_linalg_householder_product.py                          | test_linalg_householder_product_batched                                   |      1 |       0 |        1 |
| tests/test_special_hermite_polynomial_h.py                        | test_special_hermite_polynomial_h                                         |      1 |       0 |        1 |
| tests/test_special_hermite_polynomial_h.py                        | test_special_hermite_polynomial_h_scalar                                  |      1 |       0 |        1 |
| tests/test_special_hermite_polynomial_h.py                        | test_special_hermite_polynomial_h_out_of_range                            |      1 |       0 |        1 |
| tests/test_fft.py                                                 | test_fft                                                                  |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d.py                            | test_accuracy__upsample_nearest_exact1d                                   |      1 |       0 |        1 |
| tests/test_convolution_overrideable.py                            | test_convolution_overrideable                                             |      1 |       0 |        1 |
| tests/test_convolution_overrideable.py                            | test_convolution_overrideable_transposed                                  |      1 |       0 |        1 |
| tests/test_convolution_overrideable.py                            | test_convolution_overrideable_out                                         |      1 |       0 |        1 |
| tests/test_mse_loss.py                                            | test_mse_loss                                                             |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_real_dtypes                                                  |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_complex_dtypes                                               |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_0d                                                           |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_1d_matches_native_error                                      |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_non_contiguous                                               |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_mutation_through_view                                        |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_conjugate_bit_input                                          |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_dispatch_stability                                           |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_matches_native_view_table                                    |      1 |       0 |        1 |
| tests/test_adjoint.py                                             | test_adjoint_diagnostics_parity                                           |      1 |       0 |        1 |
| tests/test_moe_sum.py                                             | test_moe_sum                                                              |      1 |       0 |        1 |
| tests/test_randn.py                                               | test_randn                                                                |      1 |       0 |        1 |
| tests/test_randn.py                                               | test_randn_large_output_offsets                                           |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bin_ct                                            |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bin_ct_with_range                                 |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bin_ct_with_weight                                |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bin_ct_with_density                               |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bins_tensor                                       |      1 |       0 |        1 |
| tests/test_histogram.py                                           | test_accuracy_histogram_bins_tensor_nonuniform                            |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine.py                    | test_accuracy_fake_quantize_per_channel_affine                            |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine.py                    | test_accuracy_fake_quantize_per_channel_affine_multi_dim                  |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine.py                    | test_accuracy_fake_quantize_per_channel_affine_half_to_even               |      1 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine.py                    | test_accuracy_fake_quantize_per_channel_affine_empty                      |      1 |       0 |        1 |
| tests/test__linalg_slogdet.py                                     | test__linalg_slogdet                                                      |      1 |       0 |        1 |
| tests/test_assert_async.py                                        | test_assert_async                                                         |      1 |       0 |        1 |
| tests/test_special_softmax.py                                     | test_special_softmax                                                      |      1 |       0 |        1 |
| tests/test_special_softmax.py                                     | test_special_softmax_with_dtype                                           |      1 |       0 |        1 |
| tests/test_cumsum.py                                              | test_cumsum                                                               |      1 |       0 |        1 |
| tests/test_cumsum.py                                              | test_cumsum_empty                                                         |      1 |       0 |        1 |
| tests/test_cumsum.py                                              | test_cumsum_out                                                           |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa                                                |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_uint8_and_layout                               |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_transposed_input                               |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_out                                            |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_vec                                            |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_vec_requires_one_size                          |      1 |       0 |        1 |
| tests/test_special_digamma.py                                     | test_special_digamma_large                                                |      1 |       0 |        1 |
| tests/test_special_digamma.py                                     | test_special_digamma_small_positive                                       |      1 |       0 |        1 |
| tests/test_special_digamma.py                                     | test_special_digamma_negative                                             |      1 |       0 |        1 |
| tests/test_special_digamma.py                                     | test_special_digamma_mid                                                  |      1 |       0 |        1 |
| tests/test_resolve_conj.py                                        | test_resolve_conj                                                         |      1 |       0 |        1 |
| tests/test_normed_cumsum.py                                       | test_normed_cumsum                                                        |      1 |       0 |        1 |
| tests/test_normed_cumsum.py                                       | test_normed_cumsum_output_is_normalized                                   |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H                                                             |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_strided_input                                               |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_conjugate_state                                             |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_write_through                                               |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_0d_scalar                                                   |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_backward                                                    |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_registered_dispatch_preserves_lazy_conj_and_backward        |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_special_values                                              |      1 |       0 |        1 |
| tests/test_matrix_H.py                                            | test_matrix_H_rejects_non_matrix                                          |      1 |       0 |        1 |
| tests/test_kthvalue.py                                            | test_kthvalue                                                             |      1 |       0 |        1 |
| tests/test_kthvalue.py                                            | test_kthvalue_default_dim                                                 |      1 |       0 |        1 |
| tests/test_kthvalue.py                                            | test_kthvalue_invalid_dim                                                 |      1 |       0 |        1 |
| tests/test_kthvalue.py                                            | test_kthvalue_empty_tensor                                                |      1 |       0 |        1 |
| tests/test_affine_grid_generator.py                               | test_affine_grid_generator                                                |      1 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf                                                                |      1 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_non_contiguous                                                 |      1 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_return_structure                                               |      1 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_zero_and_tiny_sizes                                            |      1 |       0 |        1 |
| tests/test_fused_adam.py                                          | test_fused_adam                                                           |      1 |       0 |        1 |
| tests/test_fused_adam.py                                          | test_fused_adam_                                                          |      1 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_accuracy_diag_embed                                                  |      1 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward                                                  |      1 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward_default_dims                                     |      1 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward_non_contiguous                                   |      1 |       0 |        1 |
| tests/test_te_rmsnorm.py                                          | test_te_rmsnorm_fwd                                                       |      1 |       0 |        1 |
| tests/test_te_rmsnorm.py                                          | test_te_rmsnorm_fwd_n1                                                    |      1 |       0 |        1 |
| tests/test_te_rmsnorm.py                                          | test_te_rmsnorm_fwd_vs_te                                                 |      1 |       0 |        1 |
| tests/test_te_rmsnorm.py                                          | test_te_rmsnorm_bwd                                                       |      1 |       0 |        1 |
| tests/test_te_rmsnorm.py                                          | test_te_rmsnorm_bwd_vs_te                                                 |      1 |       0 |        1 |
| tests/test_softshrink.py                                          | test_softshrink                                                           |      1 |       0 |        1 |
| tests/test_softshrink.py                                          | test_softshrink_out                                                       |      1 |       0 |        1 |
| tests/test_quantized_lstm.py                                      | test_accuracy_quantized_lstm                                              |      1 |       0 |        1 |
| tests/test_quantized_lstm.py                                      | test_accuracy_quantized_lstm_bidirectional                                |      1 |       0 |        1 |
| tests/test_quantized_lstm.py                                      | test_accuracy_quantized_lstm_no_bias                                      |      1 |       0 |        1 |
| tests/test_dgeglu.py                                              | test_dgeglu                                                               |      1 |       0 |        1 |
| tests/test_bitwise_right_shift_.py                                | test_bitwise_right_shift_                                                 |      1 |       0 |        1 |
| tests/test_adaptive_avg_pool2d_backward.py                        | test_adaptive_avg_pool2d_backward                                         |      1 |       0 |        1 |
| tests/test_gru.py                                                 | test_gru                                                                  |      1 |       0 |        1 |
| tests/test_gru.py                                                 | test_gru_data                                                             |      1 |       0 |        1 |
| tests/test_linalg_slogdet.py                                      | test_linalg_slogdet                                                       |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear                                        |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear_with_bias                              |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear_n_ne_k                                 |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear_activation                             |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear_with_bias_shapes                       |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py                       | test_sparse_semi_structured_linear_non_contiguous_weight                  |      1 |       0 |        1 |
| tests/test_histogramdd.py                                         | test_histogramdd_basic                                                    |      1 |       0 |        1 |
| tests/test_histogramdd.py                                         | test_histogramdd_with_range                                               |      1 |       0 |        1 |
| tests/test_histogramdd.py                                         | test_histogramdd_density                                                  |      1 |       0 |        1 |
| tests/test_histogramdd.py                                         | test_histogramdd_empty                                                    |      1 |       0 |        1 |
| tests/test_histogramdd.py                                         | test_histogramdd_degenerate_range                                         |      1 |       0 |        1 |
| tests/test_nll_loss2d_backward.py                                 | test_nll_loss2d_backward                                                  |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward                                             |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_pos_rejected                        |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_neg_degrades_to_zero                |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_attn_bias_shapes                            |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dual_dim_bias                               |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_onesided_cumseq_rejected                    |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_varlen_rejected                             |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_value_head_dim                              |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander                                                               |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_non_contiguous                                                |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_int                                                           |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_int64_large                                                   |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_negative_values                                               |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_fp64                                                          |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_complex                                                       |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_zero_n                                                        |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_negative_n                                                    |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_invalid_rank                                                  |      1 |       0 |        1 |
| tests/test_vander.py                                              | test_vander_output_dtype                                                  |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_w8a8_config_dtype                                                    |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_other_moe_config_dtypes_unchanged                                    |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_w8a8_flag_reaches_config_lookup                                      |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_w8a8_excludes_plain_half_gemm_config                                 |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_plain_half_gemm_optimization_remains_enabled                         |      1 |       0 |        1 |
| tests/test_fused_moe.py                                           | test_w8a8_large_m_accuracy                                                |      1 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py                       | test_binary_cross_entropy_backward                                        |      1 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py                       | test_binary_cross_entropy_backward_weighted                               |      1 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth                                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_noncontiguous_and_lower_triangle                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_invalid_input                                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_out                                              |      1 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_out_resize                                       |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                          | test_linalg_eig                                                           |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                          | test_linalg_eig_batched                                                   |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy                                                          |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out                                                      |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_values_view                                              |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out_into_offset_buffer                                   |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out_into_strided_view                                    |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_special_values                                           |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out_special_values                                       |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_backward                                                 |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_rejects_dense_input                                      |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_rejects_vendor_unsupported_layout                        |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_rejects_non_tensor_input                                 |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out_rejects_mismatched_dtype                             |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test__values_copy                                                         |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test__values_copy_empty                                                   |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test__values_copy_out                                                     |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_noncontiguous_stored_values                              |      1 |       0 |        1 |
| tests/test_values_copy.py                                         | test_values_copy_out_noncontiguous_stored_values                          |      1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_post_vs_ref                                                      |      1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_split_sinkhorn                                                   |      1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_pre_vs_ref                                                       |      1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_bwd_vs_ref                                                       |      1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_hc_head_fused_kernel_vs_ref                                          |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                   | test_scaled_grouped_mm                                                    |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                   | test_scaled_grouped_mm_fp8_core                                           |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                   | test_scaled_grouped_mm_fp8_encodings                                      |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                   | test_scaled_grouped_mm_scale_before_output_cast                           |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                   | test_scaled_grouped_mm_reduction_addressing                               |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_accuracy                                                        |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_broadcast                                                       |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_non_contiguous                                                  |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_negative_p                                                      |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_fp64                                                            |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_empty                                                           |      1 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_empty_raises                                                    |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as                                                           |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_non_contiguous                                            |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_view_strided                                              |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_materialized_copy                                         |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_special                                                   |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_backward                                                  |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_other_has_no_gradient                                     |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_complex_copy                                              |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_dispatch                                                  |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_tile_kernel                                               |      1 |       0 |        1 |
| tests/test_reshape_as.py                                          | test_reshape_as_tile_plan_rejects_inexpressible                           |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_index_layouts                                       |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_index_layouts                                   |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy                                                     |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out                                                 |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_spec_shapes                                         |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_spec_shapes                                     |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_value_ranges                                        |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_value_ranges                                    |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_empty_bsr                                           |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_empty_bsr                                       |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_unchecked_uncoalesced                               |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_unchecked_uncoalesced                           |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_nan_inf_values                                      |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_negative_dense                                      |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_negative_csc                                        |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_negative_coo                                        |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_negative_non_tensor                                 |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_negative_dense                                  |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_negative_coo                                    |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_negative_wrong_dtype                            |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_noncontiguous_source                                |      1 |       0 |        1 |
| tests/test_col_indices_copy.py                                    | test_col_indices_copy_out_noncontiguous_out                               |      1 |       0 |        1 |
| tests/test_embedding_bag_dense_backward.py                        | test_embedding_bag_dense_backward                                         |      1 |       0 |        1 |
| tests/test_embedding_bag_dense_backward.py                        | test_embedding_bag_dense_backward_with_weights                            |      1 |       0 |        1 |
| tests/test_embedding_bag_dense_backward.py                        | test_embedding_bag_dense_backward_mode_sum                                |      1 |       0 |        1 |
| tests/test_sspaddmm.py                                            | test_sspaddmm                                                             |      1 |       0 |        1 |
| tests/test_sspaddmm.py                                            | test_sspaddmm_out                                                         |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_mm.py                           | test_sparse_semi_structured_mm                                            |      1 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_tensor                                                          |      1 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_scalar                                                          |      1 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                             | test_max_pool3d_with_indices                                              |      1 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                             | test_max_pool3d_with_indices_ties                                         |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos                                                                 |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward                                                        |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward_registered_dispatch                                    |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward_special_values                                         |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward_noncontiguous                                          |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward_empty                                                  |      1 |       0 |        1 |
| tests/test_acos.py                                                | test_acos_backward_fused_kernel_count                                     |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy                                             |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_default_storage_offset                      |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_out_noncontiguous                           |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_out_resizes                                 |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_out_aliases_input                           |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_out_dtype_mismatch_raises                   |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_float8_byte_path                            |      1 |       0 |        1 |
| tests/test_as_strided_copy.py                                     | test_accuracy_as_strided_copy_out_float8_byte_path                        |      1 |       0 |        1 |
| tests/test_igamma.py                                              | test_igamma                                                               |      1 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                              | test_wrapped_linear_prepack                                               |      1 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                              | test_wrapped_linear_prepack_rounds_ties_to_even_and_clamps                |      1 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                              | test_wrapped_linear_prepack_rejects_invalid_shapes                        |      1 |       0 |        1 |
| tests/test_conv2d.py                                              | test_conv2d                                                               |      1 |       0 |        1 |
| tests/test_conv2d.py                                              | test_conv2d_padding                                                       |      1 |       0 |        1 |
| tests/test_silu_and_mul_with_clamp.py                             | test_silu_and_mul_with_clamp                                              |      1 |       0 |        1 |
| tests/test_silu_and_mul_with_clamp.py                             | test_silu_and_mul_with_clamp_out                                          |      1 |       0 |        1 |
| tests/test__adaptive_avg_pool3d_backward.py                       | test_adaptive_avg_pool3d_backward                                         |      1 |       0 |        1 |
| tests/test_all.py                                                 | test_all_reduction_limits                                                 |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_scalar_and_tail_accuracy                                         |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_grid_stride_accuracy                                             |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_stage_one_every_tile_accuracy                                    |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_stage_two_multitile_accuracy                                     |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_real_core_dim_and_ub_limits                                      |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_noncontiguous_accuracy                                           |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_empty_identity_without_launch                                    |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_floating_truth_accuracy                                          |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_dim_and_dims_exact_accuracy                                      |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_dim_nested_loops_accuracy                                        |      0 |       1 |        1 |
| tests/test_all.py                                                 | test_all_dim_tail_masks_review                                            |      0 |       1 |        1 |
| tests/test_upsample_lanczos2d_aa.py                               | test_upsample_lanczos2d_aa_dispatch                                       |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_w8a16_fp8                                                       |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_w8a16_fp8_grouped_scale_nvidia                                  |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_w8a16_fp8_row_scale                                             |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_edges                                                       |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_all_encodings                                               |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_empty                                                       |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_strides_graph_and_ties                                      |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_validation                                                  |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_partition_tails                                             |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_single_group_scales                                         |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_ascend                                                      |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_edges_ascend                                                |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_maximum_row_ascend                                          |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_all_finite_encodings_ascend                                 |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_graph_replay_changed_inputs_ascend                          |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_large_cutoff_ties_and_zero_scale_ascend                     |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_unaligned_storage_and_current_stream_ascend                 |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_large_k_ascend                                              |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_over_sort_capacity_ascend                                   |      0 |       1 |        1 |
| tests/test_topk_w8a16_fp8.py                                      | test_topk_fp8_output_dtype_and_strides_ascend                             |      0 |       1 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_scale                                       |      0 |       1 |        1 |
| tests/test_mhc_ops.py                                             | test_hc_head_fused_kernel_vs_vllm                                         |      0 |       1 |        1 |
| tests/test_metax_index_put_large_stride.py                        | test_index_put_large_paged_stride_and_graph                               |      0 |       1 |        1 |
| tests/test_metax_index_put_large_stride.py                        | test_index_put_transposed_and_accumulate                                  |      0 |       1 |        1 |
| tests/test_fused_indexer_q_rope_quant.py                          | test_fused_indexer_q_rope_quant                                           |      0 |       1 |        1 |
| TOTAL                                                             |                                                                           |    465 |      38 |      503 |
|                         filepath                          |                                function                                | passed | failed | skipped | SUBTOTAL |
| --------------------------------------------------------- | ---------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_linalg_vecdot.py                               | test_linalg_vecdot                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_vecdot.py                               | test_linalg_vecdot_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_gt_scalar_.py                                  | test_gt_scalar_                                                        |      1 |      0 |       0 |        1 |
| tests/test_gt_scalar_.py                                  | test_gt_scalar__byte                                                   |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid_backward.py                        | test_hardsigmoid_backward                                              |      1 |      0 |       0 |        1 |
| tests/core/test_ref_as_candidate_benchmark.py             | test_reference_is_timed_on_both_sides                                  |      1 |      0 |       0 |        1 |
| tests/core/test_ref_as_candidate_benchmark.py             | test_missing_reference_is_an_error                                     |      1 |      0 |       0 |        1 |
| tests/core/test_ref_as_candidate_benchmark.py             | test_custom_execution_must_explicitly_support_contract                 |      1 |      0 |       0 |        1 |
| tests/core/test_ref_as_candidate_benchmark.py             | test_candidate_timer_cannot_be_omitted                                 |      1 |      0 |       0 |        1 |
| tests/core/test_ref_as_candidate_benchmark.py             | test_timer_must_actually_invoke_reference                              |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_benchmark_runs_original_baseline_once                             |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_backward_reference_uses_original_grad_semantics                   |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_skip_native_is_not_a_pass_or_failure                              |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_failure_is_not_passed                                   |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_capability_failures_are_recorded_individually_and_later_cases_run |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_failure_categories_do_not_guess_api_or_dtype_support              |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_capability_failure_does_not_continue_after_sync_failure           |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_interrupt_propagates_and_leaves_unexecuted_cases        |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_capability_failure_before_invocation_keeps_zero_call_count        |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_source_skip_preserves_node_semantics_and_unexecuted_records       |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_unsupported_custom_runner_cannot_execute_candidate                |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_conflicting_modes_are_rejected                                    |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_report_replaces_stale_data_and_preserves_case_skips     |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_unknown_case_selection_fails                            |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness  |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_mode_cannot_run_correctness_pytest                      |      1 |      0 |       0 |        1 |
| tests/core/test_reference_only.py                         | test_reference_only_defaults_to_core_without_changing_normal_defaults  |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                             | test_constant_pad_nd                                                   |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                             | test_constant_pad_nd_non_contiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                             | test_constant_pad_nd_zero_value                                        |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                             | test_constant_pad_nd_partial_dims                                      |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy                                              |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_noncontiguous_input                          |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_strided_input                                |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_broadcast_input                              |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_storage_offset                               |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_independent_storage                          |      1 |      0 |       0 |        1 |
| tests/test_select_copy.py                                 | test_accuracy_select_copy_error_paths                                  |      1 |      0 |       0 |        1 |
| tests/test_weight_norm.py                                 | test_weight_norm                                                       |      1 |      0 |       0 |        1 |
| tests/test_special_i1.py                                  | test_special_i1                                                        |      1 |      0 |       0 |        1 |
| tests/test_special_i1.py                                  | test_special_i1_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d                                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_quant_params                                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_qint_dtypes                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_qint_dtypes                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_default_stride                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_default_stride                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_invalid_params                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_invalid_params                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_invalid_rank                                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_zero_sized_dims                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_empty_batch                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_rejects_non_quantized                        |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_rejects_per_channel                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_resizes                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_adopts_input_qparams                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_dtype_mismatch                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_preserves_storage_offset                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_out_non_contiguous                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                        | test_quantized_max_pool3d_non_contiguous_input                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_sign_special_values.py                 | test_sign_special_values                                               |      1 |      0 |       0 |        1 |
| tests/test_matmuladd.py                                   | test_matmuladd                                                         |      1 |      0 |       0 |        1 |
| tests/test_logaddexp.py                                   | test_logaddexp                                                         |      1 |      0 |       0 |        1 |
| tests/test_logaddexp.py                                   | test_logaddexp_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm                                                  |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_layout                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_fp64_inference_variance                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_out                                              |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_adversarial                                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_invalid                                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_autograd_after_registration                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_singleton                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_parameter_shape                                  |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_fused_boundary                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_out_invalid                                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_out_registered                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_singleton_stride                                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_out_inference_placeholders                       |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_fast_path_boundaries                             |      1 |      0 |       0 |        1 |
| tests/test_cudnn_batch_norm.py                            | test_cudnn_batch_norm_out_fast_path                                    |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_rnn_layer.py                            | test_mkldnn_rnn_layer                                                  |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_rnn_layer.py                            | test_mkldnn_rnn_layer_direct_wrapper                                   |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_rnn_layer.py                            | test_mkldnn_rnn_layer_direct_backward                                  |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky                                                          |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_upper                                                    |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_batch                                                    |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_non_contiguous                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_non_contiguous_batch                                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_batched_4d                                               |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_not_positive_definite_raises                             |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_empty_batch                                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_non_square_raises                                        |      1 |      0 |       0 |        1 |
| tests/test_cholesky.py                                    | test_cholesky_large_n_fallback                                         |      1 |      0 |       0 |        1 |
| tests/test_functional_sym_constrain_range_for_size.py     | test_functional_sym_constrain_range_for_size                           |      1 |      0 |       0 |        1 |
| tests/test_signbit.py                                     | test_signbit                                                           |      1 |      0 |       0 |        1 |
| tests/test_signbit.py                                     | test_signbit_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_special_bessel_j1.py                           | test_special_bessel_j1                                                 |      1 |      0 |       0 |        1 |
| tests/test_istft.py                                       | test_istft_roundtrip                                                   |      1 |      0 |       0 |        1 |
| tests/test_istft.py                                       | test_istft_1d_signal                                                   |      1 |      0 |       0 |        1 |
| tests/test_istft.py                                       | test_istft_variants                                                    |      1 |      0 |       0 |        1 |
| tests/test_istft.py                                       | test_istft_center_false                                                |      1 |      0 |       0 |        1 |
| tests/test_istft.py                                       | test_istft_errors                                                      |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward                                       |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_grad_input                            |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_layout                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_impulse                               |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_resize                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_invalid                               |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_out_error                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_deterministic                         |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_nonfinite_cpu_reference               |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_fp64_opmath                           |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_extreme_positive_scale                |      1 |      0 |       0 |        1 |
| tests/test_log_normal_.py                                 | test_log_normal_                                                       |      1 |      0 |       0 |        1 |
| tests/test_rrelu_.py                                      | test_rrelu_                                                            |      1 |      0 |       0 |        1 |
| tests/test_rrelu_.py                                      | test_rrelu__custom_bounds                                              |      1 |      0 |       0 |        1 |
| tests/test_rrelu_.py                                      | test_rrelu__inplace                                                    |      1 |      0 |       0 |        1 |
| tests/test_rrelu_.py                                      | test_rrelu__training                                                   |      1 |      0 |       0 |        1 |
| tests/test_rrelu_.py                                      | test_rrelu__non_contiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_softplus.py                                    | test_softplus                                                          |      1 |      0 |       0 |        1 |
| tests/test_softplus.py                                    | test_softplus_backward                                                 |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                | test_column_stack                                                      |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                | test_column_stack_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                | test_exception_column_stack                                            |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p                                                     |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p_non_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p_negative                                            |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p_nan_inf                                             |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                               | test_special_log1p_small_values                                        |      1 |      0 |       0 |        1 |
| tests/test_select_backward.py                             | test_select_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_select_backward.py                             | test_select_backward_non_contiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_select_backward.py                             | test_select_backward_small_and_edge                                    |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                        | test_mish                                                              |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                        | test_mish_                                                             |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                        | test_mish_backward                                                     |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_csr                                          |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_bsr                                          |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_mutation_through_view                        |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_unsupported_layouts                          |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_dispatch_stability                           |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_dispatch_sentinel                            |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_int32_indices                                |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_batched                                      |      1 |      0 |       0 |        1 |
| tests/test_col_indices.py                                 | test_accuracy_col_indices_empty_csr                                    |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py   | test_fused_recurrent_gated_delta_rule_fwd_accuracy                     |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh                                                          |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh_explicit                                                 |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh_out_explicit                                             |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh_special_values                                           |      1 |      0 |       0 |        1 |
| tests/test_hardtanh.py                                    | test_hardtanh_out_special_values                                       |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_aa.py                       | test_upsample_bicubic2d_aa                                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d_aa.py                       | test_upsample_bicubic2d_aa_backward                                    |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_lengths                                            |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_offsets                                            |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_backward_lengths                                   |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_lengths_out                                        |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_offsets_out                                        |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_backward_lengths_out                               |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_backward_offsets_out                               |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_backward_offsets                                   |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                              | test_segment_reduce_indices_unsupported                                |      1 |      0 |       0 |        1 |
| tests/test_ne.py                                          | test_ne                                                                |      1 |      0 |       0 |        1 |
| tests/test_ne.py                                          | test_ne_scalar                                                         |      1 |      0 |       0 |        1 |
| tests/flagtune/test_official_triton_benchmark_protocol.py | test_official_triton_replay_uses_fixed_retry_count                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_official_triton_benchmark_protocol.py | test_official_triton_vendor_backend_uses_graph_replay                  |      1 |      0 |       0 |        1 |
| tests/flagtune/test_official_triton_benchmark_protocol.py | test_official_triton_unsupported_replay_falls_back_to_event            |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu.py                                   | test_linalg_lu                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu.py                                   | test_linalg_lu_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_allclose.py                                    | test_allclose                                                          |      1 |      0 |       0 |        1 |
| tests/test_index_copy_.py                                 | test_index_copy                                                        |      1 |      0 |       0 |        1 |
| tests/test_index_copy_.py                                 | test_index_copy_                                                       |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max                                                         |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_                                                        |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_low_precision                                           |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_uint8_int_scalar                                        |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_int8_overflow_scalar                                    |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_uint8_overflow_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                                   | test_clamp_max_uint8_boundary_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_digmma.py                                      | test_digamma_                                                          |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                   | test_transpose_same_dim                                                |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                   | test_transpose_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                   | test_transpose_invalid_dims                                            |      1 |      0 |       0 |        1 |
| tests/test_flash_mla_sparse_fwd.py                        | test_flashmla_sparse                                                   |      1 |      0 |       0 |        1 |
| tests/test_cast_Long.py                                   | test_cast_Long                                                         |      1 |      0 |       0 |        1 |
| tests/test_cast_Long.py                                   | test_cast_Long_strided                                                 |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_preserves_nan_payloads                                      |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_out_alias_across_sort_paths                                 |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort                                                             |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_noncontiguous_empty_scalar_and_special_values               |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_rejects_complex                                             |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                       | test_msort_out_resize_noncontiguous_alias_and_dtype_error              |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d                                                  |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_list_padding                                     |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_empty_padding                                    |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_3d_input                                         |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_out                                              |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_out_3d_input                                     |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad2d.py                            | test_reflection_pad2d_out_empty_padding                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul_structure_and_zeros                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul_out                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul_rejects_shape_mismatch                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul_rejects_dtype_mismatch                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_sparse_sparse_matmul_rejects_dense_operand                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                        | test_dispatch_reaches_gems_impl                                        |      1 |      0 |       0 |        1 |
| tests/test_cp_gather_indexer_k_quant_cache.py             | test_cp_gather_indexer_k_quant_cache_matches_reference                 |      1 |      0 |       0 |        1 |
| tests/test_bartlett_window.py                             | test_bartlett_window                                                   |      1 |      0 |       0 |        1 |
| tests/test_add_relu.py                                    | test_add_relu                                                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell                                            |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_non_contiguous                             |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_zero_sized                                 |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_invalid_input                              |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_complex_rejected                           |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_autograd_graph                             |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                               | test_accuracy_rnn_tanh_cell_keyword_arguments                          |      1 |      0 |       0 |        1 |
| tests/test_mvlgamma_.py                                   | test_special_multigammaln                                              |      1 |      0 |       0 |        1 |
| tests/test_mvlgamma_.py                                   | test_mvlgamma_                                                         |      1 |      0 |       0 |        1 |
| tests/test_nested_view_from_jagged_copy.py                | test_nested_view_from_jagged_copy_2d                                   |      0 |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged_copy.py                | test_nested_view_from_jagged_copy_1d                                   |      0 |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged_copy.py                | test_nested_view_from_jagged_copy_lengths                              |      0 |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged_copy.py                | test_nested_view_from_jagged_copy_min_seqlen                           |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2                                              |      0 |      0 |       1 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2_tensor_wise                                  |      0 |      0 |       1 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2_out_dtype_default                            |      0 |      0 |       1 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2_use_fast_accum                               |      0 |      0 |       1 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2_rejects_bias                                 |      0 |      0 |       1 |        1 |
| tests/test_scaled_grouped_mm_v2.py                        | test_scaled_grouped_mm_v2_matches_native                               |      0 |      0 |       1 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_iluvatar_fp64_rejected                |      0 |      0 |       1 |        1 |
| tests/test_upsample_bicubic2d_backward.py                 | test_upsample_bicubic2d_backward_bfloat16_scalar_rounding              |      0 |      0 |       1 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py   | test_fused_recurrent_gated_delta_rule_matches_vllm                     |      0 |      0 |       1 |        1 |
| tests/flagtune/test_published_regression.py               | test_published_regression                                              |      0 |      0 |       1 |        1 |
| tests/test_transpose.py                                   | test_transpose                                                         |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_sparse_fwd.py                        | test_flash_mla_sparse_flashmla                                         |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_sparse_decode_v32_fp8                                             |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_sparse_decode_model1_topk_length_attn_sink_out                    |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_sparse_decode_model1_extra_kv                                     |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_dense_decode_seq_q1                                               |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_error_v32_rejects_topk_length                                     |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_error_model1_extra_cache_requires_extra_indices                   |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_error_dense_rejects_sparse_only_args                              |      0 |      0 |       1 |        1 |
| tests/test_flash_mla_with_kvcache.py                      | test_error_sched_meta_reuse_mismatch                                   |      0 |      0 |       1 |        1 |
| TOTAL                                                     |                                                                        |    222 |      4 |      20 |      246 |
