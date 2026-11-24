|                           filepath                            |                                function                                | passed | failed | skipped | SUBTOTAL |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_ilshift.py                                         | test_ilshift                                                           |      1 |      0 |       0 |        1 |
| tests/test_ilshift.py                                         | test_ilshift_int8                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv_ex.py                                   | test_linalg_inv_ex                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv_ex.py                                   | test_linalg_inv_ex_singular                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv_ex.py                                   | test_linalg_inv_ex_check_errors                                        |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                                     | test_multinomial_with_replacement                                      |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                                     | test_multinomial_with_replacement_1                                    |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                                     | test_multinomial_does_not_mutate_input                                 |      1 |      0 |       0 |        1 |
| tests/test_multinomial.py                                     | test_multinomial_without_replacement                                   |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable.py | test_scaled_dot_product_fused_attention_overrideable                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_strided                                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_sparse_coo                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_sparse_compressed                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_matches_reference_table                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_disjoint_from_native_neighbors                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_dispatch_sentinel                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_coexistence                                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_bsr_bsc_coverage                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_dim.py                                      | test_sparse_dim_unsupported_layout_parity                              |      1 |      0 |       0 |        1 |
| tests/test_max_pool1d.py                                      | test_max_pool1d                                                        |      1 |      0 |       0 |        1 |
| tests/test_max_pool1d.py                                      | test_max_pool1d_default_stride                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve.py                                    | test_linalg_solve                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve.py                                    | test_linalg_solve_batched                                              |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_formula_matches_aten_cpu          |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_strided_grad_output               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_odd_class_count                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_hinge_uses_input_dtype            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward                                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_target_patterns                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_noncontiguous                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_class_count                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_extra_reduction_values            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_special_values                    |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_empty_batch                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_scalar_grad_output                |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_half_dtypes                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_invalid_target_dtype              |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_invalid_is_target_dtype           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_invalid_target_shape              |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_backward.py                 | test_multilabel_margin_loss_backward_invalid_reduction                 |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_src                                                       |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_reduce_add                                                |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_add_                                                      |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_add_2d_noncontiguous                                      |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_add                                                       |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_add_2d                                                    |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_reduce_multiply                                           |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_src_                                                      |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_reduce_add_                                               |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                         | test_scatter_reduce_multiply_                                          |      1 |      0 |       0 |        1 |
| tests/test_triu.py                                            | test_triu                                                              |      1 |      0 |       0 |        1 |
| tests/test_triu.py                                            | test_triu_noncontiguous                                                |      1 |      0 |       0 |        1 |
| tests/test_triu.py                                            | test_triu_                                                             |      1 |      0 |       0 |        1 |
| tests/test_triu.py                                            | test_triu_inplace_noncontiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_expand.py                                          | test_expand                                                            |      1 |      0 |       0 |        1 |
| tests/test_expand.py                                          | test_expand_                                                           |      1 |      0 |       0 |        1 |
| tests/test_expand.py                                          | test_expand_invalid_sizes                                              |      1 |      0 |       0 |        1 |
| tests/test_expand.py                                          | test_expand_zero_size_singleton_stride                                 |      1 |      0 |       0 |        1 |
| tests/test_cummin.py                                          | test_cummin                                                            |      1 |      0 |       0 |        1 |
| tests/test_cummin.py                                          | test_cummin_with_nan                                                   |      1 |      0 |       0 |        1 |
| tests/test_acos_.py                                           | test_acos_                                                             |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward                                              |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_layout                                       |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_empty_batch                                  |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_empty_channel                                |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_invalid                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_deterministic                                |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_out                                          |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_large_reduction                              |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_fp64_precision                               |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_64bit_address                                |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_out_strided                                  |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_out_input_alias                              |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_invalid_tensors                              |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_spatial_boundaries                           |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_repeated_input_alias                         |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_channel_tiles                                |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_large_dilation                               |      1 |      0 |       0 |        1 |
| tests/test_convolution_backward.py                            | test_convolution_backward_fresh_buffers                                |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                                  | test_typed_pointer                                                     |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                                  | test_typed_pointer_reinterpret_with_offset                             |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                                  | test_typed_pointer_as_is                                               |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                                  | test_strided_buffer_slice                                              |      1 |      0 |       0 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_public_api                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_noncontiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_rank_deficient                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_invalid_input                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_out_noncontiguous_buffers                            |      1 |      0 |       0 |        1 |
| tests/test_addcdiv_.py                                        | test_addcdiv_                                                          |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                             | test_cat                                                               |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                             | test_cat_empty_tensor                                                  |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                             | test_cat_out_matches_reference                                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu                                            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_dilation                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_asymmetric                                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_3d                                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_1d                                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_no_bias                                    |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_bias_shapes                                |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_boundary                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_out                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                          | test_cudnn_convolution_relu_out_no_bias                                |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve                                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_noncontiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_ignored_entries                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_singular_and_near_singular                       |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out                                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_nonempty_resize                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_empty                                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_aliases_inputs                               |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_invalid_shapes                                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_invalid_shapes                               |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_rejected_dtype                                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_rejected_dtype                               |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_mismatched_inputs                                |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_invalid_outputs                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_rejects_expanded_output                      |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_rejects_autograd                             |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_scipy_reference                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_wide_rhs                                         |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_large_ignored_entries                            |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_out_large_ignored_entries                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_float64_precision                                |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_finite_inverse_overflow                          |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_chunked_previous_rows                            |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_panel_wide_rhs                                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_float64_offdiagonal_precision                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_nonempty                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_empty                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_hybrid_input                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_uncoalesced_input                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_keeps_old_buffers_intact                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_size_mismatch_raises                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_non_coo_rejected                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                        | test_sparse_resize_and_clear_leaf_requires_grad_rejected               |      1 |      0 |       0 |        1 |
| tests/test_prod.py                                            | test_prod                                                              |      1 |      0 |       0 |        1 |
| tests/test_prod.py                                            | test_prod_dim_int                                                      |      1 |      0 |       0 |        1 |
| tests/test_prod.py                                            | test_prod_dim_multi_tile                                               |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py           | test_nested_from_padded_and_nested_example_3d                          |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py           | test_nested_from_padded_and_nested_example_2d                          |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py           | test_nested_from_padded_and_nested_example_ragged_idx                  |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py           | test_nested_from_padded_and_nested_example_zero_length                 |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_scalar.py                              | test_bitwise_and_scalar                                                |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_scalar.py                              | test_bitwise_and_scalar_                                               |      1 |      0 |       0 |        1 |
| tests/test_accuracy_utils.py                                  | test_gems_log_prefix                                                   |      1 |      0 |       0 |        1 |
| tests/test_accuracy_utils.py                                  | test_gems_log_logger                                                   |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward_dim_sweep                                |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward_cpu_oracle                               |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward_native_dropout                           |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward_empty_gems_cpu_oracle                    |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_scaled_dot_product_cudnn_attention_backward                       |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_efficient_attention_backward                                      |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_scaled_dot_product_efficient_attention_backward                   |      1 |      0 |       0 |        1 |
| tests/test_flash_attention_backward.py                        | test_scaled_dot_product_flash_attention_backward                       |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_same_tensors                              |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_different_tensors_same_shape              |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_view_of_same_storage                      |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_as_strided_views                          |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_different_dtypes                          |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_different_shapes_different_storages       |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_empty_tensors                             |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_non_contiguous                            |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_chunk_views                               |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_dtype_sweep                               |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                          | test__has_same_storage_numel_dispatch_stability                        |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_                                                            |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_mixed_dtype_                                                |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_broadcast_noncontiguous_and_empty_                          |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_special_values_                                             |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_extreme_integral_exponents_                                 |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_fp64_exponent_precision_                                    |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_complex_                                                    |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_complex128_exponent_precision_                              |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_overlapping_operands_                                       |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_rejects_internally_overlapping_self_                        |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_rejects_integer_self_                                       |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_rejects_complex_to_real_                                    |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_rejects_expanding_self_                                     |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_mixed_precision_extreme_exponent_                           |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_int64_exponent_outside_int32_range_                         |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_complex_integral_extremes_                                  |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                          | test_ldexp_conjugated_complex_exponent_                                |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_fresh_tensor                                              |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_bumps_on_inplace_ops                                      |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_inplace_under_no_grad                                     |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_views_share_base_counter                                  |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_noncontiguous_and_strided_views                           |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_dtype_sweep                                               |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_sparse_layouts                                            |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_inference_mode_raises                                     |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_meta_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_matches_reference_table                                   |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_dispatch_sentinel_composite_key                           |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_disjoint_from_native_neighbors                            |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_non_tensor_input                                          |      1 |      0 |       0 |        1 |
| tests/test_version.py                                         | test_version_bump_sequence_matches_native                              |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_fast_float32                                         |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_rank1_float32                                        |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_rank1_zero_basis                                     |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_gram_ill_conditioned_orthonormal                     |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_gram_zero_and_repeated_singular_values               |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_tiny_rank_degenerate_inputs                          |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_rank2_boundary_degenerate_inputs                     |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_fallback_modes                                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_non_contiguous_empty_and_complex                     |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_complex64_reconstruction                             |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_low_precision_float32_route                          |      1 |      0 |       0 |        1 |
| tests/test_svd.py                                             | test_accuracy_svd_some_false_reconstruction                            |      1 |      0 |       0 |        1 |
| tests/test_slice_scatter.py                                   | test_slice_scatter                                                     |      1 |      0 |       0 |        1 |
| tests/test_slice_scatter.py                                   | testslice_scatter_with_self_overlapping_input                          |      1 |      0 |       0 |        1 |
| tests/test_bitwise_or_scalar_tensor.py                        | test_bitwise_or_scalar_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                           | test_expm1                                                             |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                           | test_expm1_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                           | test_expm1_backward_expanded_grad                                      |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                           | test_expm1_                                                            |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                           | test_expm1_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_scalar                                                 |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_scalar_                                                |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_tensor_value                                           |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_duplicate_index                                        |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_large_row_fill                                         |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_grid_stride                                            |      1 |      0 |       0 |        1 |
| tests/test_index_fill.py                                      | test_index_fill_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft                                                          |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_empty_batch                                              |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_zero_length_transform                                    |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_noncontiguous_transposed                                 |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_noncontiguous_strided                                    |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_transform_length                                         |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_invalid_length                                           |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_norm_modes                                               |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_invalid_norm                                             |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_dim_selection                                            |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_dim_out_of_range                                         |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_zero_dim_input                                           |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_rejects_complex                                          |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_defaults_bind                                            |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_dc_and_nyquist_are_real                                  |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_constant_row_is_pure_dc                                  |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_integer_input_is_promoted                                |      1 |      0 |       0 |        1 |
| tests/test_fft_rfft.py                                        | test_fft_rfft_autograd_input_is_detached                               |      1 |      0 |       0 |        1 |
| tests/test_atanh_.py                                          | test_atanh_                                                            |      1 |      0 |       0 |        1 |
| tests/test_logical_xor.py                                     | test_logical_xor                                                       |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot                                                         |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_invalid_dims                                            |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_dtype_device_validation                                 |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_out_contract                                            |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_unsupported_dtypes                                      |      1 |      0 |       0 |        1 |
| tests/test_tensordot.py                                       | test_tensordot_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_fft.py                                             | test_fft                                                               |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d.py                        | test_accuracy__upsample_nearest_exact1d                                |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_core_matrix                                              |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_edge_cases                                               |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_unbatched                                                |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_registered_intlist_forward                               |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_registered_intlist_backward                              |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_registered_tensor_backward                               |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_forward_no_grad_path                                     |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_full_length_no_grad_reduce_fast_path                     |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_no_grad_large_concatenated_variable_lengths_regression   |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_no_grad_padded_noncontiguous_blank_zero_infinity_none    |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_small_shape_inputs_raise                         |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_log_probs_rank_raises                            |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_padded_target_width_raises                       |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_target_rank_raises                               |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_length_stats_cache_observes_inplace_mutation             |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_concatenated_target_raises                       |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_reduction_raises                                 |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_invalid_blank_raises                                     |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_int_log_probs_raises                                     |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_float_lengths_raise                                      |      1 |      0 |       0 |        1 |
| tests/test_ctc_loss.py                                        | test_ctc_loss_float_targets_match_pytorch                              |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded.py                              | test_nested_from_padded                                                |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded.py                              | test_nested_from_padded_fuse_transform_0213                            |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded.py                              | test_nested_from_padded_int_dtypes                                     |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded.py                              | test_nested_from_padded_non_contiguous_input                           |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded.py                              | test_nested_from_padded_all_padding                                    |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_real_dtypes                                               |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_complex_dtypes                                            |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_0d                                                        |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_1d_matches_native_error                                   |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_non_contiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_mutation_through_view                                     |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_conjugate_bit_input                                       |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_dispatch_stability                                        |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_matches_native_view_table                                 |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                         | test_adjoint_diagnostics_parity                                        |      1 |      0 |       0 |        1 |
| tests/test_mm.py                                              | test_mm                                                                |      1 |      0 |       0 |        1 |
| tests/test_mm.py                                              | test_mm_broadcast_stride_zero                                          |      1 |      0 |       0 |        1 |
| tests/test_mm.py                                              | test_mm_out_vllm_tma_column_major_weight                               |      1 |      0 |       0 |        1 |
| tests/test_mm.py                                              | test_mm_self_transpose                                                 |      1 |      0 |       0 |        1 |
| tests/test_mm.py                                              | test_mm_out_self_transpose                                             |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse                                                          |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_rcond                                                    |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_batch_above_cuda_grid_y_limit                            |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_noncontiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_empty                                                    |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_rcond_and_rank_deficiency                                |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_nonfinite                                                |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_invalid_input                                            |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_autograd                                                 |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_large_multitile_and_moore_penrose                        |      1 |      0 |       0 |        1 |
| tests/test_pinverse.py                                        | test_pinverse_non_diagonal_difficult_inputs                            |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                                     | test_hash_tensor                                                       |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                                     | test_hash_tensor_int_dtypes                                            |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                                     | test_hash_tensor_multidim                                              |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                                     | test_hash_tensor_scalar                                                |      1 |      0 |       0 |        1 |
| tests/test_safe_softmax.py                                    | test_safe_softmax                                                      |      1 |      0 |       0 |        1 |
| tests/test_slice_backward.py                                  | test_slice_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_slice_backward.py                                  | test_slice_backward_oob_end                                            |      1 |      0 |       0 |        1 |
| tests/test_slice_backward.py                                  | test_slice_backward_oob_start                                          |      1 |      0 |       0 |        1 |
| tests/test_atan2_.py                                          | test_atan2_                                                            |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                                       | test_bucketize                                                         |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                                       | test_bucketize_right                                                   |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                                       | test_bucketize_int32                                                   |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                                       | test_bucketize_boundary_cases                                          |      1 |      0 |       0 |        1 |
| tests/test_bucketize.py                                       | test_bucketize_noncontiguous_input                                     |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest1d.py                              | test_upsample_nearest1d                                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa                                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_uint8_and_layout                            |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_transposed_input                            |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_out                                         |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_vec                                         |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_vec_requires_one_size                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                        | test_sparse_compressed_tensor_explicit_size                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                        | test_sparse_compressed_tensor_infer_size                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                        | test_sparse_compressed_tensor_rejects_bad_layout                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                        | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap       |      1 |      0 |       0 |        1 |
| tests/test_reshape_and_cache_flash.py                         | test_reshape_and_cache_flash                                           |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter                                                         |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_                                                        |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_out_none                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_inplace_is_self                                         |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_rejects_out                                             |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_edge_values                                             |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_nan                                                     |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_infinity                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                       | test_nextafter_small_values                                            |      1 |      0 |       0 |        1 |
| tests/test_masked_scatter_backward.py                         | test_accuracy_masked_scatter_backward                                  |      1 |      0 |       0 |        1 |
| tests/test_kthvalue.py                                        | test_kthvalue                                                          |      1 |      0 |       0 |        1 |
| tests/test_kthvalue.py                                        | test_kthvalue_default_dim                                              |      1 |      0 |       0 |        1 |
| tests/test_kthvalue.py                                        | test_kthvalue_invalid_dim                                              |      1 |      0 |       0 |        1 |
| tests/test_kthvalue.py                                        | test_kthvalue_empty_tensor                                             |      1 |      0 |       0 |        1 |
| tests/test_isfinite.py                                        | test_isfinite                                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_                                                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw__tensor_lr                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_amp                                                   |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_mixed_state                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_channels_last                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_many_tensors                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_empty                                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_invalid                                               |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_invalid_hyperparameter                                |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_native                                                |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_graph                                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_launch_boundaries                                     |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_bias_correction                                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_invalid_lr_dtype                                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_bulk_validation                                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_storage_offsets                                       |      1 |      0 |       0 |        1 |
| tests/test_fractional_max_pool3d_backward.py                  | test_accuracy_fractional_max_pool3d_backward                           |      1 |      0 |       0 |        1 |
| tests/test_nan_to_num.py                                      | test_nan_to_num                                                        |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze_                                                          |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__int_dtypes                                               |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__non_contiguous                                           |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__expanded_stride_zero                                     |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__storage_offset                                           |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__rejects_out_of_range_dim                                 |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__rejects_out_of_range_dims                                |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__rejects_duplicate_dims                                   |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__rejects_inplace_on_grad_leaf                             |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__backward                                                 |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__dim                                                      |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                        | test_squeeze__dims                                                     |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_c_contiguous_stride_normal                                        |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_c_contiguous_stride_with_zero_size                                |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_f_contiguous_stride_normal                                        |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_f_contiguous_stride_with_zero_size                                |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_ordered_stride_normal                                             |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_ordered_stride_with_zero_size                                     |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_stride_order                                                      |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_shape_empty                                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_shape1                                               |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_shape2                                               |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_stride_empty                                         |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_stride1                                              |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_the_same_stride2                                              |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_c_contiguous_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_all_c_contiguous1                                                 |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_heuristics_for_tile_size                                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_heuristics_for_num_warps                                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_contiguous                                         |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_transposed_is_not_viewable                         |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_keeps_transposed_layout                            |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_unit_dims_any_stride                               |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_scalar_and_empty                                   |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                                     | test_compute_stride_matches_torch_view                                 |      1 |      0 |       0 |        1 |
| tests/test_poisson.py                                         | test_poisson                                                           |      1 |      0 |       0 |        1 |
| tests/test_poisson.py                                         | test_poisson_varying_rates                                             |      1 |      0 |       0 |        1 |
| tests/test_poisson.py                                         | test_poisson_large_lambda                                              |      1 |      0 |       0 |        1 |
| tests/test_poisson.py                                         | test_poisson_zero_rate                                                 |      1 |      0 |       0 |        1 |
| tests/test_dgeglu.py                                          | test_dgeglu                                                            |      1 |      0 |       0 |        1 |
| tests/test_bitwise_right_shift_.py                            | test_bitwise_right_shift_                                              |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d_backward.py                    | test_adaptive_avg_pool2d_backward                                      |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                              | test_gt                                                                |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                              | test_gt_scalar                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_lower_left                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_upper_left                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_right                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_unitriangular                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_batched                                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_batched_multiblock                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_grid_above_core_count                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_linalg_solve_triangular_out                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_residual_f64                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_empty                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_large_n_f64                                                       |      1 |      0 |       0 |        1 |
| tests/test_tan.py                                             | test_tan                                                               |      1 |      0 |       0 |        1 |
| tests/test_tan.py                                             | test_tan_                                                              |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py              | test_scaled_dot_product_cudnn_attention                                |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py              | test_scaled_dot_product_cudnn_attention_with_attn_bias                 |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign                                                              |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_out                                                          |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_                                                             |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign__noncontiguous                                               |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_special_values                                               |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_noncontiguous                                                |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_out_noncontiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_out_resizes_empty_tensor                                     |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_out_rejects_mismatched_dtype                                 |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_empty                                                        |      1 |      0 |       0 |        1 |
| tests/test_sign.py                                            | test_sign_rejects_complex                                              |      1 |      0 |       0 |        1 |
| tests/test_concatenate.py                                     | test_concatenate                                                       |      1 |      0 |       0 |        1 |
| tests/test_concatenate.py                                     | test_concatenate_empty_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_fractional_max_pool3d.py                           | test_accuracy_fractional_max_pool3d                                    |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter_backward.py                       | test_diagonal_scatter_backward                                         |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter_backward.py                       | test_diagonal_scatter_backward_autograd                                |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter_backward.py                       | test_diagonal_scatter_backward_empty_diagonal                          |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter_backward.py                       | test_diagonal_scatter_backward_use_gems_entry                          |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py                  | test_nested_tensor_from_tensor_list                                    |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py                  | test_dispatch_reaches_gems_impl                                        |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss                                                           |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss_int_reduction                                             |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss_broadcast                                                 |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss_empty                                                     |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss_functional                                                |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                         | test_l1_loss_invalid_reduction                                         |      1 |      0 |       0 |        1 |
| tests/test_unfold_backward.py                                 | test_unfold_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy                                                       |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_values_view                                           |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out_into_offset_buffer                                |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out_into_strided_view                                 |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_special_values                                        |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out_special_values                                    |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_backward                                              |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_rejects_dense_input                                   |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_rejects_vendor_unsupported_layout                     |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_rejects_non_tensor_input                              |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out_rejects_mismatched_dtype                          |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test__values_copy                                                      |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test__values_copy_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test__values_copy_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_noncontiguous_stored_values                           |      1 |      0 |       0 |        1 |
| tests/test_values_copy.py                                     | test_values_copy_out_noncontiguous_stored_values                       |      1 |      0 |       0 |        1 |
| tests/test_masked_scale.py                                    | test_masked_scale                                                      |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                               | test_scaled_grouped_mm                                                 |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                               | test_scaled_grouped_mm_fp8_core                                        |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                               | test_scaled_grouped_mm_fp8_encodings                                   |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                               | test_scaled_grouped_mm_scale_before_output_cast                        |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                               | test_scaled_grouped_mm_reduction_addressing                            |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_value_ranges                            |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_tiny_scale                              |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_f32_scale_edges                         |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_constant                                |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_inf                                     |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_non_contiguous                          |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_default_reduce_range                    |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_rejects_nan                             |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_rejects_empty                           |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_rejects_complex                         |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_rejects_fp8                             |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_per_tensor.py                       | test_choose_qparams_per_tensor_rejects_non_tensor                      |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_decode.py                            | test_top_k_per_row_decode                                              |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_decode.py                            | test_top_k_per_row_decode_partial_seqlen                               |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_decode.py                            | test_topk_greater_than_row_len                                         |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_decode.py                            | test_logits_diff_in_8LSBits                                            |      1 |      0 |       0 |        1 |
| tests/test_not_equal_.py                                      | test_accuracy_not_equal_                                               |      1 |      0 |       0 |        1 |
| tests/test_not_equal_.py                                      | test_accuracy_not_equal_scalar_                                        |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py                          | test_keeps_declared_name_when_backend_is_available                     |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py                          | test_uses_registered_privateuse1_name                                  |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py                          | test_env_override_wins                                                 |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py                          | test_falls_back_to_declared_name_without_privateuse1                   |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool3d_backward.py                    | test_adaptive_avg_pool3d_backward_grad_input                           |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test__flash_attention_forward                                          |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_sparse_attention                                                  |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_flash_attention_forward_gqa_alibi_softcap                         |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_flash_attention_foward_splitkv                                    |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_flash_attention_foward_swa                                        |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_flash_fwd_dropout                                                 |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_splitkv_combine_output_layout                                     |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_splitkv_combine_graph_replay                                      |      1 |      0 |       0 |        1 |
| tests/test_flash_attention.py                                 | test_splitkv_mha_output_layout                                         |      1 |      0 |       0 |        1 |
| tests/test_unbind.py                                          | test_unbind                                                            |      1 |      0 |       0 |        1 |
| tests/test_unbind.py                                          | test_unbind_default_dim                                                |      1 |      0 |       0 |        1 |
| tests/test_unbind.py                                          | test_unbind_negative_dim                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                        | test_sparse_bsr_tensor_unsafe                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                        | test_sparse_bsr_tensor_unsafe_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                        | test_sparse_bsr_tensor_unsafe_skips_index_validation                   |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel                                              |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_last_axis                                    |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_extremes                                     |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_tie_cases                                    |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_round_trip                                   |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_float_zero_points                            |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_float_zero_points_tie_cases                  |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_float_zero_points_qscheme_differs            |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_out                                          |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_out_matches_aten_out                         |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_out_rejects_wrong_dtype                      |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_out_noncontiguous                            |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_non_float_input                      |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_float_zero_point_non_float_input     |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_bad_axis                             |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_cpu_qparams                          |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_multi_dim_qparams                    |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_short_qparams                        |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_integer_scales                       |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_unsupported_dtype                    |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_rejects_out_of_range_zero_point              |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_zero_point_error_matches_native              |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                            | test_quantize_per_channel_accepts_valid_zero_points                    |      1 |      0 |       0 |        1 |
| tests/test__addmm_activation.py                               | test_addmm_activation                                                  |      1 |      0 |       0 |        1 |
| tests/test__addmm_activation.py                               | test_addmm_activation_out                                              |      1 |      0 |       0 |        1 |
| tests/test_addcdiv.py                                         | test_addcdiv                                                           |      1 |      0 |       0 |        1 |
| tests/test_addcdiv.py                                         | test_addcdiv_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                         | test_backward_real_dispatch                                            |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                         | test_backward_uses_gems_compute                                        |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                         | test_inplace_saved_version_is_checked                                  |      1 |      0 |       0 |        1 |
| tests/test_col2im.py                                          | test_col2im                                                            |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                          | test_copy_sparse_to_sparse_                                            |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                          | test_copy_sparse_to_sparse_self_alias                                  |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                          | test_copy_sparse_to_sparse_dtype_conversion                            |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                          | test_copy_sparse_to_sparse_nonempty_shrink_raises                      |      1 |      0 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                          | test_copy_sparse_to_sparse_nonempty_grow                               |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_unused_overrides_lists_injected_but_never_called                  |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_unused_overrides_empty_once_the_candidate_runs                    |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_unused_overrides_ignores_programmatic_overrides                   |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_report_is_withheld_when_an_override_name_did_not_match            |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_report_writer_is_untouched_when_every_override_matched            |      1 |      0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py                     | test_non_timing_modes_do_not_treat_overrides_as_mistakes               |      1 |      0 |       0 |        1 |
| tests/test_unsafe_masked_index.py                             | test_unsafe_masked_index                                               |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__basic                                              |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__options                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__grad_scale                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__found_inf                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__empty                                              |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__mixed_empty_and_nonempty                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__mismatched_list_lengths                            |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__mismatched_tensor_properties                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__later_entry_bad_dtype_or_device                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__later_entry_bad_state_step                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__multi_tensor                                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__transposed_parity                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                                  | test_fused_adagrad__internal_overlap_rejected                          |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_gather_stats.py                         | test_batch_norm_gather_stats                                           |      1 |      0 |       0 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm_ascend_core_dim_boundary                                 |      0 |      0 |       1 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm_ascend_core_dim_capped                                   |      0 |      0 |       1 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm_ascend_small_shapes                                      |      0 |      0 |       1 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm_ascend_review_empty                                      |      0 |      0 |       1 |        1 |
| tests/test_rms_norm.py                                        | test_rms_norm_ascend_review_weight_gradient                            |      0 |      0 |       1 |        1 |
| tests/test_linalg_polar.py                                    | test_linalg_polar_dispatcher                                           |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_rejected_float64                                 |      0 |      0 |       1 |        1 |
| tests/test_flash_attention_backward.py                        | test_flash_attention_backward_saved_state_arithmetic_range             |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_large_shapes                                            |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_transposed_inputs_and_k_tails                           |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_large_shape_entrypoints                                 |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_empty                                            |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_out_resize                                       |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_capped_routes                                    |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_narrow_real_core_dim                             |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_panel_loops                                      |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_broadcast_materialize                            |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_specialized_launch_cap                           |      0 |      0 |       1 |        1 |
| tests/test_mm.py                                              | test_mm_ascend_review_out_validation                                   |      0 |      0 |       1 |        1 |
| tests/test_upsample_lanczos2d_aa.py                           | test_upsample_lanczos2d_aa_dispatch                                    |      0 |      0 |       1 |        1 |
| tests/test_reshape_and_cache_flash.py                         | test_reshape_and_cache_flash_vllm_accuracy                             |      0 |      0 |       1 |        1 |
| tests/test_masked_scatter_backward.py                         | test_accuracy_masked_scatter_backward_persistent                       |      0 |      0 |       1 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_alignment_classes                                     |      0 |      0 |       1 |        1 |
| tests/test_linalg_solve_triangular.py                         | test_no_tle_fallback                                                   |      0 |      0 |       1 |        1 |
| tests/test_flash_attention.py                                 | test_hygon_fp32_flash_attention_forward                                |      0 |      0 |       1 |        1 |
| tests/test_flash_attention.py                                 | test_hygon_fp32_flash_attention_negative_logit_sentinel                |      0 |      0 |       1 |        1 |
| tests/test_flash_attention.py                                 | test_flash_attention_foward_nonsquare_qk                               |      0 |      0 |       1 |        1 |
| tests/test_flash_attention.py                                 | test_ascend_varlen_forward_backward_padded_heads                       |      0 |      0 |       1 |        1 |
| tests/test_flash_attention.py                                 | test_ascend_dense_forward_backward                                     |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                                | test_triangular_solve_autograd_broadcast                               |      0 |      1 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_dispatch                                              |      0 |      1 |       0 |        1 |
| tests/test_fused_adamw_.py                                    | test_fused_adamw_optimizer                                             |      0 |      1 |       0 |        1 |
| TOTAL                                                         |                                                                        |    575 |      3 |      30 |      608 |
|                    filepath                     |                                function                                 | passed | skipped | SUBTOTAL |
| ----------------------------------------------- | ----------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_linalg_ldl_factor_ex.py              | test_linalg_ldl_factor_ex                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_accuracy_2d                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_accuracy_3d                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_shape_consistency                                         |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_custom_padding                                            |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_default_inf_padding                                       |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_single_batch                                              |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_short_sequences                                           |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_3d_single_batch                                           |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_3d_short_sequences                                        |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_block_sizes                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_int8_custom_padding                                       |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_int8_valid_region_with_default_padding                    |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_int8_out_of_range_padding_is_unvalidated                  |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_view                                    |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_axis_not_broadcast                      |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_invalid_expand                          |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_dtypes                                  |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_dispatch_stability                      |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_ignores_level                           |      1 |       0 |        1 |
| tests/test__remove_batch_dim.py                 | test_accuracy__remove_batch_dim_alias_write_through                     |      1 |       0 |        1 |
| tests/test_logit.py                             | test_logit                                                              |      1 |       0 |        1 |
| tests/test_logit.py                             | test_logit_                                                             |      1 |       0 |        1 |
| tests/test_logit.py                             | test_logit_out                                                          |      1 |       0 |        1 |
| tests/test_logit.py                             | test_logit_out_non_contiguous_out                                       |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_csc                                                    |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_bsc                                                    |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_csr_raises                                             |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_alias_and_mutation                                     |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_batched_shape                                          |      1 |       0 |        1 |
| tests/test_row_indices.py                       | test_row_indices_unsupported_layout_raises                              |      1 |       0 |        1 |
| tests/test_hamming_window.py                    | test_hamming_window                                                     |      1 |       0 |        1 |
| tests/test_hamming_window.py                    | test_hamming_window_periodic                                            |      1 |       0 |        1 |
| tests/test_hamming_window.py                    | test_hamming_window_periodic_alpha                                      |      1 |       0 |        1 |
| tests/test_hamming_window.py                    | test_hamming_window_periodic_alpha_beta                                 |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov                                                                |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_1d                                                             |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_correction                                                     |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_fweights                                                       |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_aweights                                                       |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_fweights_aweights                                              |      1 |       0 |        1 |
| tests/test_cov.py                               | test_cov_non_contiguous                                                 |      1 |       0 |        1 |
| tests/test_fliplr.py                            | test_accuracy_fliplr                                                    |      1 |       0 |        1 |
| tests/test_fliplr.py                            | test_accuracy_fliplr_non_float                                          |      1 |       0 |        1 |
| tests/test_fliplr.py                            | test_accuracy_fliplr_noncontiguous                                      |      1 |       0 |        1 |
| tests/test_fliplr.py                            | test_fliplr_rejects_inputs_with_fewer_than_two_dims                     |      1 |       0 |        1 |
| tests/test_cross_entropy_loss.py                | test_cross_entropy_loss_indices                                         |      1 |       0 |        1 |
| tests/test_cross_entropy_loss.py                | test_cross_entropy_loss_probabilities                                   |      1 |       0 |        1 |
| tests/test_can_cast.py                          | test_can_cast_category_boundaries                                       |      1 |       0 |        1 |
| tests/test_can_cast.py                          | test_can_cast_full_dtype_grid                                           |      1 |       0 |        1 |
| tests/test_can_cast.py                          | test_can_cast_registered_on_composite_key                               |      1 |       0 |        1 |
| tests/test_resize.py                            | test_resize                                                             |      1 |       0 |        1 |
| tests/test_resize.py                            | test_resize_                                                            |      1 |       0 |        1 |
| tests/test_patch_util.py                        | test_vllm_extension_legacy_and_stable_abi_fallback                      |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_channelwise                                  |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_groupwise                                    |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_noncontiguous                                |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_rounding_and_nibbles                         |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_invalid                                      |      1 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py             | test_dyn_quant_matmul_4bit_compile                                      |      1 |       0 |        1 |
| tests/test_masked_softmax_backward.py           | test_masked_softmax_backward                                            |      1 |       0 |        1 |
| tests/test_conv_depthwise2d.py                  | test_conv_depthwise2d                                                   |      1 |       0 |        1 |
| tests/test_batch_norm_stats.py                  | test_batch_norm_stats                                                   |      1 |       0 |        1 |
| tests/test_scalar_tensor.py                     | test_scalar_tensor                                                      |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_u.py    | test_special_chebyshev_polynomial_u                                     |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_u.py    | test_special_chebyshev_polynomial_u_tensor_n                            |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_u.py    | test_special_chebyshev_polynomial_u_guard_scalar                        |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_u.py    | test_special_chebyshev_polynomial_u_guard_tensor                        |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy                                                |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_noncontiguous_input                            |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_strided_input                                  |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_high_rank_noncontiguous                        |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_broadcast_input                                |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_storage_offset                                 |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_independent_storage                            |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_tail_chunk                                     |      1 |       0 |        1 |
| tests/test_split_copy.py                        | test_accuracy_split_copy_error_paths                                    |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py | test_upsample_nearest_exact2d_backward                                  |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py | test_upsample_nearest_exact2d_backward_with_scales                      |      1 |       0 |        1 |
| tests/test_randn_like.py                        | test_randn_like                                                         |      1 |       0 |        1 |
| tests/test_bitwise_or_scalar.py                 | test_bitwise_or_scalar                                                  |      1 |       0 |        1 |
| tests/test_bitwise_or_scalar.py                 | test_bitwise_or_scalar_                                                 |      1 |       0 |        1 |
| tests/test_triton_lang_helper.py                | test_patch_missing_nearbyint_uses_rint                                  |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_infer_size                                       |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_conversion_and_strides                           |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_scalar_values                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_invalid_inputs                                   |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_coalesced                                        |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_dispatch                                         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_registration_scope                               |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_mismatched_devices                               |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_autograd                                         |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_pin_memory                                       |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_pin_memory_rejection                             |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_large_reduction                                  |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_size_overflow                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_cpu_transfer                                     |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_invariant_check                                  |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_explicit_size                                    |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_explicit_validation                              |      1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_explicit_pin_memory                              |      1 |       0 |        1 |
| tests/test_replication_pad2d_backward.py        | test_replication_pad2d_backward                                         |      1 |       0 |        1 |
| tests/test_replication_pad2d_backward.py        | test_replication_pad2d_backward_grad_input                              |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack                                                          |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_unpack_data_false                                        |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_unpack_pivots_false                                      |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_batched                                                  |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_out                                                      |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_backward                                                 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_backward_entry                                           |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_backward_batched                                         |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_backward_nondifferentiable_outputs                       |      1 |       0 |        1 |
| tests/test_lu_unpack.py                         | test_lu_unpack_backward_dispatch                                        |      1 |       0 |        1 |
| tests/test_alpha_dropout_.py                    | test_alpha_dropout_                                                     |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                   | test_embedding_spmdm_matches_embedding_bag                              |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                   | test_embedding_spmdm_index_dtypes                                       |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                   | test_embedding_spmdm_noncontiguous_inputs                               |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                   | test_embedding_spmdm_validation                                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_default_identity                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_float64_preserves_small_singular_value |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_float64_tolerance_precision            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_well_separated_spectrum                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_rank_deficient                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_nonempty_zero_matrix                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_shapes                                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_matches_adjoint                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_aah_svd_matches_hermitian              |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_tolerance_combinations                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_scalar_tolerance_types                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_legacy_float_tolerance                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_per_batch_tolerance                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_broadcast_tolerance                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_false                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_true                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_uses_lower_triangle          |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_order_33                     |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_shapes                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_general_large_shapes                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_general_dense_low_rank                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_dense_low_rank               |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_empty                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_out                                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_tensor_tolerance_out                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_legacy_float_tolerance_out             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance_out            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_out_wrong_dtype                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_out_wrong_device                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_out_wrong_shape_warns_and_resizes      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_official_dtype_contract                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_rejects_complex_tolerance              |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_nonsquare_tail_energy                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_nonsquare_lowrank                      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_strict_threshold             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_negative_tolerances                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_deflated_spectrum            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_extreme_scales                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_mixed_magnitude_batch                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_tolerance_precision                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_fp64_critical_spectrum                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_ignores_strict_upper   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_deflated               |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_extreme_scales         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_critical_spectrum      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_batched                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_large_hermitian_repeatable             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_empty_validates_tolerances             |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i0.py        | test_special_modified_bessel_i0                                         |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i0.py        | test_special_modified_bessel_i0_out                                     |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py      | test_nested_view_from_buffer_copy                                       |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py      | test_nested_view_from_buffer_copy_oversized_tail                        |      1 |       0 |        1 |
| tests/test_log_normal.py                        | test_log_normal                                                         |      1 |       0 |        1 |
| tests/test_log_normal.py                        | test_log_normal_params                                                  |      1 |       0 |        1 |
| tests/test_log_normal.py                        | test_log_normal_moments                                                 |      1 |       0 |        1 |
| tests/test_log_normal.py                        | test_log_normal_moments_large_std                                       |      1 |       0 |        1 |
| tests/test_log_normal.py                        | test_log_normal_large_shapes                                            |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                  | test_accuracy_linalg_multi_dot                                          |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                  | test_accuracy_linalg_multi_dot_out                                      |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                  | test_accuracy_linalg_multi_dot_out_noncontiguous                        |      1 |       0 |        1 |
| tests/test_linalg_multi_dot.py                  | test_error_linalg_multi_dot_requires_two_tensors                        |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_random                                                  |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_random_batch                                            |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_positive_definite                                       |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_negative_determinant                                    |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_diagonal                                                |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_singular                                                |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_empty                                                   |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_non_contiguous                                          |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_errors                                                  |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_out                                                     |      1 |       0 |        1 |
| tests/test_linalg_det.py                        | test_linalg_det_out_errors                                              |      1 |       0 |        1 |
| tests/test_arccosh_.py                          | test_arccosh_                                                           |      1 |       0 |        1 |
| tests/test_hardtanh_.py                         | test_hardtanh_                                                          |      1 |       0 |        1 |
| tests/test_hardtanh_.py                         | test_hardtanh__explicit                                                 |      1 |       0 |        1 |
| tests/test_hardtanh_.py                         | test_hardtanh__special_values                                           |      1 |       0 |        1 |
| tests/test_convolution_backward_overrideable.py | test_convolution_backward_overrideable                                  |      1 |       0 |        1 |
| tests/test_convolution_backward_overrideable.py | test_convolution_backward_overrideable_out                              |      1 |       0 |        1 |
| tests/test_view_as_complex.py                   | test_view_as_complex_accuracy                                           |      1 |       0 |        1 |
| tests/test_view_as_complex.py                   | test_view_as_complex_2d_accuracy                                        |      1 |       0 |        1 |
| tests/test_view_as_complex.py                   | test_view_as_complex_3d_accuracy                                        |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_vector_1d                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_vector_nd                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_matrix_2d                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_matrix_2d_svd                                                      |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_matrix_batch                                                       |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_keepdim_dtype                                                      |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_bad_dim_length_rejected                                            |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_ord_without_dim_requires_1d_2d                                     |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_unsupported_matrix_ord_rejected                                    |      1 |       0 |        1 |
| tests/test_linalg_norm.py                       | test_fro_on_1d_rejected                                                 |      1 |       0 |        1 |
| tests/test_eq.py                                | test_eq                                                                 |      1 |       0 |        1 |
| tests/test_eq.py                                | test_eq_scalar                                                          |      1 |       0 |        1 |
| tests/test_eq.py                                | test_eq_integer_precision                                               |      1 |       0 |        1 |
| tests/test_eq.py                                | test_eq_scalar_integer_precision                                        |      1 |       0 |        1 |
| tests/test_eq.py                                | test_eq_scalar_float_scalar                                             |      1 |       0 |        1 |
| tests/test_conv_transpose3d.py                  | test_conv_transpose3d                                                   |      1 |       0 |        1 |
| tests/test_conv_transpose3d.py                  | test_conv_transpose3d_bias                                              |      1 |       0 |        1 |
| tests/test_conv_transpose3d.py                  | test_conv_transpose3d_dilation                                          |      1 |       0 |        1 |
| tests/test_conv_transpose3d.py                  | test_conv_transpose3d_groups                                            |      1 |       0 |        1 |
| tests/test_conv_transpose3d.py                  | test_conv_transpose3d_output_padding                                    |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py           | test_native_batch_norm_legit                                            |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py           | test_native_batch_norm_legit_no_stats                                   |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit.py           | test_native_batch_norm_legit_out                                        |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_csc                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_bsc                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_batched                                           |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_wrong_layout                                      |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_out                                               |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_out_non_contiguous                                |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_out_resize                                        |      1 |       0 |        1 |
| tests/test_row_indices_copy.py                  | test_row_indices_copy_out_dtype_mismatch_raises                         |      1 |       0 |        1 |
| tests/test_values.py                            | test_accuracy_values_coo                                                |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_mutation_through_view                                       |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_strided_raises                                              |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_sparse_compressed                                           |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_unsupported_layouts_reject_like_native                      |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_non_float_dtypes_alias                                      |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_int_hybrid_alias                                            |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_requires_grad_nondifferentiable_contract                    |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_sparse_requires_grad_view_contract                          |      1 |       0 |        1 |
| tests/test_values.py                            | test_values_dispatch_stability                                          |      1 |       0 |        1 |
| tests/test_bitwise_left_shift_.py               | test_bitwise_left_shift_                                                |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril                                                               |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_noncontiguous                                                 |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_wide_exact_row_dispatch                                       |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_tiny_batched_tile_dispatch                                    |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_tiny_batched_tile_correctness                                 |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out                                                           |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_resizes                                                   |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_aliases_input                                             |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_noncontiguous_out                                         |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_sliced_leading_batch_out                                  |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_extreme_diagonal_noncontiguous_out                        |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_out_strided_dispatch_guards                                   |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace                                                       |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_noncontiguous                                         |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_strided_batched_view                                  |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_expanded_view                                         |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_overlapping_as_strided_view                           |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_empty                                                         |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_empty                                                 |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_extreme_diagonal                                      |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_invalid_rank                                                  |      1 |       0 |        1 |
| tests/test_tril.py                              | test_tril_inplace_invalid_rank                                          |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py          | test__saturate_weight_to_fp16                                           |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py          | test__saturate_weight_to_fp16_extreme_values                            |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py          | test__saturate_weight_to_fp16_boundary                                  |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py          | test__saturate_weight_to_fp16_nan                                       |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_tensor_tensor                                                  |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_tensor_scalar                                                  |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_tensor_tensor_                                                 |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_tensor_scalar_                                                 |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_scalar_tensor                                                  |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_scalar_scalar                                                  |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_complex_complex                                                |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_complex_float_tensor                                           |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_complex_int_tensor                                             |      1 |       0 |        1 |
| tests/test_sub.py                               | test_sub_complex_int_scalar                                             |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_                                                            |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_float_float_                                                |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_float_tensor                                                |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_tensor_float                                                |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_honors_generator                                            |      1 |       0 |        1 |
| tests/test_normal.py                            | test_normal_tensor_tensor                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_fp8_basic                                                 |      0 |       1 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_fp8_custom_padding                                        |      0 |       1 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_fp8_default_inf_padding                                   |      0 |       1 |        1 |
| tests/test_pack_seq.py                          | test_pack_seq_fp8_block_sizes                                           |      0 |       1 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py | test_stage_deepseek_v4_mega_moe_inputs_accuracy                         |      0 |       1 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py | test_stage_deepseek_v4_mega_moe_inputs_rejects_bad_hidden_size          |      0 |       1 |        1 |
| tests/test_grouped_topk.py                      | test_grouped_topk                                                       |      0 |       1 |        1 |
| tests/test_grouped_topk.py                      | test_grouped_topk_large_scale                                           |      0 |       1 |        1 |
| tests/test_grouped_topk.py                      | test_grouped_topk_scaling_factor                                        |      0 |       1 |        1 |
| tests/test_grouped_topk.py                      | test_grouped_topk_single_token                                          |      0 |       1 |        1 |
| tests/test_grouped_topk.py                      | test_grouped_topk_sigmoid                                               |      0 |       1 |        1 |
| tests/test_fp8_fp4_paged_mqa_logits.py          | test_fp8_fp4_paged_mqa_logits                                           |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                     | test_scaled_mm_v2                                                       |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                     | test_scaled_mm_v2_bias                                                  |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                     | test_scaled_mm_v2_vs_torch                                              |      0 |       1 |        1 |
| tests/test_sparse_coo_tensor.py                 | test_sparse_coo_tensor_device_index                                     |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_fp64_rejected                          |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_no_native_fp64_k32                     |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_ascend_slow_decay_spectrum             |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_ascend_long_rectangular                |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_ascend_shape_boundaries                |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper_large   |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_report_conversion_requires_platform_identity                       |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_environment_snapshot_records_model_resolution_controls             |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_operator_yaml_compiles_shape_dispatch_and_benchmark_contract       |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mul_operator_yaml_compiles_broadcast_and_scalar_variants           |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mul_shape_loader_aligns_optional_rhs_before_count                  |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mul_executor_uses_variant_specific_tensor_and_scalar_arguments     |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mul_keeps_legacy_contract_dispatch                                 |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_route_platform_matches_default_manifest_catalog                    |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mm_executor_resolves_route_before_tuner_lookup                     |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mm_executor_keeps_non_adapted_route_out_of_model_binding           |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_worker_routes_actual_tensors_once_before_binding                   |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_hopper_atomic_splitk_is_a_model_route                              |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_operator_yaml_rejects_device_placement_policy                      |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_operator_yaml_rejects_arbitrary_invocation                         |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_shared_safe_references_cover_fields_shapes_and_invoke_args         |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_shared_safe_references_reject_forward_dependencies_and_calls       |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_operator_yaml_rejects_unknown_identity_namespace                   |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_public_operator_resolution_reports_missing_callable                |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_load_new_shape_spec_with_optional_count                            |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_load_legacy_shape_desc_without_count                               |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_shape_validation_is_strict                                         |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_variant_resolution_follows_public_mm_priority                      |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mm_variant_filter_is_deferred_until_tensor_routing                 |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_mm_parent_percentage_limit_applies_to_raw_recipes                  |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_percentage_shape_limit_rejects_zero_resolved_shapes                |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_max_shapes_parser_accepts_counts_and_percentages                   |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_max_shapes_parser_rejects_invalid_limits                           |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_count_sort_requires_count                                          |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_random_sort_is_reproducible                                        |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_cli_defaults_and_sort_parser                                       |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_intermediate_cleanup_is_strictly_scoped_to_run_directory           |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_merge_sqlite_shards_preserves_existing_rows                        |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_write_outputs_keeps_structured_jsonl_and_flat_csv                  |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_worker_success_and_failure_rows_use_platform_key                   |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_generic_scheduler_prepares_cases_from_operator_yaml                |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_public_batch_api_returns_input_order_and_fail_fast_state           |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_public_batch_api_rejects_unknown_or_misaligned_input_dtypes        |      0 |       1 |        1 |
| tests/flagtune/test_pretune.py                  | test_sqlite_url_classification                                          |      0 |       1 |        1 |
| TOTAL                                           |                                                                         |    285 |      60 |      345 |
