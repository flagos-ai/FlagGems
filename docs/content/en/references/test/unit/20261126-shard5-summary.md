|                   filepath                    |                                 function                                 | passed | failed | skipped | SUBTOTAL |
| --------------------------------------------- | ------------------------------------------------------------------------ | -----: | -----: | ------: | -------: |
| tests/test_im2col.py                          | test_im2col                                                              |      1 |      0 |       0 |        1 |
| tests/test_conv1d.py                          | test_conv1d                                                              |      1 |      0 |       0 |        1 |
| tests/test_conv1d.py                          | test_conv1d_padding                                                      |      1 |      0 |       0 |        1 |
| tests/test_conv1d.py                          | test_conv1d_dilation                                                     |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                   | test_fmax_backward                                                       |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                   | test_fmax_backward_one_input                                             |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                   | test_fmax_backward_grad_fn                                               |      1 |      0 |       0 |        1 |
| tests/test_arange.py                          | test_arange                                                              |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_                                                     |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_default_params                                       |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_inplace_semantics                                    |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_value_range                                          |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_batched_key                                          |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_stream_continuity                                    |      1 |      0 |       0 |        1 |
| tests/test__philox_uniform_.py                | test_philox_uniform_large_offset                                         |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_positive_definite                                            |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_batch                                                        |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_fp64_large_batch_4x4                                         |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_fp64_large_batch_4x4_nonfinite                               |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_fp64_large_batch_4x4_stable_lu                               |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_fp64_large_batch_4x4_pathological                            |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_negative_and_singular                                        |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_tiny_nonzero_pivots                                          |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_noncontiguous_and_exported_api                               |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_empty                                                        |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_rejects_complex                                              |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_rejects_autograd                                             |      1 |      0 |       0 |        1 |
| tests/test_logdet.py                          | test_logdet_errors                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_dtype_and_public_api                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_integer_input_with_float_dtype                       |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_complex                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_noncontiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_scalar                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_special_values                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg__powsum.py                  | test_linalg__powsum_invalid_dim                                          |      1 |      0 |       0 |        1 |
| tests/test_pairwise_distance.py               | test_pairwise_distance_accuracy                                          |      1 |      0 |       0 |        1 |
| tests/test_pairwise_distance.py               | test_pairwise_distance_broadcast                                         |      1 |      0 |       0 |        1 |
| tests/test_pairwise_distance.py               | test_pairwise_distance_ndim3plus                                         |      1 |      0 |       0 |        1 |
| tests/test_pairwise_distance.py               | test_pairwise_distance_broadcast_ndim3plus                               |      1 |      0 |       0 |        1 |
| tests/test_pairwise_distance.py               | test_pairwise_distance_fp64                                              |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d                                                 |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_geometry                                        |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_int_vs_triple_args                              |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_unbatched_input                                 |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_non_contiguous                                  |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_out                                             |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_out_unbatched                                   |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_empty_output                                    |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_empty_batch                                     |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_zero_in_channels                                |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_dilated3d.py             | test_slow_conv_dilated3d_invalid_arguments                               |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_accuracy_flipud                                                     |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_accuracy_flipud_non_float                                           |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_accuracy_flipud_complex                                             |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_accuracy_flipud_autograd                                            |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_accuracy_flipud_noncontiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                          | test_flipud_rejects_zero_dimensional_input                               |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor                                                 |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_views                                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_invalid_size                                    |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_negative_dimension_message                      |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_pin_memory_ignored                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_non_strided_layout_rejected                     |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dtype_matrix                                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_resizes_mismatched_out                      |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_non_contiguous_gapped                       |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_transposed_view                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_complex_gapped                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_zero_size                                   |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_cpu                                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_cpu_non_contiguous                          |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_zero_tensor_rejected                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_requires_grad_rejected                      |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_meta                                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_negative_size                               |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_registered_dispatch_path                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_registered_dispatch_path                    |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dispatch_non_contiguous                     |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dispatch_complex                            |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dispatch_complex_non_contiguous             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dispatch_zero_tensor_rejected               |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_downstream_dispatch                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_inplace_view_rejected                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py             | test_efficientzerotensor_out_dispatch_resize_growth                      |      1 |      0 |       0 |        1 |
| tests/test_var_mean.py                        | test_var_mean                                                            |      1 |      0 |       0 |        1 |
| tests/test_cdist.py                           | test_cdist                                                               |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csc_tensor_args.py | test_validate_sparse_csc_tensor_args                                     |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csc_tensor_args.py | test_validate_sparse_csc_tensor_args_invalid_indices                     |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csc_tensor_args.py | test_validate_sparse_csc_tensor_args_invalid_metadata                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_rejects_strided_self                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_rejects_shape_mismatch                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_rejects_unknown_reduce                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_rejects_prod                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_from_strided_buffer                              |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_empty_nested                                     |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_cpu_nested                                       |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_non_nested_error                                 |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_registered_on_autograd_key                       |      1 |      0 |       0 |        1 |
| tests/test_index_select_backward.py           | test_index_select_backward                                               |      1 |      0 |       0 |        1 |
| tests/test_index_select_backward.py           | test_index_select_backward_1d                                            |      1 |      0 |       0 |        1 |
| tests/test_functional_sym_constrain_range.py  | test_functional_sym_constrain_range                                      |      1 |      0 |       0 |        1 |
| tests/test_clamp_min.py                       | test_clamp_min                                                           |      1 |      0 |       0 |        1 |
| tests/test_clamp_min.py                       | test_clamp_min_                                                          |      1 |      0 |       0 |        1 |
| tests/test_xor.py                             | test_xor                                                                 |      1 |      0 |       0 |        1 |
| tests/test_xor.py                             | test_xor_scalar                                                          |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_unsupported_shell_input_dtype                                  |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_finite_extremes                                                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_window_promotes_low_precision_input                            |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft                                                                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_real_output_and_complex_window_promotion                       |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_empty_signal_gradients                                         |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_nonfinite                                                      |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_nonfinite_window_support                                       |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_empty_batch_or_signal_follows_native                           |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_invalid_inputs                                                 |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_center_rejects_explicit_align                                  |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_default_align_odd_window_native_boundary                       |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_empty_batch_gradients                                          |      1 |      0 |       0 |        1 |
| tests/test_stft.py                            | test_stft_center_empty_batch_gradients                                   |      1 |      0 |       0 |        1 |
| tests/test_round.py                           | test_round                                                               |      1 |      0 |       0 |        1 |
| tests/test_round.py                           | test_round_                                                              |      1 |      0 |       0 |        1 |
| tests/test_round.py                           | test_round_out                                                           |      1 |      0 |       0 |        1 |
| tests/test_round.py                           | test_round_decimals                                                      |      1 |      0 |       0 |        1 |
| tests/test_round.py                           | test_round_half_to_even                                                  |      1 |      0 |       0 |        1 |
| tests/test_outplace_fused_experts.py          | test_outplace_fused_experts_accuracy                                     |      1 |      0 |       0 |        1 |
| tests/test_outplace_fused_experts.py          | test_outplace_fused_experts_does_not_modify_input                        |      1 |      0 |       0 |        1 |
| tests/test_miopen_batch_norm_backward.py      | test_miopen_batch_norm_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_no_update.py            | test_batch_norm_no_update                                                |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                   | test_diagonal_copy                                                       |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                   | test_diagonal_copy_int                                                   |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                   | test_diagonal_copy_low_precision                                         |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                   | test_diagonal_copy_empty                                                 |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                   | test_diagonal_copy_single_element                                        |      1 |      0 |       0 |        1 |
| tests/test_frexp.py                           | test_frexp                                                               |      1 |      0 |       0 |        1 |
| tests/test_igammac_.py                        | test_special_gammaincc                                                   |      1 |      0 |       0 |        1 |
| tests/test_igammac_.py                        | test_igammac_                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_empty                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_noncontiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_identity                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_degenerate_eigenvalues                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_ascending_order                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_trace_is_preserved                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_reads_requested_triangle                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_default_uplo_is_lower                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_batch_independence                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_invalid_shape                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_invalid_uplo                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_rejects_complex                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_fp64_preserved                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_autograd_input_is_detached                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_out_resize                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                 | test_linalg_eigvalsh_out_rejects_mismatched_dtype                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices                                                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices                                                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_default_options                                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_default_options                                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_sparse_int64_extremes                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_sparse_int64_extremes                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_far_offset_fast_paths                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_far_offset_fast_paths                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_int32_emitted_value_boundary                           |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_int32_emitted_value_boundary                           |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_rejects_unrepresentable_int32_value                    |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_rejects_unrepresentable_int32_value                    |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_invalid_arguments                                      |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_invalid_arguments                                      |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_rejects_size_overflow                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_rejects_size_overflow                                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_tril_indices_rejects_allocation_size_overflow                       |      1 |      0 |       0 |        1 |
| tests/test_triangular_indices.py              | test_triu_indices_rejects_allocation_size_overflow                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_fast_float32                                           |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_rank1_float32                                          |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_rank1_zero_basis                                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_gram_ill_conditioned_orthonormal                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_gram_zero_and_repeated_singular_values                 |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_tiny_rank_degenerate_inputs                            |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_rank2_boundary_degenerate_inputs                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_fallback_modes                                         |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_non_contiguous_empty_and_complex                       |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_low_precision_float32_route                            |      1 |      0 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_some_false_reconstruction                              |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                      | test_native_batch_norm                                                   |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                      | test_batch_norm                                                          |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                      | test_batch_norm_backward                                                 |      1 |      0 |       0 |        1 |
| tests/test_blackman_window.py                 | test_blackman_window                                                     |      1 |      0 |       0 |        1 |
| tests/test_blackman_window.py                 | test_blackman_window_periodic                                            |      1 |      0 |       0 |        1 |
| tests/test_euclidean_dist.py                  | test_euclidean_dist                                                      |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char                                                           |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_strided                                                   |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_fractional                                                |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_requires_grad                                             |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_complex                                                   |      1 |      0 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_integer_limits                                            |      1 |      0 |       0 |        1 |
| tests/test_hypot_.py                          | test_hypot_                                                              |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                       | test_polygamma                                                           |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                       | test_polygamma_wide_domain                                               |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                       | test_polygamma_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                       | test_polygamma_                                                          |      1 |      0 |       0 |        1 |
| tests/test_dswiglu.py                         | test_dswiglu                                                             |      1 |      0 |       0 |        1 |
| tests/test_diag.py                            | test_diag                                                                |      1 |      0 |       0 |        1 |
| tests/test_log_.py                            | test_log_                                                                |      1 |      0 |       0 |        1 |
| tests/test_log_.py                            | test_log_large_inplace                                                   |      1 |      0 |       0 |        1 |
| tests/test_log_.py                            | test_log_special_values                                                  |      1 |      0 |       0 |        1 |
| tests/test_log_.py                            | test_log_noncontiguous                                                   |      1 |      0 |       0 |        1 |
| tests/test_log_.py                            | test_log_empty                                                           |      1 |      0 |       0 |        1 |
| tests/test_pad_circular.py                    | test_pad_circular                                                        |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_elemt.py                | test_batch_norm_elemt                                                    |      1 |      0 |       0 |        1 |
| tests/test_matrix_power.py                    | test_matrix_power                                                        |      1 |      0 |       0 |        1 |
| tests/test_matrix_power.py                    | test_matrix_power_negative                                               |      1 |      0 |       0 |        1 |
| tests/test_matrix_power.py                    | test_matrix_power_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_isneginf.py                        | test_isneginf                                                            |      1 |      0 |       0 |        1 |
| tests/test_isneginf.py                        | test_isneginf_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_fused_moe_reduction.py             | test_fused_moe_repeat_and_graph_reuse                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_moe_reduction.py             | test_fused_moe_reduction_reference                                       |      1 |      0 |       0 |        1 |
| tests/test_embedding_spmdm_grad.py            | test_embedding_spmdm_grad_matches_reference                              |      1 |      0 |       0 |        1 |
| tests/test_embedding_spmdm_grad.py            | test_embedding_spmdm_grad_validation                                     |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_tensor_tensor_broadcast               |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_fast_degrees                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_special_x                             |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_promotion_and_nonfinite_degree        |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_large_degree_fast_exit_and_recurrence |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_is_not_differentiable                 |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_n_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_x_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_scalar_noncontiguous_layouts          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_scalar_float64_precision              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_out                                   |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_out_resize_and_noncontiguous          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_n_scalar_out                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_x_scalar_out                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py   | test_special_laguerre_polynomial_l_rejects_unsupported_compute_dtype     |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit                                                       |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_eps_none                                              |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_extreme_values                                        |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_nan_input                                             |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_inf_input                                             |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_of_range                                          |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_eps_none                                          |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_out_of_range                                      |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_extreme_values                                    |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_nan_input                                         |      1 |      0 |       0 |        1 |
| tests/test_special_logit.py                   | test_special_logit_out_inf_input                                         |      1 |      0 |       0 |        1 |
| tests/test_vector_norm.py                     | test_accuracy_vectornorm                                                 |      1 |      0 |       0 |        1 |
| tests/test_deg2rad_backward.py                | test_deg2rad_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_deg2rad_backward.py                | test_deg2rad_backward_functional                                         |      1 |      0 |       0 |        1 |
| tests/test_deg2rad_backward.py                | test_deg2rad_backward_non_contiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_unique_dim.py                      | test_unique_dim_int                                                      |      1 |      0 |       0 |        1 |
| tests/test_unique_dim.py                      | test_unique_dim_float                                                    |      1 |      0 |       0 |        1 |
| tests/test_unique_dim.py                      | test_unique_dim_returns_three_tensors                                    |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py           | test_slow_conv_transpose3d                                               |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py           | test_slow_conv_transpose3d_bias                                          |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py           | test_slow_conv_transpose3d_strided_bias                                  |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py           | test_slow_conv_transpose3d_output_padding_dilation                       |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py           | test_slow_conv_transpose3d_scalar_stride                                 |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_single                               |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_multi                                |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_infers_terminal_total_L              |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_empty_output                         |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_noncontiguous_dense                  |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_float64_preserves_bits               |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_float16_preserves_bits               |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_validation                           |      1 |      0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py  | test_padded_dense_to_jagged_forward_revalidates_mutated_offsets          |      1 |      0 |       0 |        1 |
| tests/test_nan_to_num_.py                     | test_nan_to_num_                                                         |      1 |      0 |       0 |        1 |
| tests/test_lshift.py                          | test_lshift_tensor                                                       |      1 |      0 |       0 |        1 |
| tests/test_lshift.py                          | test_lshift_scalar                                                       |      1 |      0 |       0 |        1 |
| tests/test_lshift.py                          | test_lshift_output_overloads                                             |      1 |      0 |       0 |        1 |
| tests/test_isfinite.py                        | test_isfinite                                                            |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices                                                        |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_bsc                                                    |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_batched                                                |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_int32_buffer                                           |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_alias_and_mutation                                     |      1 |      0 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_unsupported_layout_raises                              |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_csr                                      |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_bsr                                      |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_batched                                  |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_fresh_copy_contract                      |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_row_index_dtype                          |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_empty                                    |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_single_row                               |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_values_dtype                             |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_contiguous_metadata                      |      1 |      0 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_repeated_dispatch                        |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py  | test_upsample_lanczos2d_aa_backward                                      |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py  | test_upsample_lanczos2d_aa_backward_dtypes                               |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py  | test_upsample_lanczos2d_aa_backward_precomputed_path                     |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py  | test_upsample_lanczos2d_aa_backward_out                                  |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py  | test_upsample_lanczos2d_aa_backward_noncontiguous_grad                   |      1 |      0 |       0 |        1 |
| tests/test_special_xlogy.py                   | test_special_xlogy                                                       |      1 |      0 |       0 |        1 |
| tests/test_special_xlogy.py                   | test_special_xlogy_                                                      |      1 |      0 |       0 |        1 |
| tests/test_special_xlogy.py                   | test_special_xlogy_tensor_scalar                                         |      1 |      0 |       0 |        1 |
| tests/test_special_xlogy.py                   | test_special_xlogy_scalar_tensor                                         |      1 |      0 |       0 |        1 |
| tests/test_rrelu_with_noise_functional.py     | test_rrelu_with_noise_functional                                         |      1 |      0 |       0 |        1 |
| tests/test_dropout.py                         | test_dropout                                                             |      1 |      0 |       0 |        1 |
| tests/test_dropout.py                         | test_native_dropout_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_cross_attention.py                 | test_cross_attention_matches_reference                                   |      1 |      0 |       0 |        1 |
| tests/test_cross_attention.py                 | test_cross_attention_mask_contract                                       |      1 |      0 |       0 |        1 |
| tests/test_cross_attention.py                 | test_fully_masked_rows_are_exact_zero                                    |      1 |      0 |       0 |        1 |
| tests/test_cross_attention.py                 | test_cross_attention_validation_contract                                 |      1 |      0 |       0 |        1 |
| tests/test_cross_attention.py                 | test_head_dimension_upper_bound                                          |      1 |      0 |       0 |        1 |
| tests/test_arm_w4a8_g128.py                   | test_w4a8_g128_packer_rejects_invalid_contracts                          |      1 |      0 |       0 |        1 |
| tests/test_erfinv_.py                         | test_erfinv                                                              |      1 |      0 |       0 |        1 |
| tests/test_erfinv_.py                         | test_erfinv_                                                             |      1 |      0 |       0 |        1 |
| tests/test_isinf.py                           | test_isinf                                                               |      1 |      0 |       0 |        1 |
| tests/test_divide.py                          | test_divide                                                              |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window                                                       |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_periodic                                              |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_beta                                                  |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_degenerate_lengths                                    |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_default_device                                        |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_only_strided_layout                                   |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_negative_length                                       |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                   | test_kaiser_window_requires_float_dtype                                  |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign                                                                |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_out                                                            |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_                                                               |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign__noncontiguous                                                 |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_special_values                                                 |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_noncontiguous                                                  |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_out_noncontiguous                                              |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_out_resizes_empty_tensor                                       |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_out_rejects_mismatched_dtype                                   |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_empty                                                          |      1 |      0 |       0 |        1 |
| tests/test_sign.py                            | test_sign_rejects_complex                                                |      1 |      0 |       0 |        1 |
| tests/test_rot90.py                           | test_rot90                                                               |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean                                                                |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_empty                                                          |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_empty_explicit_dtype                                           |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_empty_rejects_non_floating_input                               |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_empty_rejects_non_floating_dtype                               |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_dim                                                            |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_dim_empty                                                      |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_dim_empty_explicit_dtype                                       |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_dim_large_k                                                    |      1 |      0 |       0 |        1 |
| tests/test_mean.py                            | test_mean_dim_large_innerdim                                             |      1 |      0 |       0 |        1 |
| tests/test_fft_irfftn.py                      | test_fft_irfftn                                                          |      1 |      0 |       0 |        1 |
| tests/test_fft_irfftn.py                      | test_fft_irfftn_complex32                                                |      1 |      0 |       0 |        1 |
| tests/test_fft_irfftn.py                      | test_fft_irfftn_complex128                                               |      1 |      0 |       0 |        1 |
| tests/test_fft_irfftn.py                      | test_fft_irfftn_default_s                                                |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_array                                                      |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_scalarint                                                  |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_scalararray                                                |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_scalarrayint                                               |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_tensorarray                                                |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_tensorarrayint_promotion                                   |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_output_order_and_negative_dim                              |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_non_contiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_int_promotion                                              |      1 |      0 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_error_paths                                                |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py  | test_nested_tensor_from_tensor_list                                      |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_1d                                                      |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_same_shape                                              |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_dim                                                     |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_fp64                                                    |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_broadcasting                                            |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_mismatched_length                                       |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_invalid_dim                                             |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_empty_dim_skips_length_check                            |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_zero_size                                               |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_mixed_dtype                                             |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_y_broadcasting                                          |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_pair_broadcasting                                       |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_backward_single_point                                   |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_backward_empty                                          |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_registered                                              |      1 |      0 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_registered_complex                                      |      1 |      0 |       0 |        1 |
| tests/test_hardshrink.py                      | test_hardshrink                                                          |      1 |      0 |       0 |        1 |
| tests/test_hardshrink.py                      | test_hardshrink_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_operator_markers_are_declared                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_float                                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_int                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_bool                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_complex                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_rejects_unsupported_dtype                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_list_lengths                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_mixed_shapes                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_empty_and_zero_dim                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_mixed_dtypes                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_layouts                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_inplace                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_inplace_writes_through_views                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_inplace_rejects_overlap                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_inplace_rejects_promotion                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_error_parity                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_out_overload                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_foreach_unary_launch_count_is_independent_of_list_length            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_foreach_unary_launch_count_scales_with_dtype_groups                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_foreach_unary_launch_count_with_heterogeneous_numel                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_unary_complex_large                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_accuracy_foreach_abs_complex_large                                  |      1 |      0 |       0 |        1 |
| tests/test_real.py                            | test_real_complex_view_semantics                                         |      1 |      0 |       0 |        1 |
| tests/test_real.py                            | test_real_noncontiguous_complex_view                                     |      1 |      0 |       0 |        1 |
| tests/test_real.py                            | test_real_non_complex_returns_input                                      |      1 |      0 |       0 |        1 |
| tests/test_real.py                            | test_real_float8_returns_input                                           |      1 |      0 |       0 |        1 |
| tests/test_real.py                            | test_real_conjugate_view_version_tracking                                |      1 |      0 |       0 |        1 |
| tests/test_special_xlog1py.py                 | test_special_xlog1py                                                     |      1 |      0 |       0 |        1 |
| tests/test_special_xlog1py.py                 | test_special_xlog1py_x_zero                                              |      1 |      0 |       0 |        1 |
| tests/test_special_xlog1py.py                 | test_special_xlog1py_y_nan                                               |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn                                                                 |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_out                                                             |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_                                                                |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_special_values                                                  |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_complex_nonfinite_values                                        |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_noncontiguous                                                   |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_out_noncontiguous                                               |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_out_resizes_empty_tensor                                        |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_out_rejects_mismatched_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_empty                                                           |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros                                                           |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_inherits_metadata                                         |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_source_is_not_read_or_mutated                             |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_dtype_override                                            |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_explicit_device                                           |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_sparse_layout                                             |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_size_argument_forms                                       |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_bool_extent                                               |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_out_resizes_buffer                                        |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_out_writes_through_strided_buffer                         |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_empty_source_still_supplies_defaults                      |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_rejects_non_tensor_self                                   |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                       | test_new_zeros_rejects_non_sequence_size                                 |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py          | test_keeps_declared_name_when_backend_is_available                       |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py          | test_uses_registered_privateuse1_name                                    |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py          | test_env_override_wins                                                   |      1 |      0 |       0 |        1 |
| tests/test_device_name_resolution.py          | test_falls_back_to_declared_name_without_privateuse1                     |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool3d_backward.py    | test_adaptive_avg_pool3d_backward_grad_input                             |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy                                             |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy_scalar                                      |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy_non_contiguous                              |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy_same_dim_does_not_alias                     |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_transpose_copy_invalid_dims                                         |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet                                                             |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_scaled_identity                                             |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_singular                                                    |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_1x1                                                         |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_empty_matrix                                                |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_empty_batch                                                 |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_multi_dim_batch                                             |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_nan_inf                                                     |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_invalid_inputs                                              |      1 |      0 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_identity                                                    |      1 |      0 |       0 |        1 |
| tests/test_prelu_kernel.py                    | test_prelu_kernel                                                        |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward_out                                            |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward_out_noncontiguous                              |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward_contiguous_buffer                              |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward_out_contiguous_buffer                          |      1 |      0 |       0 |        1 |
| tests/test_log_sigmoid_backward.py            | test_log_sigmoid_backward_via_autograd                                   |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2                                                                |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_forward_requires_grad                                          |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_backward                                                       |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_backward_special_values                                        |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_backward_non_contiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_backward_permuted_3d                                           |      1 |      0 |       0 |        1 |
| tests/test_log2.py                            | test_log2_                                                               |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py         | test_backward_real_dispatch                                              |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py         | test_backward_uses_gems_compute                                          |      1 |      0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py         | test_inplace_saved_version_is_checked                                    |      1 |      0 |       0 |        1 |
| tests/test_col2im.py                          | test_col2im                                                              |      1 |      0 |       0 |        1 |
| tests/test_fmax.py                            | test_fmax                                                                |      1 |      0 |       0 |        1 |
| tests/test_fmax.py                            | test_fmax_with_nan                                                       |      1 |      0 |       0 |        1 |
| tests/test_fmax.py                            | test_fmax_out                                                            |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod                                                             |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_dtype                                                       |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_low_precision_multitile                                     |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace                                                     |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_long_row_int16                                      |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_low_precision_multitile                             |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_non_contiguous                                      |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_non_contiguous_tile_boundary                        |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_dtype_mismatch                                      |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_inplace_bool_unsupported                                    |      1 |      0 |       0 |        1 |
| tests/test_cumprod.py                         | test_cumprod_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_special_airy_ai.py                 | test_special_airy_ai                                                     |      1 |      0 |       0 |        1 |
| tests/test_special_airy_ai.py                 | test_special_airy_ai_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_accuracy_poisson_nll_loss                                           |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_accuracy_poisson_nll_loss_broadcast_noncontiguous                   |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_accuracy_poisson_nll_loss_empty                                     |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_accuracy_poisson_nll_loss_boundary                                  |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_mixed_dtype_promotion                              |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_mixed_dtype_exceptions                             |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_bool_input_nonlog_rejected                         |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_invalid_broadcast                                  |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_special_eps                                        |      1 |      0 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_reduction_boundary                                 |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_tensor                                                  |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_scalar                                                  |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_scalar_out                                              |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_dim_none_keepdim_shape                                  |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_maximum_reduction_one_valid                             |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_large_q_is_tiled                                        |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_gather_interpolations                                   |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_noncontiguous_out                                       |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_all_nan_and_path_boundaries                             |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_special_values_and_scalar_tensor_q                      |      1 |      0 |       0 |        1 |
| tests/test_nanquantile.py                     | test_nanquantile_errors                                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl                                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_selector                                      |      0 |      1 |       0 |        1 |
| tests/test_sparse_mm_reduce_impl.py           | test_sparse_mm_reduce_impl_dense_noncontiguous                           |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_accuracy_nested_tensor_size                                         |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_dispatch_stability                               |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_size.py              | test_nested_tensor_size_scalar_nesting                                   |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_center                                                         |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_torch_dispatch                                                 |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_gradients                                                      |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_padding_and_promotion_gradients                                |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_lazy_views                                                     |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_chunked_forward_and_backward                                   |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_fused_complex_window_gradients                                 |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_center_fused_complex_window_gradients                          |      0 |      1 |       0 |        1 |
| tests/test_svd.py                             | test_accuracy_svd_complex64_reconstruction                               |      0 |      1 |       0 |        1 |
| tests/test_cast_Char.py                       | test_cast_Char_negative_view                                             |      0 |      1 |       0 |        1 |
| tests/test_log_.py                            | test_log_unsupported_dtype_raises                                        |      0 |      1 |       0 |        1 |
| tests/test_nll_loss2d.py                      | test_nll_loss2d                                                          |      0 |      1 |       0 |        1 |
| tests/test_nll_loss2d.py                      | test_nll_loss2d_backward                                                 |      0 |      1 |       0 |        1 |
| tests/test_embedding_spmdm_grad.py            | test_embedding_spmdm_grad_noncontiguous_inputs                           |      0 |      1 |       0 |        1 |
| tests/test_avg_pool3d.py                      | test_avg_pool3d                                                          |      0 |      1 |       0 |        1 |
| tests/test_avg_pool3d.py                      | test_avg_pool3d_backward                                                 |      0 |      1 |       0 |        1 |
| tests/test_ccol_indices.py                    | test_ccol_indices_dense_and_coo_rejected                                 |      0 |      1 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_unsupported_layouts                      |      0 |      1 |       0 |        1 |
| tests/test_crow_indices_copy.py               | test_accuracy_crow_indices_copy_dispatch_sentinel                        |      0 |      1 |       0 |        1 |
| tests/test_cross_attention.py                 | test_noncontiguous_qkv_and_mask                                          |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                  | test_linalg_svdvals                                                      |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                  | test_linalg_svdvals_batch                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                  | test_linalg_svdvals_non_contiguous                                       |      0 |      1 |       0 |        1 |
| tests/test_gradient.py                        | test_gradient_quadratic_exactness                                        |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py  | test_dispatch_reaches_gems_impl                                          |      0 |      1 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_complex                                                 |      0 |      1 |       0 |        1 |
| tests/test_trapezoid.py                       | test_trapezoid_x_complex_mixed                                           |      0 |      1 |       0 |        1 |
| tests/test_foreach_unary.py                   | test_foreach_unary_registration_is_live                                  |      0 |      1 |       0 |        1 |
| tests/test_real.py                            | test_real_uses_flag_gems_view_path                                       |      0 |      1 |       0 |        1 |
| tests/test_real.py                            | test_real_autograd_view_relation                                         |      0 |      1 |       0 |        1 |
| tests/test_real.py                            | test_real_conjugate_nonleaf_inplace_view_replay                          |      0 |      1 |       0 |        1 |
| tests/test_sgn.py                             | test_sgn_complex_extreme_values                                          |      0 |      1 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy_complex                                     |      0 |      1 |       0 |        1 |
| tests/test_transpose_copy.py                  | test_accuracy_transpose_copy_float8                                      |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_complex                                                     |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_near_singular_scaled                                        |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_non_contiguous                                              |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_large                                                       |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_large_1025                                                  |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_backward                                                    |      0 |      1 |       0 |        1 |
| tests/test_slogdet.py                         | test_slogdet_batched_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_complex                                            |      0 |      1 |       0 |        1 |
| tests/test_poisson_nll_loss.py                | test_poisson_nll_loss_complex_target_full_rejected                       |      0 |      1 |       0 |        1 |
| tests/test_stft.py                            | test_stft_ascend_complex_half_promotion                                  |      0 |      0 |       1 |        1 |
| tests/test_cslt_sparse_mm.py                  | test_accuracy_cslt_sparse_mm                                             |      0 |      0 |       1 |        1 |
| tests/test_cslt_sparse_mm.py                  | test_accuracy_cslt_sparse_mm_with_alpha                                  |      0 |      0 |       1 |        1 |
| tests/test_cslt_sparse_mm.py                  | test_accuracy_cslt_sparse_mm_transpose                                   |      0 |      0 |       1 |        1 |
| tests/test_arm_w4a8_g128.py                   | test_w4a8_g128_matches_reference                                         |      0 |      0 |       1 |        1 |
| tests/test_arm_w4a8_g128.py                   | test_w4a8_g128_decode_is_repeatable                                      |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_fused_inv_rope_fp8_quant                                            |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_output_layout                                                       |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_per_group_contiguity                                                |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_scales_are_power_of_two                                             |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_large_values                                                        |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_nope_dims_unchanged                                                 |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_zero_positions                                                      |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_dequant_numerical_accuracy                                          |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_unfused_path                                                        |      0 |      0 |       1 |        1 |
| tests/test_fused_inv_rope_fp8_quant.py        | test_with_real_deepseek_v4_rope                                          |      0 |      0 |       1 |        1 |
| TOTAL                                         |                                                                          |    506 |     49 |      16 |      571 |
|                        filepath                         |                                function                                 | passed | skipped | SUBTOTAL |
| ------------------------------------------------------- | ----------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_median_dim.py                                | test_median_dim                                                         |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_keepdim_shapes                                          |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_ties                                                    |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_strided_nonlast_reduction                               |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_even_count_picks_lower_middle                           |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_propagates_nan                                          |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_scalar_input                                            |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_empty_reduction_dim_raises                              |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_empty_but_nonzero_reduction                             |      1 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_named_dim_preserves_names                               |      1 |       0 |        1 |
| tests/test_unsafe_masked_index.py                       | test_unsafe_masked_index                                                |      1 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                     | test_fractional_max_pool2d                                              |      1 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                     | test_fractional_max_pool2d_backward                                     |      1 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py              | test_case_list_contract_excludes_private_materialization_state          |      1 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py              | test_shapes_and_parameters_roundtrip                                    |      1 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py              | test_metadata_does_not_silently_stringify_non_json_objects              |      1 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py              | test_plan_defaults_are_not_shared                                       |      1 |       0 |        1 |
| tests/test_baddbmm_.py                                  | test_baddbmm_                                                           |      1 |       0 |        1 |
| tests/test_linalg_ldl_factor_ex.py                      | test_linalg_ldl_factor_ex                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_accuracy_2d                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_accuracy_3d                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_shape_consistency                                         |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_custom_padding                                            |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_default_inf_padding                                       |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_single_batch                                              |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_short_sequences                                           |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_3d_single_batch                                           |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_3d_short_sequences                                        |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_block_sizes                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_int8_custom_padding                                       |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_int8_valid_region_with_default_padding                    |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_int8_out_of_range_padding_is_unvalidated                  |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_contiguous                                     |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_non_contiguous                                 |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_transposed_and_expanded                        |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_levels                                         |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_mutation_through_view                          |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_dispatch_stability                             |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_inference_tensor                               |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_scalar_and_empty                               |      1 |       0 |        1 |
| tests/test__fw_primal.py                                | test_accuracy__fw_primal_dtype_dispatch                                 |      1 |       0 |        1 |
| tests/test_slow_conv2d_forward.py                       | test_slow_conv2d_forward                                                |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_grow                                         |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_identity                                     |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_empty_to_dense_dims                          |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_empty_change_dims                            |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_nonempty_dense_template_raises               |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__coo_shrink_raises                                |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__unsupported_layouts                              |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__storage_alias                                    |      1 |       0 |        1 |
| tests/test_resize_as_sparse_.py                         | test_resize_as_sparse__hybrid                                           |      1 |       0 |        1 |
| tests/test_relu.py                                      | test_relu                                                               |      1 |       0 |        1 |
| tests/test_relu.py                                      | test_relu_                                                              |      1 |       0 |        1 |
| tests/test_nested_tensor_from_mask.py                   | test_nested_tensor_from_mask                                            |      1 |       0 |        1 |
| tests/test_nested_tensor_from_mask.py                   | test_nested_tensor_from_mask_no_mask_check                              |      1 |       0 |        1 |
| tests/test_nested_tensor_from_mask.py                   | test_nested_tensor_from_mask_mask_check_raises                          |      1 |       0 |        1 |
| tests/test_dimV.py                                      | test_accuracy_dimV_coo                                                  |      1 |       0 |        1 |
| tests/test_dimV.py                                      | test_accuracy_dimV_matches_indices_formula                              |      1 |       0 |        1 |
| tests/test_dimV.py                                      | test_accuracy_dimV_unsupported_layouts_raise                            |      1 |       0 |        1 |
| tests/test_dimV.py                                      | test_accuracy_dimV_dispatch_stability                                   |      1 |       0 |        1 |
| tests/test_neg.py                                       | test_neg                                                                |      1 |       0 |        1 |
| tests/test_neg.py                                       | test_neg_                                                               |      1 |       0 |        1 |
| tests/test_max_unpool3d.py                              | test_max_unpool3d                                                       |      1 |       0 |        1 |
| tests/test_max_unpool3d.py                              | test_max_unpool3d_backward                                              |      1 |       0 |        1 |
| tests/test_quantile.py                                  | test_quantile                                                           |      1 |       0 |        1 |
| tests/test_quantile.py                                  | test_quantile_dim                                                       |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double                                                        |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_strided                                                |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_special_values                                         |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_integer_limits                                         |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_autograd                                               |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_complex                                                |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_negative_view                                          |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_cpu                                                    |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_sparse                                                 |      1 |       0 |        1 |
| tests/test_relu6_.py                                    | test_relu6_                                                             |      1 |       0 |        1 |
| tests/test_archsinh.py                                  | test_arcsinh_                                                           |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex                                                 |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_empty_batch                                     |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_empty_matrix                                    |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_noncontiguous                                   |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_triangle_is_isolated                            |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_not_positive_definite                           |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_check_errors_raises                             |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_check_errors_passes_when_definite               |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_default_arguments                               |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_binds_schema_keywords                           |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_invalid_input                                   |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_rejects_complex                                 |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_beyond_max_order                                |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_max_order_supported                             |      1 |       0 |        1 |
| tests/test_linalg_cholesky_ex.py                        | test_linalg_cholesky_ex_autograd_input_is_detached                      |      1 |       0 |        1 |
| tests/test_tanh.py                                      | test_tanh                                                               |      1 |       0 |        1 |
| tests/test_tanh.py                                      | test_tanh_                                                              |      1 |       0 |        1 |
| tests/test_tanh.py                                      | test_tanh_backward                                                      |      1 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py           | test_validate_sparse_csr_tensor_args                                    |      1 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py           | test_validate_sparse_csr_tensor_args_invalid_indices                    |      1 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py           | test_validate_sparse_csr_tensor_args_invalid_metadata                   |      1 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py           | test_validate_sparse_csr_tensor_args_dispatch                           |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_basic                                      |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_batch_sizes                                |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_with_lengths                               |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_minmax_seqlen                              |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_ragged_idx                                 |      1 |       0 |        1 |
| tests/test_nested_view_from_jagged.py                   | test_nested_view_from_jagged_int32_offsets                              |      1 |       0 |        1 |
| tests/test_bitwise_or_scalar.py                         | test_bitwise_or_scalar                                                  |      1 |       0 |        1 |
| tests/test_bitwise_or_scalar.py                         | test_bitwise_or_scalar_                                                 |      1 |       0 |        1 |
| tests/test_triton_lang_helper.py                        | test_patch_missing_nearbyint_uses_rint                                  |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell                                       |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_noncontiguous                         |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_empty                                 |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_hidden_bias_without_input_bias        |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_rejects_input_bias_only               |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_rejects_unsupported_dtype                      |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_out                                   |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_accuracy_thnn_fused_gru_cell_out_resize                            |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_out_aliases_input                              |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_out_rejects_invalid_dtype                      |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_out_rejects_invalid_device                     |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_out_rejects_overlapping_outputs                |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                       | test_thnn_fused_gru_cell_backward_end_to_end                            |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor                                                |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out                                            |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_half_way_values                                |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_non_contiguous                                 |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_rejects_non_float32                            |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_empty                                          |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_dtype_mismatch                             |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_shape_mismatch                             |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_memory_format                                  |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_public_entry_keeps_layout                      |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_zero_point_range                               |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_contiguous_unchanged                           |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_matches_native_layout                      |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_preserves_offset_and_strides               |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_mismatched_qparams                         |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_rejects_per_channel_out                    |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_aliasing_input                             |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_uses_triton_not_native_copy                |      1 |       0 |        1 |
| tests/test_quantize_per_tensor.py                       | test_quantize_per_tensor_out_rejects_internally_overlapping_out         |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_accuracy                                                 |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_accuracy_no_bias                                         |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_projection                                               |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_dropout_training                                         |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_dropout_state                                            |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_dropout_eval                                             |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_dropout_applied                                          |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_cy_shape                                                 |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_lstm_no_cx                                               |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_weight_buf                                               |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_packed                                                   |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_packed_bidirectional                                     |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_dispatch                                                 |      1 |       0 |        1 |
| tests/test_cudnn_rnn.py                                 | test_cudnn_rnn_deterministic                                            |      1 |       0 |        1 |
| tests/test_addr_.py                                     | test_addr_                                                              |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                           | test_embedding_spmdm_matches_embedding_bag                              |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                           | test_embedding_spmdm_index_dtypes                                       |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                           | test_embedding_spmdm_noncontiguous_inputs                               |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                           | test_embedding_spmdm_validation                                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_default_identity                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_float64_preserves_small_singular_value |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_float64_tolerance_precision            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_well_separated_spectrum                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_rank_deficient                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_nonempty_zero_matrix                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_shapes                                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_matches_adjoint                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_aah_svd_matches_hermitian              |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_tolerance_combinations                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_scalar_tolerance_types                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_legacy_float_tolerance                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_per_batch_tolerance                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_broadcast_tolerance                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_false                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_true                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_uses_lower_triangle          |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_order_33                     |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_shapes                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_general_large_shapes                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_general_dense_low_rank                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_dense_low_rank               |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_empty                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_out                                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_tensor_tolerance_out                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_legacy_float_tolerance_out             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance_out            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_out_wrong_dtype                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_out_wrong_device                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_out_wrong_shape_warns_and_resizes      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_official_dtype_contract                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_rejects_complex_tolerance              |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_nonsquare_tail_energy                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_nonsquare_lowrank                      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_strict_threshold             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_negative_tolerances                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_deflated_spectrum            |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_extreme_scales                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_mixed_magnitude_batch                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_tolerance_precision                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_fp64_critical_spectrum                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_ignores_strict_upper   |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_deflated               |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_extreme_scales         |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_critical_spectrum      |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_batched                |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_large_hermitian_repeatable             |      1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_empty_validates_tolerances             |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take                                                               |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_out                                                           |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_negative_index                                                |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_out_of_bounds                                                 |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_noncontiguous                                                 |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_out_noncontiguous                                             |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_out_resizes_empty_tensor                                      |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_out_rejects_mismatched_dtype                                  |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_empty                                                         |      1 |       0 |        1 |
| tests/test_take.py                                      | test_take_multidim_index                                                |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum                                                            |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_cpu_scalar_operand                                         |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_cpu_scalar_precision                                       |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_rejects_non_scalar_cpu_operand                             |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward                                                   |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_ties                                              |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_special_values                                    |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_signed_zeros                                      |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_single_sided                                      |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_broadcast                                         |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_expanded_grad                                     |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_non_contiguous                                    |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_mixed_dtype                                       |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_float64                                           |      1 |       0 |        1 |
| tests/test_maximum.py                                   | test_maximum_backward_empty_broadcast                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv                                                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_empty                                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_round_trip                                              |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_row_swap_pivoting                                       |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_batch_independence                                      |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_binds_single_argument                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_invalid_shape                                           |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_rejects_low_precision                                   |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_singular_does_not_hang_or_crash                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_rejects_complex                                         |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_fp64_preserved                                          |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_autograd_input_is_detached                              |      1 |       0 |        1 |
| tests/test_isposinf.py                                  | test_isposinf                                                           |      1 |       0 |        1 |
| tests/test_convolution_backward_overrideable.py         | test_convolution_backward_overrideable                                  |      1 |       0 |        1 |
| tests/test_convolution_backward_overrideable.py         | test_convolution_backward_overrideable_out                              |      1 |       0 |        1 |
| tests/test_view_as_complex.py                           | test_view_as_complex_accuracy                                           |      1 |       0 |        1 |
| tests/test_view_as_complex.py                           | test_view_as_complex_2d_accuracy                                        |      1 |       0 |        1 |
| tests/test_view_as_complex.py                           | test_view_as_complex_3d_accuracy                                        |      1 |       0 |        1 |
| tests/test_gelu_and_mul.py                              | test_gelu_and_mul                                                       |      1 |       0 |        1 |
| tests/test_arctanh.py                                   | test_arctanh                                                            |      1 |       0 |        1 |
| tests/test_arctanh.py                                   | test_arctanh_                                                           |      1 |       0 |        1 |
| tests/test_arctanh.py                                   | test_arctanh_out                                                        |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_vector_1d                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_vector_nd                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_matrix_2d                                                          |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_matrix_2d_svd                                                      |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_matrix_batch                                                       |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_keepdim_dtype                                                      |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_bad_dim_length_rejected                                            |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_ord_without_dim_requires_1d_2d                                     |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_unsupported_matrix_ord_rejected                                    |      1 |       0 |        1 |
| tests/test_linalg_norm.py                               | test_fro_on_1d_rejected                                                 |      1 |       0 |        1 |
| tests/test_normal_functional.py                         | test_normal_functional                                                  |      1 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py | test_scaled_dot_product_attention_math_for_mps                          |      1 |       0 |        1 |
| tests/test_values.py                                    | test_accuracy_values_coo                                                |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_mutation_through_view                                       |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_strided_raises                                              |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_sparse_compressed                                           |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_unsupported_layouts_reject_like_native                      |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_non_float_dtypes_alias                                      |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_int_hybrid_alias                                            |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_requires_grad_nondifferentiable_contract                    |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_sparse_requires_grad_view_contract                          |      1 |       0 |        1 |
| tests/test_values.py                                    | test_values_dispatch_stability                                          |      1 |       0 |        1 |
| tests/test_bitwise_left_shift_.py                       | test_bitwise_left_shift_                                                |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril                                                               |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_noncontiguous                                                 |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_wide_exact_row_dispatch                                       |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_tiny_batched_tile_dispatch                                    |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_tiny_batched_tile_correctness                                 |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out                                                           |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_resizes                                                   |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_aliases_input                                             |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_noncontiguous_out                                         |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_sliced_leading_batch_out                                  |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_extreme_diagonal_noncontiguous_out                        |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_out_strided_dispatch_guards                                   |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace                                                       |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_noncontiguous                                         |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_strided_batched_view                                  |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_expanded_view                                         |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_overlapping_as_strided_view                           |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_empty                                                         |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_empty                                                 |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_extreme_diagonal                                      |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_invalid_rank                                                  |      1 |       0 |        1 |
| tests/test_tril.py                                      | test_tril_inplace_invalid_rank                                          |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py                  | test__saturate_weight_to_fp16                                           |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py                  | test__saturate_weight_to_fp16_extreme_values                            |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py                  | test__saturate_weight_to_fp16_boundary                                  |      1 |       0 |        1 |
| tests/test__saturate_weight_to_fp16.py                  | test__saturate_weight_to_fp16_nan                                       |      1 |       0 |        1 |
| tests/test_feature_dropout.py                           | test_feature_dropout                                                    |      1 |       0 |        1 |
| tests/test_feature_dropout.py                           | test_feature_dropout_no_train                                           |      1 |       0 |        1 |
| tests/test_feature_dropout.py                           | test_feature_dropout_p_zero                                             |      1 |       0 |        1 |
| tests/test_feature_dropout.py                           | test_feature_dropout_p_one                                              |      1 |       0 |        1 |
| tests/test_feature_dropout.py                           | test_feature_dropout_inplace                                            |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_                                                            |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_float_float_                                                |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_float_tensor                                                |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_tensor_float                                                |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_honors_generator                                            |      1 |       0 |        1 |
| tests/test_normal.py                                    | test_normal_tensor_tensor                                               |      1 |       0 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_fp8_basic                                                 |      0 |       1 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_fp8_custom_padding                                        |      0 |       1 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_fp8_default_inf_padding                                   |      0 |       1 |        1 |
| tests/test_pack_seq.py                                  | test_pack_seq_fp8_block_sizes                                           |      0 |       1 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py         | test_stage_deepseek_v4_mega_moe_inputs_accuracy                         |      0 |       1 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py         | test_stage_deepseek_v4_mega_moe_inputs_rejects_bad_hidden_size          |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                             | test_scaled_mm_v2                                                       |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                             | test_scaled_mm_v2_bias                                                  |      0 |       1 |        1 |
| tests/test__scaled_mm_v2.py                             | test_scaled_mm_v2_vs_torch                                              |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_fp64_rejected                          |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_no_native_fp64_k32                     |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_ascend_slow_decay_spectrum             |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_ascend_long_rectangular                |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_ascend_shape_boundaries                |      0 |       1 |        1 |
| tests/test_linalg_matrix_rank.py                        | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper_large   |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_gate_engages                                         |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_gate_defers                                          |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_gate_limits_follow_element_width                     |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax                                                      |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_out                                                  |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_half_to_float                                        |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_neg_inf                                              |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_at_row_cap                                           |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                             | test_rdna4_softmax_non_contiguous                                       |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py            | test_failure_isolates_platform_dtype_and_recovery                       |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py            | test_identity_discovery_failure_recovers_without_stale_state            |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py            | test_direct_tuner_retry_bypasses_best_config_cache                      |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py            | test_dispatch_switches_only_for_current_model                           |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_cuda_runtime_owns_visibility_and_tensor_operations                 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_hip_runtime_uses_rocm_visibility_names_without_changing_torch_api  |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_maca_runtime_uses_metax_visibility_without_changing_torch_api      |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_musa_runtime_uses_mthreads_visibility_and_torch_api                |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_runtime_rejects_missing_or_empty_device_api                        |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_runtime_metadata_keeps_architecture_separate_from_platform         |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                   | test_runtime_resolves_benchmark_only_through_triton_boundary            |      0 |       1 |        1 |
| tests/test_get_scheduler_metadata.py                    | test_scheduler_metadata_correctness                                     |      0 |       1 |        1 |
| TOTAL                                                   |                                                                         |    317 |      36 |      353 |
