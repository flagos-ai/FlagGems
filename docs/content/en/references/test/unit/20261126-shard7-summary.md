|                                filepath                                |                              function                              | passed | skipped | SUBTOTAL |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------ | -----: | ------: | -------: |
| tests/test_zeros_like.py                                               | test_zeros_like                                                    |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10                                                         |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_                                                        |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_out                                                     |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_special_values                                          |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_empty                                                   |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_int_promotes_to_float                                   |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_backward                                                |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_backward_expanded_grad                                  |      1 |       0 |        1 |
| tests/test_log10.py                                                    | test_log10_backward_special_values                                 |      1 |       0 |        1 |
| tests/test_rand.py                                                     | test_rand                                                          |      1 |       0 |        1 |
| tests/test_special_ndtr.py                                             | test_special_ndtr                                                  |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler                                            |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_non_contiguous                             |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_nan_inf_grid                               |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_special_values                             |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_invalid_grid                               |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_out                                        |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_out_resize                                 |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_out_strided                                |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                       | test_cudnn_grid_sampler_out_dtype_mismatch                         |      1 |       0 |        1 |
| tests/test_pad.py                                                      | test_pad                                                           |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_accuracy_nested_tensor_strides                                |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_stored_noncontiguous                    |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_stored_transposed                       |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_empty_nested                            |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_scalar_nesting                          |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_cpu_nested                              |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_non_nested_error                        |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_dispatch_stability                      |      1 |       0 |        1 |
| tests/test_nested_tensor_strides.py                                    | test_nested_tensor_strides_registered_on_autograd_key              |      1 |       0 |        1 |
| tests/test_nll_loss_forward.py                                         | test_nll_loss_forward                                              |      1 |       0 |        1 |
| tests/test_t_copy.py                                                   | test_accuracy_t_copy                                               |      1 |       0 |        1 |
| tests/test_t_copy.py                                                   | test_accuracy_t_copy_out                                           |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                                            | test_fft_ifftshift                                                 |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                                            | test_fft_ifftshift_int_dim                                         |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                                            | test_fft_ifftshift_empty                                           |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                                            | test_fft_ifftshift_scalar                                          |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy                                          |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_non_contiguous                           |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_empty                                    |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_empty_non_contiguous                     |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_zero_dim                                 |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_single_element                           |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_autograd_contract                        |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_dispatch_stability                       |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_neg_view                                 |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_neg_view_complex                         |      1 |       0 |        1 |
| tests/test_detach_copy.py                                              | test_accuracy_detach_copy_complex_layouts                          |      1 |       0 |        1 |
| tests/test_zeros.py                                                    | test_zeros                                                         |      1 |       0 |        1 |
| tests/test_clamp_tensor.py                                             | test_clamp_tensor                                                  |      1 |       0 |        1 |
| tests/test_clamp_tensor.py                                             | test_clamp_tensor_                                                 |      1 |       0 |        1 |
| tests/test_trace_backward.py                                           | test_trace_backward                                                |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                                        | test_sparse_bsc_tensor_explicit_size                               |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                                        | test_sparse_bsc_tensor_infer_size                                  |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                                        | test_sparse_bsc_tensor_blocksize                                   |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                                        | test_sparse_bsc_tensor_rejects_wrong_layout                        |      1 |       0 |        1 |
| tests/test_minimum.py                                                  | test_minimum                                                       |      1 |       0 |        1 |
| tests/test_fused_moving_avg_obs_fq_helper.py                           | test_fused_moving_avg_obs_fq_helper                                |      1 |       0 |        1 |
| tests/test_fused_moving_avg_obs_fq_helper.py                           | test_fused_moving_avg_obs_fq_helper_rejects_non_fp32               |      1 |       0 |        1 |
| tests/test_atleast_3d.py                                               | test_atleast_3d                                                    |      1 |       0 |        1 |
| tests/test_atleast_3d.py                                               | test_atleast_3d_nan_inf                                            |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations                                                  |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_int_bool                                         |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_row_order                                        |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_non_contiguous                                   |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_nan_inf                                          |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_default_arguments                                |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_rejects_negative_r                               |      1 |       0 |        1 |
| tests/test_combinations.py                                             | test_combinations_rejects_non_1d                                   |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact3d_backward.py                        | test_upsample_nearest_exact3d_backward                             |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact3d_backward.py                        | test_upsample_nearest_exact3d_backward_uint8                       |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact3d_backward.py                        | test_upsample_nearest_exact3d_backward_noncontiguous               |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact3d_backward.py                        | test_upsample_nearest_exact3d_backward_grad_input                  |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i1.py                               | test_special_modified_bessel_i1                                    |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i1.py                               | test_special_modified_bessel_i1_out                                |      1 |       0 |        1 |
| tests/test_not_equal.py                                                | test_accuracy_not_equal                                            |      1 |       0 |        1 |
| tests/test_not_equal.py                                                | test_accuracy_not_equal_scalar                                     |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_2d                                           |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_2d_dilation                                  |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_1d                                           |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_3d                                           |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_invalid_padding                              |      1 |       0 |        1 |
| tests/test_convolution_mode.py                                         | test_convolution_mode_same_strided_raises                          |      1 |       0 |        1 |
| tests/test_asin.py                                                     | test_asin                                                          |      1 |       0 |        1 |
| tests/test_asin.py                                                     | test_asin_                                                         |      1 |       0 |        1 |
| tests/test_embedding_bag_per_sample_weights_backward.py                | test_embedding_bag_per_sample_weights_backward                     |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__tensor                                                    |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__scalar                                                    |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__backward                                                  |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__backward_other                                            |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__backward_view                                             |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__scalar_backward                                           |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__backward_rejects_leaf                                     |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__scalar_backward_rejects_leaf                              |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__special_values                                            |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__aliased_operand                                           |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__view_receiver                                             |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__invalid_operand                                           |      1 |       0 |        1 |
| tests/test_ge_.py                                                      | test_ge__unsupported_dtype                                         |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner                                                         |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_1d_dot                                                  |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_scalar                                                  |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_matvec                                                  |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_empty_contraction                                       |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_2d                                                      |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_dtype_mismatch                                          |      1 |       0 |        1 |
| tests/test_inner.py                                                    | test_inner_contraction_mismatch                                    |      1 |       0 |        1 |
| tests/test_log_sigmoid.py                                              | test_log_sigmoid                                                   |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange                                                        |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange_start_dtype_inference                                  |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange_start_float_args_follow_default_dtype                  |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_fast_path                                 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_fallback_cases                            |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_int_fallback                              |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_writes_through_view                       |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_registration_is_live                      |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_abs_inplace_rejects_empty_list                        |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_fast_path                               |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_half_to_even                            |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_large_magnitude                         |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_fallback_cases                          |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_writes_through_view                     |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_registration_is_live                    |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_round_inplace_rejects_empty_list                      |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_fast_path                                |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_fallback_cases                           |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_int_fallback                             |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_writes_through_view                      |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_registration_is_live                     |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                                   | test_foreach_sign_inplace_rejects_empty_list                       |      1 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum                                                        |      1 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_dim                                                    |      1 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_dim_out                                                |      1 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_out                                                    |      1 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_edge                                                   |      1 |       0 |        1 |
| tests/test_lcm.py                                                      | test_lcm                                                           |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter                                              |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter_offset                                       |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter_negative_offset                              |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter_non_last_dims                                |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter_large_offset                                 |      1 |       0 |        1 |
| tests/test_diagonal_scatter.py                                         | test_diagonal_scatter_non_square                                   |      1 |       0 |        1 |
| tests/test_baddbmm.py                                                  | test_baddbmm                                                       |      1 |       0 |        1 |
| tests/test_baddbmm.py                                                  | test_baddbmm_out                                                   |      1 |       0 |        1 |
| tests/test_baddbmm.py                                                  | test_baddbmm_backward                                              |      1 |       0 |        1 |
| tests/test_baddbmm.py                                                  | test_baddbmm_backward_broadcast                                    |      1 |       0 |        1 |
| tests/test_less_.py                                                    | test_less_                                                         |      1 |       0 |        1 |
| tests/test_less_.py                                                    | test_less_scalar_                                                  |      1 |       0 |        1 |
| tests/test_sigmoid.py                                                  | test_sigmoid                                                       |      1 |       0 |        1 |
| tests/test_sigmoid.py                                                  | test_sigmoid_                                                      |      1 |       0 |        1 |
| tests/test_sigmoid.py                                                  | test_sigmoid_backward                                              |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized                                        |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_memory_format                          |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_device_default_cpu                     |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_out_matches_reference                  |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_requires_quantized_dtype               |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_rejects_string_memory_format           |      1 |       0 |        1 |
| tests/test_empty_affine_quantized.py                                   | test_empty_affine_quantized_registered_in_full_config              |      1 |       0 |        1 |
| tests/test_unfold_copy.py                                              | test_unfold_copy                                                   |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_reference                          |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu                                    |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_alpha_types                        |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_bias_shape                         |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_special_values                     |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_empty                              |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_invalid_ndim                       |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                               | test_cudnn_convolution_add_relu_out                                |      1 |       0 |        1 |
| tests/test_new_full.py                                                 | test_new_full                                                      |      1 |       0 |        1 |
| tests/test_arctan_.py                                                  | test_arctan                                                        |      1 |       0 |        1 |
| tests/test_arctan_.py                                                  | test_arctan_                                                       |      1 |       0 |        1 |
| tests/test_native_norm.py                                              | test_native_norm                                                   |      1 |       0 |        1 |
| tests/test_margin_ranking_loss.py                                      | test_margin_ranking_loss                                           |      1 |       0 |        1 |
| tests/test_margin_ranking_loss.py                                      | test_margin_ranking_loss_backward                                  |      1 |       0 |        1 |
| tests/test_replication_pad1d.py                                        | test_replication_pad1d                                             |      1 |       0 |        1 |
| tests/test_replication_pad1d.py                                        | test_replication_pad1d_out                                         |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_accuracy_fill_diagonal_                                       |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_accuracy_fill_diagonal__wrap                                  |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_accuracy_fill_diagonal__noncontiguous                         |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_fill_diagonal__wrap_rejects_out_of_bounds_strided_view        |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_accuracy_fill_diagonal__non_float                             |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_fill_diagonal__rejects_tensors_with_fewer_than_two_dims       |      1 |       0 |        1 |
| tests/test_fill_diagonal_.py                                           | test_fill_diagonal__rejects_unequal_higher_dimensions              |      1 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward                                                |      1 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward_oob_end                                        |      1 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward_oob_start                                      |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__identity                                       |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__sparse_dims_grow                               |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__hybrid_sparse_dims_grow                        |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__dense_dims_grow                                |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__dense_dims_grow_complex                        |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__empty_reinterpret                              |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__non_empty_rejects                              |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                                           | test_sparse_resize__rejects_non_coo                                |      1 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                                        | test_bitwise_or_tensor                                             |      1 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                                        | test_bitwise_or_tensor_                                            |      1 |       0 |        1 |
| tests/test_take_along_dim.py                                           | test_take_along_dim                                                |      1 |       0 |        1 |
| tests/test_take_along_dim.py                                           | test_take_along_dim_dtypes                                         |      1 |       0 |        1 |
| tests/test_take_along_dim.py                                           | test_take_along_dim_none                                           |      1 |       0 |        1 |
| tests/test_take_along_dim.py                                           | test_take_along_dim_noncontiguous                                  |      1 |       0 |        1 |
| tests/test_take_along_dim.py                                           | test_take_along_dim_out                                            |      1 |       0 |        1 |
| tests/test_dirichlet_grad.py                                           | test_dirichlet_grad                                                |      1 |       0 |        1 |
| tests/test_special_i1e.py                                              | test_special_i1e                                                   |      1 |       0 |        1 |
| tests/test_special_i1e.py                                              | test_special_i1e_out                                               |      1 |       0 |        1 |
| tests/test_sinh.py                                                     | test_sinh                                                          |      1 |       0 |        1 |
| tests/test_sinh.py                                                     | test_sinh_large_values                                             |      1 |       0 |        1 |
| tests/test_sinh.py                                                     | test_sinh_                                                         |      1 |       0 |        1 |
| tests/test_one_hot.py                                                  | test_one_hot                                                       |      1 |       0 |        1 |
| tests/test_one_hot.py                                                  | test_one_hot_inference_mode                                        |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_moe_load_balance_loss_matches_reference                       |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_tle_path_with_irregular_large_shape                           |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_noncontiguous_logits_and_mask                                 |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_all_masked_tokens_return_zero                                 |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_default_top_k_is_two                                          |      1 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                                    | test_moe_load_balance_loss_validation                              |      1 |       0 |        1 |
| tests/test_linalg_ldl_factor.py                                        | test_linalg_ldl_factor                                             |      1 |       0 |        1 |
| tests/test_adaptive_max_pool2d_backward.py                             | test_adaptive_max_pool2d_backward                                  |      1 |       0 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_fused_marlin_moe_w8a16_int8                                   |      1 |       0 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_rejects_unsupported_quant_type                                |      1 |       0 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_rejects_act_order                                             |      1 |       0 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_rejects_fp8_input_dtype                                       |      1 |       0 |        1 |
| tests/test_channel_shuffle.py                                          | test_channel_shuffle                                               |      1 |       0 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py           | test_combine_topk_swa_indices_accuracy                             |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_aa.py                                   | test_upsample_bilinear2d_aa                                        |      1 |       0 |        1 |
| tests/test_special_log_softmax.py                                      | test_special_log_softmax                                           |      1 |       0 |        1 |
| tests/test_special_log_softmax.py                                      | test_special_log_softmax_large_n                                   |      1 |       0 |        1 |
| tests/test_reflection_pad1d_backward.py                                | test_reflection_pad1d_backward                                     |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_tensor                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_tensor_                                            |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_float                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_int                                           |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_float_                                        |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_int_                                          |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_float                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_int                                           |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_float_                                        |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_int_                                          |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_scalar                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_scalar_                                            |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_                                                   |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_tensor                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_scalar                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_complex                                           |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_float_tensor                                      |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_int                                                |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_int_scalar                                        |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_tensor_tensor                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_tensor_scalar                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_scalar_tensor                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_tensor                                               |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_scalar                                               |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_tensor_                                              |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_scalar_                                              |      1 |       0 |        1 |
| tests/test_contiguous.py                                               | test_accuracy_contiguous                                           |      1 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py                | test_amp_foreach_non_finite_check_and_unscale_                     |      1 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py                | test_amp_foreach_non_finite_check_and_unscale__inf                 |      1 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py                | test_amp_foreach_non_finite_check_and_unscale__nan                 |      1 |       0 |        1 |
| tests/test_flatten.py                                                  | test_flatten                                                       |      1 |       0 |        1 |
| tests/test_lift.py                                                     | test_lift                                                          |      1 |       0 |        1 |
| tests/test_lift.py                                                     | test_lift_out                                                      |      1 |       0 |        1 |
| tests/test_lift.py                                                     | test_lift_low_precision                                            |      1 |       0 |        1 |
| tests/test_lift.py                                                     | test_lift_out_low_precision                                        |      1 |       0 |        1 |
| tests/test_unsafe_masked_index_put_accumulate.py                       | test_unsafe_masked_index_put_accumulate                            |      1 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py                       | test_scaled_dot_product_cudnn_attention                            |      1 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py                       | test_scaled_dot_product_cudnn_attention_with_attn_bias             |      1 |       0 |        1 |
| tests/test_arccos.py                                                   | test_arccos                                                        |      1 |       0 |        1 |
| tests/test_arccos.py                                                   | test_arccos_                                                       |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode                                                          |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_repeated_values                                          |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_byte_boundaries                                          |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_run_boundaries                                           |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_adjacent_values                                          |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_many_rows                                                |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_histogram_launch_boundary                                |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_byte_histogram_launch_boundary                           |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_small_rows_launch_boundary                               |      1 |       0 |        1 |
| tests/test_mode.py                                                     | test_mode_large_radix_launch_boundary                              |      1 |       0 |        1 |
| tests/test_logical_and.py                                              | test_logical_and                                                   |      1 |       0 |        1 |
| tests/test_logical_and.py                                              | test_logical_and_                                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_random                                      |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_random_batch                                |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_large_norm                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_tiny_norm                                   |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_nan                                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_batch_consistency                           |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_diagonal                                    |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_zero                                        |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_one_by_one                                  |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_empty                                       |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_non_contiguous                              |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_errors                                      |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_out                                         |      1 |       0 |        1 |
| tests/test_linalg_matrix_exp.py                                        | test_linalg_matrix_exp_out_errors                                  |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values                                             |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values_keepdim_shapes                              |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values_ties                                        |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values_strided_nonlast_reduction                   |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values_out_wrong_device                            |      1 |       0 |        1 |
| tests/test_median_dim_values.py                                        | test_median_dim_values_out_rejects_bad_index_dtype                 |      1 |       0 |        1 |
| tests/test_bmm.py                                                      | test_bmm                                                           |      1 |       0 |        1 |
| tests/test_bmm.py                                                      | test_bmm_out                                                       |      1 |       0 |        1 |
| tests/test_bmm.py                                                      | test_bmm_empty_dimensions                                          |      1 |       0 |        1 |
| tests/test_bmm.py                                                      | test_bmm_out_empty_dimensions                                      |      1 |       0 |        1 |
| tests/test_bmm.py                                                      | test_bmm_empty_dimensions_reject_mismatch                          |      1 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable_backward.py | test_scaled_dot_product_fused_attention_overrideable_backward      |      1 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable_backward.py | test_scaled_dot_product_fused_attention_overrideable_backward_mask |      1 |       0 |        1 |
| tests/test_ior_tensor.py                                               | test_ior_tensor                                                    |      1 |       0 |        1 |
| tests/test_w8a8_block_fp8_matmul.py                                    | test_w8a8_block_fp8_matmul                                         |      1 |       0 |        1 |
| tests/test_nested_select_backward.py                                   | test_nested_select_backward                                        |      1 |       0 |        1 |
| tests/test_nested_select_backward.py                                   | test_nested_select_backward_1d_components                          |      1 |       0 |        1 |
| tests/test_avg_pool2d.py                                               | test_avg_pool2d                                                    |      1 |       0 |        1 |
| tests/test_avg_pool2d.py                                               | test_avg_pool2d_backward                                           |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize                                          |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_int                                      |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_out                                      |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_identity_and_metadata                    |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_same_tensor_noop                         |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_cast                                     |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_strides                                  |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_out_alias                                |      1 |       0 |        1 |
| tests/test_copy_from_and_resize.py                                     | test_copy_from_and_resize_out_options                              |      1 |       0 |        1 |
| tests/test_rwkv_ka_fusion.py                                           | test_accuracy_rwkv_kafusion                                        |      1 |       0 |        1 |
| tests/test_conv_depthwise3d.py                                         | test_conv_depthwise3d                                              |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp                                                 |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out                                             |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_large_dynamic_range                             |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_mixed_infinities                                |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_nan_propagation                                 |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_infinite_inputs                                 |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_wide_dynamic_range_rows                         |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_fp64_precision                                  |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_complex                                         |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_complex_special_values                          |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_scalar                                          |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_scalar_invalid_dim                              |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_invalid_dim                                     |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_empty                                           |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_empty                                       |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_resizes                                     |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_noncontiguous                               |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_rejects_mismatched_dtype                    |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_rejects_mismatched_device                   |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_out_invalid_dim_does_not_touch_out              |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_noncontiguous_input                             |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_all_dims                                        |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_long_rows                                       |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_long_rows_special_values                        |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_long_row_fp64                                   |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_long_complex_rows                               |      1 |       0 |        1 |
| tests/test__logcumsumexp.py                                            | test__logcumsumexp_long_complex_rows_special_values                |      1 |       0 |        1 |
| tests/test_infinitely_differentiable_gelu_backward.py                  | test_infinitely_differentiable_gelu_backward                       |      1 |       0 |        1 |
| tests/test_sspaddmm.py                                                 | test_sspaddmm                                                      |      1 |       0 |        1 |
| tests/test_sspaddmm.py                                                 | test_sspaddmm_out                                                  |      1 |       0 |        1 |
| tests/test_scaled_softmax.py                                           | test_scaled_softmax_forward                                        |      1 |       0 |        1 |
| tests/test_scaled_softmax.py                                           | test_scaled_softmax_backward                                       |      1 |       0 |        1 |
| tests/test_nested_tensor_from_mask_left_aligned.py                     | test_nested_tensor_from_mask_left_aligned                          |      1 |       0 |        1 |
| tests/test_nested_tensor_from_mask_left_aligned.py                     | test_nested_tensor_from_mask_left_aligned_edge                     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward                                          |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_shapes                                   |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_dim_forms                                |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_dim_forms_large                          |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_zero_nnz_and_zero_axis                   |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_noncontiguous                            |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_stored_offsets                           |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_out                                      |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_uncoalesced_self                         |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_duplicate_coordinates                    |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_special_values                           |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_copy_special_values                      |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_regather_rejects_non_float_dtypes        |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_negative_dense_self                      |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_negative_out_of_range_dim                |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_negative_aliased_dim                     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test__sparse_sum_backward_negative_sparse_grad_for_summed_dims     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                     | test_sparse_sum_backward_dispatch                                  |      1 |       0 |        1 |
| tests/test_le.py                                                       | test_le                                                            |      1 |       0 |        1 |
| tests/test_le.py                                                       | test_le_scalar                                                     |      1 |       0 |        1 |
| tests/test_hypot.py                                                    | test_hypot                                                         |      1 |       0 |        1 |
| tests/test_hypot.py                                                    | test_hypot_out                                                     |      1 |       0 |        1 |
| tests/test_fused_add_rms_norm.py                                       | test_fused_add_rms_norm                                            |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py               | test_fake_quantize_learnable_per_channel_affine                    |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py               | test_fake_quantize_learnable_per_channel_affine_grad_factor        |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t                                |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t_out_of_domain                  |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t_degrees                        |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t_tensor_degree                  |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t_out                            |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_t.py                           | test_special_chebyshev_polynomial_t_rejects_unsupported_dtype      |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod                                               |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_ignores_unstored_zeros                        |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_stored_zero_wins                              |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_empty_input                                   |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_dtype                                         |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_unsorted_columns                              |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_empty_rows_and_columns                        |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_out                                           |      1 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                          | test_sparse_csr_prod_rejects_bad_request                           |      1 |       0 |        1 |
| tests/test__sobol_engine_scramble_.py                                  | test_sobol_engine_scramble_                                        |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_accuracy_orgqr                                                |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_k_cases                                                 |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_invalid_shapes                                          |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_complex_is_explicitly_unsupported                       |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_accuracy_orgqr_out                                            |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_out_multiple_noncontiguous_batch_dimensions             |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_out_casts_only_final_result                             |      1 |       0 |        1 |
| tests/test_orgqr.py                                                    | test_orgqr_out_stride_alias_dtype_and_device_contract              |      1 |       0 |        1 |
| tests/test_zeros_like.py                                               | test_zeros_like_launch_core_dim_limit                              |      0 |       1 |        1 |
| tests/test_zeros_like.py                                               | test_zeros_like_factory_capped_grid_exact                          |      0 |       1 |        1 |
| tests/test_zeros_like.py                                               | test_zeros_like_ascend_review_options                              |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_inplace_boundaries_exact                                 |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_overwrites_special_values_exact                          |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_noncontiguous_storage_exact                              |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_repeated_calls_exact                                     |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zeros_factory_poisoned_allocation_exact                       |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_launch_core_dim_limit                                    |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_real_large_input_exact                                   |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_capped_grid_covers_all_tiles_exact                       |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zeros_factory_capped_grid_exact                               |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                       | test_rms_norm_w8a16_fp8_weight_updates                             |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                       | test_rms_norm_w8a16_fp8_encodings                                  |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                       | test_rms_norm_w8a16_fp8                                            |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                       | test_rms_norm_w8a16_fp8_mthreads_shapes                            |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8                          |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_deepgemm                 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_int8_extremes            |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_pipeline                 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_changing_expert_rows     |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_async                    |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_gemv_cache_tail          |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8                        |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_int8_extremes          |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_changing_expert_rows   |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_dtypes_tails           |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_fp8_encodings          |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_int8_cancellation      |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                                        | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_sparse_changing_groups |      0 |       1 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_fused_marlin_moe_w4a16_int4                                   |      0 |       1 |        1 |
| tests/test_fused_marlin_moe.py                                         | test_fused_marlin_moe_w4a16_mxfp4                                  |      0 |       1 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py           | test_combine_topk_swa_indices_vllm_accuracy                        |      0 |       1 |        1 |
| tests/test_mode.py                                                     | test_mode_histogram16_batch_boundary                               |      0 |       1 |        1 |
| tests/test_mode.py                                                     | test_mode_ascend_dispatch                                          |      0 |       1 |        1 |
| tests/test_bmm.py                                                      | test_bmm_w8a8_fp8                                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mm_w8a8_int8                                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mm_w8a8_int8_strides                                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mm_w8a8_int8_updates                                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mm_w8a8_int8_empty                                            |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mm_w8a8_int8_reject_float                                     |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_scaled_bias                                                   |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_scaled_empty_bias                                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_scaled_graph_bias_updates                                     |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_scaled_validation                                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_tensor_scales_without_bias                                    |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_scaled_unaligned_weight                                       |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_prequantized                                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_long_k_overflow                                               |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_mixed_scales                                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_graph_updates                                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                                             | test_invalid                                                       |      0 |       1 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py               | test_fake_quantize_learnable_per_channel_affine_transposed         |      0 |       1 |        1 |
| TOTAL                                                                  |                                                                    |    408 |      53 |      461 |
|                       filepath                       |                                  function                                   | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_group_norm.py                             | test_native_group_norm                                                      |      1 |      0 |       0 |        1 |
| tests/test_group_norm.py                             | test_group_norm                                                             |      1 |      0 |       0 |        1 |
| tests/test_group_norm.py                             | test_group_norm_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_trace.py                                  | test_trace                                                                  |      1 |      0 |       0 |        1 |
| tests/test_exp.py                                    | test_exp                                                                    |      1 |      0 |       0 |        1 |
| tests/test_exp.py                                    | test_exp_                                                                   |      1 |      0 |       0 |        1 |
| tests/test_exp.py                                    | test_exp_out                                                                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_nested_metadata_matches_even_when_shape_key_is_a_descriptor            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_tensor_identity_mismatch_does_not_use_first_shape_shortcut             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_scalar_type_and_value_mismatches_fall_back                             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_ambiguity_does_not_select_first_record                                 |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_named_inputs_require_binding_and_compare_scalars                       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_legacy_shape_and_microsecond_units_and_private_operator_names          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_invalid_latency_is_a_miss                                              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_invalid_file_warns_and_disables_comparison                             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_missing_file_warns_and_valid_metadata_is_preserved                     |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_scaling_selects_chip_and_resource_and_prefers_measured                 |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_invalid_scaling_falls_back_to_raw_ratio                                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_external_ratio_preserves_native_measurements_and_records_provenance    |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_skip_native_can_use_external_data_and_a_miss_keeps_na                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_real_summary_prefers_external_per_case_and_preserves_fallback          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_malformed_metadata_does_not_fall_through_to_legacy_shape               |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_malformed_resource_and_oversized_latency_do_not_break_benchmarks       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_factory_device_matches_selected_backend_without_weakening_other_inputs |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_thead_runtime_vendor_uses_ops_baseline_t_head_scaling                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_h200_does_not_use_h20_scaling                                          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_device_normalization_requires_the_selected_target                      |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_factory_normalization_still_checks_all_other_parameters                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_named_device_metadata_uses_the_same_target_rule                        |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_overlapping_models_select_the_exact_chip_and_preserve_suffixes         |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_vendor_alias_is_explicit_and_does_not_match_other_vendors              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_malformed_kwargs_metadata_falls_back_without_crashing                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_legacy_shape_cannot_establish_multi_input_or_parameter_abi             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_legacy_sglang_packed_silu_and_mul_does_not_match_two_tensor_gems_abi   |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_incomplete_named_metadata_cannot_choose_a_semantic_baseline            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                 | test_complete_named_metadata_still_matches_and_checks_all_parameters        |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update_functional.py      | test_batch_norm_with_update_functional                                      |      1 |      0 |       0 |        1 |
| tests/test_matrix_exp_backward.py                    | test_matrix_exp_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_resize_output.py                          | test_resize_output                                                          |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_radix_final_config_prefers_large_vocab           |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_full_vocab                                       |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_large_vocab_partial_nonzero_range                |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_variable_lengths                                 |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_nonzero_starts                                   |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_topk_greater_than_row_len                                              |      1 |      0 |       0 |        1 |
| tests/test_logit.py                                  | test_logit                                                                  |      1 |      0 |       0 |        1 |
| tests/test_logit.py                                  | test_logit_                                                                 |      1 |      0 |       0 |        1 |
| tests/test_logit.py                                  | test_logit_out                                                              |      1 |      0 |       0 |        1 |
| tests/test_logit.py                                  | test_logit_out_non_contiguous_out                                           |      1 |      0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                   | test_slow_conv2d_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                   | test_slow_conv2d_backward_dispatches_to_gems                                |      1 |      0 |       0 |        1 |
| tests/test_rsqrt.py                                  | test_rsqrt                                                                  |      1 |      0 |       0 |        1 |
| tests/test_rsqrt.py                                  | test_rsqrt_                                                                 |      1 |      0 |       0 |        1 |
| tests/test_grouped_mm.py                             | test_grouped_mm                                                             |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_1d                                                                 |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_correction                                                         |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_fweights                                                           |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_aweights                                                           |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_fweights_aweights                                                  |      1 |      0 |       0 |        1 |
| tests/test_cov.py                                    | test_cov_non_contiguous                                                     |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack                                                                 |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_int_bool                                                        |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_zero_sized                                                      |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_element_order                                                   |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_non_contiguous                                                  |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_nan_inf                                                         |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_mixed_dtype_promotion                                           |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_mixed_real_complex                                              |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_rejects_mismatched_shapes                                       |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_rejects_empty_list                                              |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_rejects_non_tensor                                              |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                                 | test_dstack_backward                                                        |      1 |      0 |       0 |        1 |
| tests/test_masked_softmax_backward.py                | test_masked_softmax_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_multiply.py                               | test_multiply                                                               |      1 |      0 |       0 |        1 |
| tests/test_multiply.py                               | test_multiply_scalar                                                        |      1 |      0 |       0 |        1 |
| tests/test_native_batch_norm_legit_functional.py     | test_native_batch_norm_legit_functional                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d                                                   |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_channels_last                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_4d_output_is_channels_last                        |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_matches_native_cuda                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_3d_output_stays_contiguous                        |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_zero_batch                                        |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_extreme_values                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_padding_sentinel                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_default_stride                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_runs_triton_kernel                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out                                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_resizes                                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_preserves_offset_and_strides                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_channels_last_out                             |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_aliasing_input                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_mismatched_qparams                            |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_rejects_per_channel_out                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_out_rejects_mismatched_out                        |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_invalid_params                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_invalid_shapes                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                   | test_quantized_max_pool2d_rejects_per_channel_input                         |      1 |      0 |       0 |        1 |
| tests/test_scalar_tensor.py                          | test_scalar_tensor                                                          |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi                                                            |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_layout                                                     |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_boundary_values                                            |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_special_values                                             |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_out_promoted                                               |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_rejects_non_tensor_input                                   |      1 |      0 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_out_rejects_incompatible_dtype                             |      1 |      0 |       0 |        1 |
| tests/test_std.py                                    | test_std                                                                    |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py      | test_upsample_nearest_exact2d_backward                                      |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py      | test_upsample_nearest_exact2d_backward_with_scales                          |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest2d.py                     | test_upsample_nearest2d                                                     |      1 |      0 |       0 |        1 |
| tests/test_gcd_.py                                   | test_gcd_                                                                   |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_accuracy_addbmm                                                        |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_addbmm_backward                                                        |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_addbmm_backward_registered_dispatch                                    |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_addbmm_backward_single_input                                           |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_addbmm_backward_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                 | test_addbmm_backward_fused_kernel_count                                     |      1 |      0 |       0 |        1 |
| tests/test_special_hermite_polynomial_he.py          | test_special_hermite_polynomial_he                                          |      1 |      0 |       0 |        1 |
| tests/test_special_hermite_polynomial_he.py          | test_special_hermite_polynomial_he_scalar_x                                 |      1 |      0 |       0 |        1 |
| tests/test_special_hermite_polynomial_he.py          | test_special_hermite_polynomial_he_scalar_scalar                            |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_t.py | test_special_shifted_chebyshev_polynomial_t                                 |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_t.py | test_special_shifted_chebyshev_polynomial_t_scalar_n                        |      1 |      0 |       0 |        1 |
| tests/test_uniform_.py                               | test_uniform_                                                               |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_rejects_mismatched_metadata                   |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_rejects_mismatched_device                     |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_validates_normalized_shape                    |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_backward_residual_only                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_forward.py                | test_cudnn_attention_forward                                                |      1 |      0 |       0 |        1 |
| tests/test__linalg_det.py                            | test_accuracy_linalg_det                                                    |      1 |      0 |       0 |        1 |
| tests/test_get_paged_mqa_logits_metadata.py          | test_get_paged_mqa_logits_metadata                                          |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy                                             |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_dtype_mapping                               |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_values                                      |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_noncontiguous                               |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_transposed                                  |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_high_rank_noncontiguous                     |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_expanded                                    |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_storage_offset                              |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_1d_strided                                  |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_conj                                        |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_empty                                       |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_zero_dim                                    |      1 |      0 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_error_paths                                 |      1 |      0 |       0 |        1 |
| tests/test_negative.py                               | test_negative                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_empty                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_noncontiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_noncontiguous_multidim_lead                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_recovers_known_solution                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_dims                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_dims_defaults_to_none                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_dims_out_of_range                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_dims_duplicate                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_shape_mismatch                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_residual_is_small                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_singular_does_not_hang_or_crash                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_rejects_complex                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_autograd_input_is_detached                          |      1 |      0 |       0 |        1 |
| tests/flagtune/test_model_source.py                  | ModelSourceTests.test_defaults_and_once_only                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_model_source.py                  | ModelSourceTests.test_explicit_sources_including_empty_are_preserved        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_model_source.py                  | ModelSourceTests.test_manifest_schema_and_wheel_data                        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_model_source.py                  | ModelSourceTests.test_version_preferences_preserved                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_2d                                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_2d_keepdim                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_batch                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_batch_nondefault_dim                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_dtype_param                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_large                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_1d_rejected                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_same_dim_rejected                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_unsupported_ord_rejected                                               |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp                                                                   |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp_                                                                  |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp_scalar                                                            |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp_scalar_                                                           |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp_tensor_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_lerp.py                                   | test_lerp_scalar_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_acc_false                                                    |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__acc_false                                                   |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_error_all_none                                               |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__error_all_none                                              |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_mixed_none_and_tensor                                        |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__large_strided_offset                                        |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__single_element_index                                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_operator_markers_are_declared                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_ops                                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_ops_registration_key_exists                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_paired_noncontiguous                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_ops_length_mismatch_rejected                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_reduction_orders                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_zero_writes_through_views                             |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_ops_launch_count_is_independent_of_list_length                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_int_promotion                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_inplace_rejects_promotion                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_ops_dtype_sets_match_aten                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_bool                                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_add_alpha                                             |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_alpha_keeps_one_launch                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_copy_mixed_dtype                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_max_large_values                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_norm_orders                                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_norm_all_zero                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_accuracy_foreach_pow_bool_base                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_pow_list_rejects_bool                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_clamp_scalar_rejects_bool                                      |      1 |      0 |       0 |        1 |
| tests/test_cosh_backward.py                          | test_cosh_backward                                                          |      1 |      0 |       0 |        1 |
| tests/test_cosh_backward.py                          | test_cosh_backward_expanded_grad                                            |      1 |      0 |       0 |        1 |
| tests/test_cosh_backward.py                          | test_cosh_backward_noncontiguous_input                                      |      1 |      0 |       0 |        1 |
| tests/test_cosh_backward.py                          | test_cosh_backward_empty                                                    |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index                                                           |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_with_none_and_tensor                                      |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_error_bad_index_dtype                                     |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_error_non_broadcastable_indices                           |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_error_empty_indices                                       |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_error_too_many_indices                                    |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_non_contiguous_input                                      |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_non_contiguous_input_with_none                            |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index.py                           | test_unsafe_index_zero_dim_index_negative                                   |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_scalar_tensor.py              | test_bitwise_and_scalar_tensor                                              |      1 |      0 |       0 |        1 |
| tests/test_logspace.py                               | test_logspace                                                               |      1 |      0 |       0 |        1 |
| tests/test_ones_like.py                              | test_ones_like                                                              |      1 |      0 |       0 |        1 |
| tests/test_ones_like.py                              | test_ones_like_empty                                                        |      1 |      0 |       0 |        1 |
| tests/test_sobol_engine_initialize_state_.py         | test_sobol_engine_initialize_state_                                         |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                    | test_triplet_margin_loss                                                    |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                    | test_triplet_margin_loss_p1_eps                                             |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                    | test_triplet_margin_loss_extremal_p                                         |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                    | test_triplet_margin_loss_empty_feature_dim                                  |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                    | test_triplet_margin_loss_broadcast_and_promotion                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_explicit_size                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_aliases_values                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_rejects_dtype_mismatch                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_roundtrip_to_csr                                     |      1 |      0 |       0 |        1 |
| tests/test_addbmm_.py                                | test_accuracy_addbmm_                                                       |      1 |      0 |       0 |        1 |
| tests/test_Concat.py                                 | test_Concat                                                                 |      1 |      0 |       0 |        1 |
| tests/test_flash_mla.py                              | test_flash_mla                                                              |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update_functional.py      | test_batch_norm_with_update_functional_no_running_stats                     |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                  | test_top_k_per_row_prefill_vs_vllm                                          |      0 |      0 |       1 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_edge                                                                   |      0 |      0 |       1 |        1 |
| tests/test_linalg_matrix_norm.py                     | test_out                                                                    |      0 |      0 |       1 |        1 |
| tests/test_index_put.py                              | test_index_put__bf16_strided_boundary                                       |      0 |      0 |       1 |        1 |
| tests/test_fp8_einsum.py                             | test_accuracy_fp8_einsum                                                    |      0 |      0 |       1 |        1 |
| tests/test_cov.py                                    | test_cov                                                                    |      0 |      1 |       0 |        1 |
| tests/test_special_psi.py                            | test_special_psi_rejects_unsupported_dtype                                  |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_forward                                       |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_no_grad_with_grad_inputs                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_optional_affine                               |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_large_normalized_shape                        |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_noncontiguous_fallback                        |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_backward                                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_backward_optional_affine                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py               | test_post_layer_norm_residual_backward_large_normalized_shape               |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                      | test_accuracy_view_as_real_copy_independent_storage                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_addmm.py           | test_sparse_semi_structured_addmm                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_addmm.py           | test_sparse_semi_structured_addmm_with_alpha_beta                           |      0 |      1 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_fp64_preserved                                      |      0 |      1 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                     | test_linalg_tensorsolve_fp64_accuracy_is_double_grade                       |      0 |      1 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_acc_true                                                     |      0 |      1 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__acc_true                                                    |      0 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                            | test_foreach_ops_registration_is_live                                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_infer_size                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                      | test_sparse_csc_tensor_infer_size_trailing_empty_rows                       |      0 |      1 |       0 |        1 |
| tests/test_hspmm.py                                  | test_hspmm_accuracy                                                         |      0 |      1 |       0 |        1 |
| tests/test_hspmm.py                                  | test_hspmm_non_contiguous                                                   |      0 |      1 |       0 |        1 |
| tests/test_hspmm.py                                  | test_hspmm_uncoalesced_input                                                |      0 |      1 |       0 |        1 |
| TOTAL                                                |                                                                             |    236 |     23 |       6 |      265 |
