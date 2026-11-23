|                             filepath                              |                                 function                                 | passed | failed | skipped | SUBTOTAL |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------ | -----: | -----: | ------: | -------: |
| tests/test_cosine_embedding_loss.py                               | test_cosine_embedding_loss                                               |      1 |      0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination                                          |      1 |      0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_higher_rank                              |      1 |      0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_extreme                                  |      1 |      0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_out                                      |      1 |      0 |       0 |        1 |
| tests/test_rand.py                                                | test_rand                                                                |      1 |      0 |       0 |        1 |
| tests/test_special_ndtr.py                                        | test_special_ndtr                                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler                                                  |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_non_contiguous                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_nan_inf_grid                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_special_values                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_invalid_grid                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_out                                              |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_out_resize                                       |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_out_strided                                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                  | test_cudnn_grid_sampler_out_dtype_mismatch                               |      1 |      0 |       0 |        1 |
| tests/test_pad.py                                                 | test_pad                                                                 |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_accuracy_nested_tensor_strides                                      |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_stored_noncontiguous                          |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_stored_transposed                             |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_empty_nested                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_scalar_nesting                                |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_cpu_nested                                    |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_non_nested_error                              |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_dispatch_stability                            |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_strides.py                               | test_nested_tensor_strides_registered_on_autograd_key                    |      1 |      0 |       0 |        1 |
| tests/test_nll_loss_forward.py                                    | test_nll_loss_forward                                                    |      1 |      0 |       0 |        1 |
| tests/test_t_copy.py                                              | test_accuracy_t_copy                                                     |      1 |      0 |       0 |        1 |
| tests/test_t_copy.py                                              | test_accuracy_t_copy_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_fft_ifftshift.py                                       | test_fft_ifftshift                                                       |      1 |      0 |       0 |        1 |
| tests/test_fft_ifftshift.py                                       | test_fft_ifftshift_int_dim                                               |      1 |      0 |       0 |        1 |
| tests/test_fft_ifftshift.py                                       | test_fft_ifftshift_empty                                                 |      1 |      0 |       0 |        1 |
| tests/test_fft_ifftshift.py                                       | test_fft_ifftshift_scalar                                                |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy                                                |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_non_contiguous                                 |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_empty                                          |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_empty_non_contiguous                           |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_zero_dim                                       |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_single_element                                 |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_autograd_contract                              |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_dispatch_stability                             |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_neg_view                                       |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_neg_view_complex                               |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                         | test_accuracy_detach_copy_complex_layouts                                |      1 |      0 |       0 |        1 |
| tests/test_zeros.py                                               | test_zeros                                                               |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                                        | test_clamp_tensor                                                        |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                                        | test_clamp_tensor_                                                       |      1 |      0 |       0 |        1 |
| tests/test_trace_backward.py                                      | test_trace_backward                                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                                 | test_sparse_broadcast_to_identity                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                                 | test_sparse_broadcast_to_prepend                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                                 | test_sparse_broadcast_to_expand_sparse_dim                               |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices_backward.py                    | test_accuracy_max_pool3d_with_indices_backward                           |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices_backward.py                    | test_accuracy_max_pool3d_with_indices_backward_channels_last_3d          |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_forward                                |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_forward_no_indices                     |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_int_output_size                        |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_nan                                    |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_ties                                   |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_all_negative                           |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_mixed_sign                             |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_empty_output                           |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                 | test_accuracy_adaptive_max_pool3d_all_zero                               |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                            | test_attention_mask_broadcast                                            |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                            | test_attention_mask_broadcast_graph                                      |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                            | test_attention_mask_broadcast_gqa                                        |      1 |      0 |       0 |        1 |
| tests/test_convert_weight_to_int4pack.py                          | test_convert_weight_to_int4pack                                          |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_acc_false                                          |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_acc_true                                           |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_index_dtype                                        |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_error_all_none                                     |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_error_too_many_indices                             |      1 |      0 |       0 |        1 |
| tests/test_unsafe_index_put.py                                    | test_unsafe_index_put_mixed_none_and_tensor                              |      1 |      0 |       0 |        1 |
| tests/test_special_log_ndtr.py                                    | test_special_log_ndtr                                                    |      1 |      0 |       0 |        1 |
| tests/test_renorm_.py                                             | test_renorm_                                                             |      1 |      0 |       0 |        1 |
| tests/test_norm_except_dim.py                                     | test_norm_except_dim                                                     |      1 |      0 |       0 |        1 |
| tests/test_norm_except_dim.py                                     | test_norm_except_dim_dim                                                 |      1 |      0 |       0 |        1 |
| tests/test_norm_except_dim.py                                     | test_norm_except_dim_pow                                                 |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_optimized.py                            | test_choose_qparams_optimized                                            |      1 |      0 |       0 |        1 |
| tests/test_choose_qparams_optimized.py                            | test_choose_qparams_optimized_constant_input                             |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_many_short_rows                                             |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_many_tiny_rows_extrema                                      |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_accuracy_argsort                                                    |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_byte_boundaries                                             |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_stable_extrema                                              |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_scalar_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_invalid_dim                                                 |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_strided_3d                                                  |      1 |      0 |       0 |        1 |
| tests/test_argsort.py                                             | test_argsort_large_tiled_launch_boundary                                 |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                 | test_special_erf                                                         |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                 | test_erf                                                                 |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                 | test_erf_                                                                |      1 |      0 |       0 |        1 |
| tests/test_greater_.py                                            | test_greater_                                                            |      1 |      0 |       0 |        1 |
| tests/test_greater_.py                                            | test_greater_fp64_precision                                              |      1 |      0 |       0 |        1 |
| tests/test_is_same_size.py                                        | test_is_same_size                                                        |      1 |      0 |       0 |        1 |
| tests/test_is_same_size.py                                        | test_is_same_size_non_contiguous                                         |      1 |      0 |       0 |        1 |
| tests/test_is_same_size.py                                        | test_is_same_size_scalar_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_accuracy_logsumexp_backward                                         |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_logsumexp_backward_matches_aten_composite                           |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_accuracy_logsumexp_backward_large_and_neg_inf                       |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_accuracy_logsumexp_backward_non_contiguous                          |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_accuracy_logsumexp_backward_empty_reduction                         |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_logsumexp_backward_no_grad_input                                    |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_accuracy_logsumexp                                                  |      1 |      0 |       0 |        1 |
| tests/test_logsumexp.py                                           | test_logsumexp_backward_high_rank_and_duplicate_dims                     |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid.py                                         | test_hardsigmoid_                                                        |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid.py                                         | test_hardsigmoid__non_contiguous                                         |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid.py                                         | test_hardsigmoid__special_values                                         |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid.py                                         | test_hardsigmoid                                                         |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid.py                                         | test_hardsigmoid_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor_unsafe.py                            | test_sparse_csc_tensor_unsafe                                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor_unsafe.py                            | test_sparse_csc_tensor_unsafe_hybrid                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor_unsafe.py                            | test_sparse_csc_tensor_unsafe_dtype                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor_unsafe.py                            | test_sparse_csc_tensor_unsafe_skips_index_validation                     |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py | test_fake_quantize_learnable_per_channel_affine_backward                 |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py | test_fake_quantize_learnable_per_channel_affine_backward_extreme         |      1 |      0 |       0 |        1 |
| tests/test_per_token_group_quant_fp8.py                           | test_per_token_group_quant_fp8                                           |      1 |      0 |       0 |        1 |
| tests/test_dunder_or.py                                           | test_dunder_or                                                           |      1 |      0 |       0 |        1 |
| tests/test_dunder_or.py                                           | test_dunder_or_scalar                                                    |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_accuracy_conv_transpose2d_supported                                 |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_accuracy_conv_transpose2d_extended_parameters                       |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_invalid_groups_raise                               |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_invalid_arguments_raise                            |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_rejects_triplet_hyperparameters                    |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_unsupported_dtype_raise                            |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_unbatched_3d_matches_pytorch                       |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose2d.py                                    | test_conv_transpose2d_noncontiguous_tensors_match_pytorch                |      1 |      0 |       0 |        1 |
| tests/test_baddbmm.py                                             | test_baddbmm                                                             |      1 |      0 |       0 |        1 |
| tests/test_baddbmm.py                                             | test_baddbmm_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_baddbmm.py                                             | test_baddbmm_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test_baddbmm.py                                             | test_baddbmm_backward_broadcast                                          |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_generic_and_cached_binding_agree                                    |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_current_tensors_and_unspecialized_values_are_not_frozen             |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_defaults_and_explicit_keyword_values_share_correct_plan             |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_unspecialized_integer_width_matches_generic                         |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_alignment_and_dtype_changes_do_not_reuse_wrong_entry                |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_cached_binding_preserves_backend_tensor_specialization              |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_custom_constexpr_uses_generic_fallback                              |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_hooks_receive_current_arguments                                     |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_replaced_kernel_cache_entry_refreshes_launch_constants              |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_launch_plan_does_not_retain_input_tensors                           |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_offline_benchmark_clear_also_releases_launch_plans                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_cached_binding_requires_explicit_opt_in                             |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_unsupported_triton_uses_generic_binding                             |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_disabled_binding_does_not_enter_fast_path                           |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_cold_launch_uses_generic_path                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_cached_binding_preserves_custom_specialization_key                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_binding_plans_are_scoped_to_device                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                             | test_mode_change_clears_cached_binding_plans                             |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_basic                                                    |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_default_args                                             |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_negative_start_end                                       |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_step                                                     |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_edge_cases                                               |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_negative_dim                                             |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_int_dtype                                                |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_all_dtypes                                               |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_float8_bitwise                                           |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_complex                                                  |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_lazy_conj_neg                                            |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_non_contiguous_input                                     |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_invalid_args                                             |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_vs_aten_all_dtypes                                   |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_mismatched_size                                      |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_non_contiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_dtype_mismatch                                       |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_device_mismatch                                      |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_overlapping                                          |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_internal_overlap                                     |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_out_non_contiguous_input                                 |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                          | test_slice_copy_int64_index                                              |      1 |      0 |       0 |        1 |
| tests/test_lift_fresh_copy.py                                     | test_accuracy_lift_fresh_copy                                            |      1 |      0 |       0 |        1 |
| tests/test_lift_fresh_copy.py                                     | test_accuracy_lift_fresh_copy_empty                                      |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_tensor                                                         |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_scalar                                                         |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_tensor_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_scalar_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_scalar_                                                        |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_backward                                                       |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_tensor_value_gradient_mixed_dtype                              |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_tensor_double_backward                                         |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_scalar_sliced_view                                             |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_                                                               |      1 |      0 |       0 |        1 |
| tests/test_fill.py                                                | test_fill_sliced_view_tensor                                             |      1 |      0 |       0 |        1 |
| tests/test_floor.py                                               | test_floor                                                               |      1 |      0 |       0 |        1 |
| tests/test_floor.py                                               | test_floor_backward                                                      |      1 |      0 |       0 |        1 |
| tests/test_floor.py                                               | test_floor_                                                              |      1 |      0 |       0 |        1 |
| tests/test_floor.py                                               | test_floor_out                                                           |      1 |      0 |       0 |        1 |
| tests/test_floor.py                                               | test_floor_byte                                                          |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_                                                          |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_zero_boundary                                             |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_nan                                                       |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_inf                                                       |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_finfo_extremes                                            |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_y                                                  |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_x                                                  |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_nan_y                                              |      1 |      0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_inf_y                                              |      1 |      0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                          | test_replication_pad1d_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                          | test_replication_pad1d_backward_grad_output_dtype_mismatch               |      1 |      0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                          | test_replication_pad1d_backward_grad_input_takes_input_options           |      1 |      0 |       0 |        1 |
| tests/test_or_tensor.py                                           | test_or_tensor                                                           |      1 |      0 |       0 |        1 |
| tests/test_repeat_interleave.py                                   | test_repeat_interleave_self_int                                          |      1 |      0 |       0 |        1 |
| tests/test_repeat_interleave.py                                   | test_repeat_interleave_self_int_non_contiguous                           |      1 |      0 |       0 |        1 |
| tests/test_repeat_interleave.py                                   | test_repeat_interleave_tensor                                            |      1 |      0 |       0 |        1 |
| tests/test_repeat_interleave.py                                   | test_repeat_interleave_self_tensor                                       |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_basic                                      |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_with_range                                 |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_with_weight                                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_density                                    |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_boundary_and_outliers                      |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_auto_range_constant_dim                    |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_empty_input                                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_rejects_inverted_range                     |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_range_bound_formatting                     |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_auto_range_must_be_finite                  |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_finite_auto_range_unchanged                |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_auto_range_reports_lowest_dimension        |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_zero_width_explicit_range                  |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_rejects_non_finite_explicit_range          |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_finite_check_precedes_min_max              |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_zero_width_explicit_range_2d               |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_zero_width_edges_match_linspace            |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_out                                        |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_out_weighted_density                       |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_bin_boundary_precision                     |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_density_zero_total                         |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_density_cancelling_weights                 |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_out_resize                                 |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_out_dtype_mismatch                         |      1 |      0 |       0 |        1 |
| tests/test_histogramdd_from_bin_cts.py                            | test_histogramdd_from_bin_cts_out_non_contiguous                         |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout                                               |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_identity                                      |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_p_one_special_values                          |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_empty_returns_input                           |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_noncontiguous                                 |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_invalid_probability                           |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                               | test_feature_alpha_dropout_requires_two_dimensions                       |      1 |      0 |       0 |        1 |
| tests/test__linalg_slogdet.py                                     | test__linalg_slogdet                                                     |      1 |      0 |       0 |        1 |
| tests/test_assert_async.py                                        | test_assert_async                                                        |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_tensor_tensor_broadcast               |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_fast_degrees                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_special_x                             |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_promotion_and_nonfinite_degree        |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_large_degree_fast_exit_and_recurrence |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_is_not_differentiable                 |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_n_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_x_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_scalar_noncontiguous_layouts          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_scalar_float64_precision              |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_out                                   |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_out_resize_and_noncontiguous          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_n_scalar_out                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_x_scalar_out                          |      1 |      0 |       0 |        1 |
| tests/test_special_laguerre_polynomial_l.py                       | test_special_laguerre_polynomial_l_rejects_unsupported_compute_dtype     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_transpose.py                         | test_cudnn_convolution_transpose                                         |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized                                  |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_axis                             |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_float_zero_points                |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_device_default_cpu               |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_memory_format                    |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_requires_quantized_dtype         |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_rejects_bad_qparams              |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                  | test_empty_per_channel_affine_quantized_rejects_string_memory_format     |      1 |      0 |       0 |        1 |
| tests/test_special_i0.py                                          | test_special_i0                                                          |      1 |      0 |       0 |        1 |
| tests/test_special_i0.py                                          | test_special_i0_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_import_enumerates_all_cases_without_device_calls                    |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_ascend_known_fnuz_conversion_is_unsupported                         |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_unrelated_compiler_errors_are_not_skipped                           |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_metax_known_native_rowwise_rejection                                |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_other_native_errors_are_not_skipped                                 |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_resource_and_unknown_errors_propagate                               |      1 |      0 |       0 |        1 |
| tests/test_scaled_mm.py                                           | test_unselected_tests_do_not_probe                                       |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T                                                             |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_shares_storage                                              |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_low_rank_distinct_object                                    |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_noncontiguous                                               |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_strided_and_expanded                                        |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_lazy_conj_bit_forwarded                                     |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_autograd                                                    |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                             | test_numpy_T_rejects_non_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_cond.py                                         | test_linalg_cond                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_cond.py                                         | test_linalg_cond_with_p                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_cond.py                                         | test_linalg_cond_svd                                                     |      1 |      0 |       0 |        1 |
| tests/test_affine_grid_generator.py                               | test_affine_grid_generator                                               |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                      | test_fused_rms_norm                                                      |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                      | test_fused_rms_norm_no_weight                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                      | test_fused_rms_norm_aten_dispatch                                        |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                      | test_fused_rms_norm_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                      | test_fused_rms_norm_backward_no_weight                                   |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf                                                               |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_non_contiguous                                                |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_return_structure                                              |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                               | test_geqrf_zero_and_tiny_sizes                                           |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_accuracy_diag_embed                                                 |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward                                                 |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward_default_dims                                    |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                                          | test_diag_embed_backward_non_contiguous                                  |      1 |      0 |       0 |        1 |
| tests/test_topk.py                                                | test_topk                                                                |      1 |      0 |       0 |        1 |
| tests/test_topk.py                                                | test_topk_3d_lastdim                                                     |      1 |      0 |       0 |        1 |
| tests/test_topk.py                                                | test_topk_radix_tle_config_uses_large_fp32_shape_heuristic               |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm                                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_broadcast                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_rejects_shape_mismatch                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_dense_mat1_fallback                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_dense_mat1_uses_gems_addmm                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_out_resize                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_dispatch_reaches_gems_impl                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_sparse_addmm_out_dense_mat1                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_addmm.py                                        | test_out_overload_dispatch_reaches_gems_impl                             |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel                                                               |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_contiguous_view_path                                          |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_copy_path                                                     |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_copy_path_ranks                                               |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_complex                                                       |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_lazy_conj_bit_kept_on_view_path                               |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_neg_materialized_value                                        |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_autograd                                                      |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_rejects_non_tensor                                            |      1 |      0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_rejects_sparse_layout                                         |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_tensor                                                   |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_tensor_                                                  |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_mode_float                                               |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_mode_int                                                 |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_mode_float_                                              |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_mode_int_                                                |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_mode_float                                               |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_mode_int                                                 |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_mode_float_                                              |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_mode_int_                                                |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_scalar                                                   |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_scalar_                                                  |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_                                                         |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_tensor                                                   |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_scalar_scalar                                                   |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_complex_complex                                                 |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_complex_float_tensor                                            |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_tensor_int                                                      |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_complex_int_scalar                                              |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_out_tensor_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_out_tensor_scalar                                               |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_out_scalar_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_mode_tensor                                                     |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_mode_scalar                                                     |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_mode_tensor_                                                    |      1 |      0 |       0 |        1 |
| tests/test_div.py                                                 | test_div_mode_scalar_                                                    |      1 |      0 |       0 |        1 |
| tests/test_contiguous.py                                          | test_accuracy_contiguous                                                 |      1 |      0 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py           | test_amp_foreach_non_finite_check_and_unscale_                           |      1 |      0 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py           | test_amp_foreach_non_finite_check_and_unscale__inf                       |      1 |      0 |       0 |        1 |
| tests/test_amp_foreach_non_finite_check_and_unscale_.py           | test_amp_foreach_non_finite_check_and_unscale__nan                       |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_mixed                                                  |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_scalar_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_float                                                  |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_float_                                                 |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_int                                                    |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_int_                                                   |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_scalar_scalar                                          |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_scalar_inplace_float                                   |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                                        | test_floor_divide_scalar_inplace_int                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                                        | test_linalg_cross                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                                        | test_linalg_cross_noncontiguous_input_and_out                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                                        | test_linalg_cross_rejects_different_input_ranks                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                                        | test_linalg_cross_conjugated_view                                        |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py                              | test_unique_dim_consecutive_int                                          |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py                              | test_unique_dim_consecutive_float                                        |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py                              | test_unique_dim_consecutive_edge                                         |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py                              | test_unique_dim_consecutive_wide_rows                                    |      1 |      0 |       0 |        1 |
| tests/test_abs.py                                                 | test_abs                                                                 |      1 |      0 |       0 |        1 |
| tests/test_abs.py                                                 | test_abs_                                                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm                                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_channels_last                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_empty                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_zero_output_scale                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_bad_size                                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_bad_dtype                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out                                            |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_mismatched_qparams                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_resizes                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_storage_offset                             |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_empty                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_wrong_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_wrong_device                               |      1 |      0 |       0 |        1 |
| tests/test_mse_loss_backward.py                                   | test_mse_loss_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_pos_rejected                       |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dual_dim_bias                              |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_onesided_cumseq_rejected                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_varlen_rejected                            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_value_head_dim                             |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim                                                             |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_int_dtype                                                   |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_intlist_overload                                            |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_identity                                                    |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_result_aliases_input                                        |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_non_contiguous_input                                        |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_empty_tensor                                                |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_autograd                                                    |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_out_of_range                                                |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_destination_out_of_range                                    |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_length_mismatch                                             |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_repeated_axis                                               |      1 |      0 |       0 |        1 |
| tests/test_movedim.py                                             | test_movedim_0d_out_of_range                                             |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values                                                   |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values_keepdim_shapes                                    |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values_ties                                              |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values_strided_nonlast_reduction                         |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values_out_wrong_device                                  |      1 |      0 |       0 |        1 |
| tests/test_median_dim_values.py                                   | test_median_dim_values_out_rejects_bad_index_dtype                       |      1 |      0 |       0 |        1 |
| tests/test_bmm.py                                                 | test_bmm                                                                 |      1 |      0 |       0 |        1 |
| tests/test_bmm.py                                                 | test_bmm_out                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_noncontiguous_and_lower_triangle                |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_invalid_input                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_out                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_matrix_sqrth.py                                 | test_linalg_matrix_sqrth_out_resize                                      |      1 |      0 |       0 |        1 |
| tests/test_ior_tensor.py                                          | test_ior_tensor                                                          |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min                                                                 |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_dim                                                             |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward                                                        |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_ties                                                   |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_nan_forward_gap                                        |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_dim                                                    |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_dim_ties                                               |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_dim_non_contiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_min.py                                                 | test_min_backward_grid_stride_and_large_grid_limit                       |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_uses_flag_gems_view_path                                       |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_complex_view_semantics                                         |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_noncontiguous_complex_view                                     |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_non_complex_returns_input                                      |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_float8_returns_input                                           |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_autograd_view_relation                                         |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_conjugate_view_version_tracking                                |      1 |      0 |       0 |        1 |
| tests/test_real.py                                                | test_real_conjugate_nonleaf_inplace_view_replay                          |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half                                                           |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half_strided                                                   |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half_special_values                                            |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half_complex_layout                                            |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half_complex_noncontiguous_grad                                |      1 |      0 |       0 |        1 |
| tests/test_cast_Half.py                                           | test_cast_Half_autograd                                                  |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_accuracy                                                       |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_broadcast                                                      |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_non_contiguous                                                 |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_negative_p                                                     |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_fp64                                                           |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_empty                                                          |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                                | test_dist_empty_raises                                                   |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_                                                      |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_default_params                                        |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_mean_std                                              |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_inplace_semantics                                     |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_non_contiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_empty                                                 |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_batched_key                                           |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_stream_continuity                                     |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_large_offset                                          |      1 |      0 |       0 |        1 |
| tests/test_philox_normal_.py                                      | test_philox_normal_statistics                                            |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve                                                            |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_batch                                                      |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_broadcast                                                  |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_non_contiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_out_non_contiguous                                         |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_dtype_mismatch                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                            | test_lu_solve_pivots_dtype                                               |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_scalar_tensor.py                           | test_bitwise_xor_scalar_tensor                                           |      1 |      0 |       0 |        1 |
| tests/test_empty.py                                               | test_empty_permuted                                                      |      1 |      0 |       0 |        1 |
| tests/test_empty.py                                               | test_empty                                                               |      1 |      0 |       0 |        1 |
| tests/test_empty.py                                               | test_empty_default_dtype                                                 |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy                                                               |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_out                                                           |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_special_values                                                |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_tensor_scalar                                                 |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_tensor_scalar_out                                             |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_scalar_tensor                                                 |      1 |      0 |       0 |        1 |
| tests/test_xlogy.py                                               | test_xlogy_scalar_tensor_out                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_mm.py                           | test_sparse_semi_structured_mm                                           |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_tensor                                                         |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_scalar                                                         |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                             | test_max_pool3d_with_indices                                             |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                             | test_max_pool3d_with_indices_ties                                        |      1 |      0 |       0 |        1 |
| tests/test_acosh.py                                               | test_acosh                                                               |      1 |      0 |       0 |        1 |
| tests/test_acosh.py                                               | test_acosh_                                                              |      1 |      0 |       0 |        1 |
| tests/test_acosh.py                                               | test_acosh__integer_rejects                                              |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                                  | test_as_strided_scatter                                                  |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                                  | test_as_strided_scatter_preserves_storage_geometry                       |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                                  | test_as_strided_scatter_noncontiguous_self                               |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal                                                       |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar                                                |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar_byte                                           |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_                                                      |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal__byte                                                 |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal__byte_scalar                                          |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar_                                               |      1 |      0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar__byte                                          |      1 |      0 |       0 |        1 |
| tests/test_glu_jvp.py                                             | test_glu_jvp                                                             |      1 |      0 |       0 |        1 |
| tests/test_glu_jvp.py                                             | test_glu_jvp_non_contiguous                                              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_measurement_preserves_native_execution_and_skip_contract            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_named_binding_handles_default_parameters                            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_no_baseline_does_not_describe_inputs_or_query_device                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_real_tensor_descriptors_keep_layout_nested_roles_and_mixed_dtypes   |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_pytest_option_loads_baseline_and_can_reset_to_disabled              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_scheduler_passes_absolute_baseline_path_as_one_shell_argument       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_scheduler_imports_shared_report_parser                              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_factory_benchmark_passes_target_device_to_external_matching         |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_device_objects_are_described_without_allocating_on_an_accelerator   |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_argsort_subclass_accepts_dtype_in_both_runners                      |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_blas_serial_and_worker_run_apply_external_baseline                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_blas_subprocess_forwards_baseline_and_preserves_worker_result       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_blas_paired_inputs_match_native_abi_before_timing                   |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_blas_external_miss_retains_native_result                            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline_runtime.py                      | test_named_metadata_missing_default_parameter_keeps_native_ratio         |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy                                       |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_dtype_mapping                         |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_values                                |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_noncontiguous_batch                   |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_transposed                            |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_high_rank_noncontiguous               |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_expanded                              |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_storage_offset                        |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_independent_storage                   |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_empty                                 |      1 |      0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                | test_accuracy_view_as_complex_copy_error_paths                           |      1 |      0 |       0 |        1 |
| tests/test_greater.py                                             | test_greater                                                             |      1 |      0 |       0 |        1 |
| tests/test_greater.py                                             | test_greater_scalar                                                      |      1 |      0 |       0 |        1 |
| tests/test_greater.py                                             | test_greater_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_greater.py                                             | test_greater_scalar_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_atanh.py                                               | test_atanh                                                               |      1 |      0 |       0 |        1 |
| tests/test_atanh.py                                               | test_atanh__integer_rejects                                              |      1 |      0 |       0 |        1 |
| tests/test_count_nonzero.py                                       | test_count_nonzero                                                       |      1 |      0 |       0 |        1 |
| tests/test_zeros.py                                               | test_zero_inplace_boundaries_exact                                       |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_overwrites_special_values_exact                                |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_noncontiguous_storage_exact                                    |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_repeated_calls_exact                                           |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zeros_factory_poisoned_allocation_exact                             |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_launch_core_dim_limit                                          |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_real_large_input_exact                                         |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zero_capped_grid_covers_all_tiles_exact                             |      0 |      0 |       1 |        1 |
| tests/test_zeros.py                                               | test_zeros_factory_capped_grid_exact                                     |      0 |      0 |       1 |        1 |
| tests/test_argsort.py                                             | test_argsort_ascend_launcher_abi                                         |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm                                                           |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm_out                                                       |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm_empty                                                     |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm_schema                                                    |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm_fast_route                                                |      0 |      0 |       1 |        1 |
| tests/test_scaled_mm.py                                           | test_scaled_mm_descriptor_dtype_switch                                   |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8                                                         |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_prequantized                                   |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_large_k_scales                                 |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_skinny_epilogue                                |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_dense_reuse                                    |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_graph_updates                                           |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_empty                                                   |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_output_dtype                                            |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_independent_and_invalid_out                             |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_broadcast                                      |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_invalid_inputs_and_scales                               |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_scalar_shapes                                           |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_scaled_mm_interface                            |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_default_dtype                                  |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_invalid_epilogue                               |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_mthreads_strided_epilogue                               |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_hopper_external_scales                                  |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_hopper_torch_interface                                  |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                         | test_mm_w8a8_fp8_hopper_torch_rowwise                                    |      0 |      0 |       1 |        1 |
| tests/test_topk.py                                                | test_topk_radix_tle_large_fp32_k256_correctness                          |      0 |      0 |       1 |        1 |
| tests/test_topk.py                                                | test_topk_ascend_dsa                                                     |      0 |      0 |       1 |        1 |
| tests/test_ravel.py                                               | test_ravel_strided_offset_lazy_bits                                      |      0 |      0 |       1 |        1 |
| tests/test_ravel.py                                               | test_ravel_bfloat16_lazy_neg_copy                                        |      0 |      0 |       1 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_scale                                      |      0 |      0 |       1 |        1 |
| tests/test_bmm.py                                                 | test_bmm_w8a8_fp8                                                        |      0 |      0 |       1 |        1 |
| tests/test_min.py                                                 | test_min_all_inf                                                         |      0 |      0 |       1 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_neg_degrades_to_zero               |      0 |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_attn_bias_shapes                           |      0 |      1 |       0 |        1 |
| TOTAL                                                             |                                                                          |    530 |      3 |      42 |      575 |
|                                 filepath                                 |                             function                             | passed | skipped | xfailed | xpassed | SUBTOTAL |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------- | -----: | ------: | ------: | ------: | -------: |
| tests/test_nonzero_static.py                                             | test_nonzero_static_dtypes                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_cases                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_non_contiguous                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_argument_errors                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_rejects_bool_arguments                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_out                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_cuda_complex                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_cuda_paths                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_semi_structured_tile.py                                | test_sparse_semi_structured_tile                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_contiguous                              |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_non_contiguous                          |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_transposed_and_expanded                 |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_levels                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_mutation_through_view                   |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_dispatch_stability                      |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_inference_tensor                        |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_scalar_and_empty                        |      1 |       0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                                                 | test_accuracy__fw_primal_dtype_dispatch                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_erfinv.py                                             | test_special_erfinv                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_erfinv.py                                             | test_special_erfinv_out                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_erfinv.py                                             | test_special_erfinv_                                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                           | test_reflection_pad3d                                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                           | test_reflection_pad3d_list_padding                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                           | test_reflection_pad3d_empty_padding                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                           | test_reflection_pad3d_4d_input                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                           | test_reflection_pad3d_out                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_quant_params                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_qint_dtypes                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_qint_dtypes                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_default_stride                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_default_stride                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_invalid_params                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_invalid_params                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_invalid_rank                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_zero_sized_dims                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_empty_batch                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_rejects_non_quantized                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_rejects_per_channel                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_resizes                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_adopts_input_qparams               |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_dtype_mismatch                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_preserves_storage_offset           |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_out_non_contiguous                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_quantized_max_pool3d.py                                       | test_quantized_max_pool3d_non_contiguous_input                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_pad_packed_sequence.py                                        | test_pad_packed_sequence                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_pad_packed_sequence.py                                        | test_pad_packed_sequence_padding                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_dimV.py                                                       | test_accuracy_dimV_coo                                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_dimV.py                                                       | test_accuracy_dimV_matches_indices_formula                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_dimV.py                                                       | test_accuracy_dimV_unsupported_layouts_raise                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_dimV.py                                                       | test_accuracy_dimV_dispatch_stability                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_deepseek_v4_attention_compute_global_topk_indices_and_lens.py | test_compute_global_topk_indices_and_lens_accuracy               |      1 |       0 |       0 |       0 |        1 |
| tests/test_cdist_forward.py                                              | test_cdist_forward                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_and_tensor.py                                                 | test_and_tensor                                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero.py                                                    | test_nonzero                                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_grid_sampler_2d_backward                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_accuracy_grid_sampler_2d_backward_grad_input_only           |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_accuracy_grid_sampler_2d_backward_grad_grid_only            |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_grid_sampler_2d_backward_out_of_range_grid                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_grid_sampler_2d_backward_bicubic_raises                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_grid_sampler_2d_backward.py                                   | test_grid_sampler_2d_backward_reflection_boundaries              |      1 |       0 |       0 |       0 |        1 |
| tests/test_fix.py                                                        | test_fix                                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential.py                                                | test_exponential                                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential.py                                                | test_exponential_moments                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential.py                                                | test_exponential_far_left_tail                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_float_power_.py                                               | test_float_power_                                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_float_power_.py                                               | test_float_power_scalar                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky                                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_upper                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_batch                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_non_contiguous                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_non_contiguous_batch                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_batched_4d                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_not_positive_definite_raises                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_empty_batch                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_non_square_raises                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky.py                                                   | test_cholesky_large_n_fallback                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy_int                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy_noncontiguous                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy_out                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy_high_rank                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                             | test_fw_primal_copy_out_high_rank                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_scalar_tensor.py                                              | test_scalar_tensor                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_round.py                                              | test_special_round                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_round.py                                              | test_special_round_out                                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_round.py                                              | test_special_round_midpoints                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_round.py                                              | test_special_round_decimals                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_round.py                                              | test_special_round_out_decimals                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_std.py                                                        | test_std                                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py                          | test_upsample_nearest_exact2d_backward                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest_exact2d_backward.py                          | test_upsample_nearest_exact2d_backward_with_scales               |      1 |       0 |       0 |       0 |        1 |
| tests/test_negative_.py                                                  | test_negative_                                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_frobenius_norm.py                                             | test_frobenius_norm                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_frobenius_norm.py                                             | test_frobenius_norm_zero_size                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_frobenius_norm.py                                             | test_frobenius_norm_duplicate_dim                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial                                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial_large_count                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial_varying_prob                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial_edge_probs                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial_zero_count                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_binomial.py                                                   | test_binomial_out                                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_cosine_similarity.py                                          | test_cosine_similarity_accuracy                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cosine_similarity.py                                          | test_cosine_similarity_broadcast                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_cosine_similarity.py                                          | test_cosine_similarity_eps_clamp                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_cosine_similarity.py                                          | test_cosine_similarity_fp64                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_gammaln.py                                            | test_special_gammaln                                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_gammaln.py                                            | test_special_gammaln_out                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_gammaln.py                                            | test_special_gammaln_edge_cases                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_grad_input                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_grad_input_contiguous           |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_channels_last_tiles             |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_strided_tiles                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_cpu_legacy_reference            |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_native_device_scales            |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_scale_contract                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_grad_input_scale_contract       |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_grad_input_resize               |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_grad_input_mismatch             |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_invalid                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_requires_positive_scales        |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_deterministic                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_accumulates_fp32                |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_preserves_fp64                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_uint8                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_int64_address                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_median_out.py                                                 | test_median_out                                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_median_out.py                                                 | test_median_out_even_count_picks_lower_middle                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_median_out.py                                                 | test_median_out_rejects_bad_out_dtype                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_median_out.py                                                 | test_median_out_rejects_cpu_out                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_larger_shapes                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_rhs_boundaries                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_upper                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_transpose_contiguous_factor                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_blocked_single_rhs                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_small_gather_single_rhs                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_fp64_blocked                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_fp32_blocked_upper                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_fp32_blocked_lower                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_batch                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_broadcast_batch                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_noncontiguous_inputs                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_scaled_inputs                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_conditioned_matrix                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_accuracy                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_direct                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_noncontiguous                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_broadcast                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_small_gather_conditioned             |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_blocked                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_blocked_conditioned                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_lazy_conjugate_upper                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_complex_lazy_conjugate_with_all_gems         |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out_resizes_for_broadcast_result             |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out_noncontiguous_and_alias                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out_safe_dtype_cast                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out_rejects_incompatible_dtype               |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_out_rejects_different_device                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_empty_input                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_invalid_inputs                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_isclose.py                                                    | test_isclose                                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take                                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_out                                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_negative_index                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_out_of_bounds                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_noncontiguous                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_out_noncontiguous                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_out_resizes_empty_tensor                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_out_rejects_mismatched_dtype                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_empty                                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_take.py                                                       | test_take_multidim_index                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_maximum.py                                                    | test_maximum                                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_maximum.py                                                    | test_maximum_cpu_scalar_operand                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_maximum.py                                                    | test_maximum_cpu_scalar_precision                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_maximum.py                                                    | test_maximum_rejects_non_scalar_cpu_operand                      |      1 |       0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                            | test_status_uses_stderr_and_flush                                |      1 |       0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                            | test_exception_chain_redacts_url_secrets                         |      1 |       0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                            | test_status_closed_stream_does_not_fail                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_empty                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_noncontiguous                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_noncontiguous_multidim_lead              |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_recovers_known_solution                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_dims                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_dims_defaults_to_none                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_dims_out_of_range                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_dims_duplicate                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_shape_mismatch                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_residual_is_small                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_singular_does_not_hang_or_crash          |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_rejects_complex                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_fp64_preserved                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_fp64_accuracy_is_double_grade            |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                         | test_linalg_tensorsolve_autograd_input_is_detached               |      1 |       0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool1d.py                                        | test_adaptive_max_pool1d_accuracy                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool1d.py                                        | test_adaptive_max_pool1d_list_output_size                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool1d.py                                        | test_adaptive_max_pool1d_edge_cases                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_lerp.py                                                       | test_lerp                                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_lerp.py                                                       | test_lerp_                                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_lerp.py                                                       | test_lerp_scalar                                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_lerp.py                                                       | test_lerp_scalar_                                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse                                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_batch                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_requires_pivoting                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_identity_product                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_noncontiguous                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_empty                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_unsupported_dtype                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_flash_attn_varlen_func.py                                     | test_flash_attn_varlen_func                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_flash_attn_varlen_func.py                                     | test_flash_attn_varlen_func_noncontiguous_kv_cache               |      1 |       0 |       0 |       0 |        1 |
| tests/test_flash_attn_varlen_func.py                                     | test_flash_attn_varlen_func_swap_qg                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp                                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_multi_dim                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_single_element                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_large_values                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_negative_large_values                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_all_negative_inf                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_zeros                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_special_logsumexp.py                                          | test_special_logsumexp_extreme_mixed                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv                                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_batched                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_large                                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_large_batched                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_out                                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_rejects_complex                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_out_resizes_and_warns                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_out_rejects_unsafe_cast                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_linalg_pinv.py                                                | test_linalg_pinv_out_noncontiguous                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_                                                |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_fast                                            |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_backward_                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_inplace_version                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_lower_tail                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_no_repeated_blocks                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_far_left_tail                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_consecutive_calls_advance_generator             |      1 |       0 |       0 |       0 |        1 |
| tests/test_exponential_.py                                               | test_exponential_race_matches_categorical                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_cast_Short.py                                                 | test_cast_Short                                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_cast_Short.py                                                 | test_cast_Short_strided                                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_cast_Short.py                                                 | test_cast_Short_complex_real_component                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian                                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_large_flat                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_hygon_flat_above_block_limit                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_large_radix_path                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_all_nan_rows                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_non_contiguous                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_out                                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_values                                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_values_large_int                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_values_non_contiguous_out                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_empty                                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_out_empty                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_empty                                         |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_bool_unsupported                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_byte_boundaries                                   |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_out_byte_boundaries                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_byte_boundaries                               |      1 |       0 |       0 |       0 |        1 |
| tests/test_nanmedian.py                                                  | test_nanmedian_dim_values_byte_boundaries                        |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_self                                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_self_scalar                                           |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_self_scalar_other                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_self_out                                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_self_out_cross_device                                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_uniform_condition_strided_storage                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_uniform_condition_boundaries                          |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_uniform_condition_scalar                              |      1 |       0 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_invalid_broadcast_raises_runtime_error                |      1 |       0 |       0 |       0 |        1 |
| tests/test_clamp.py                                                      | test_clamp                                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_clamp.py                                                      | test_clamp_                                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_nnz                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_aliases_payload     |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_default_uncoalesced |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_forwards_coalesced  |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_out                 |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_registration_names  |      1 |       0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims_and_tensors.py                    | test_sparse_coo_tensor_with_dims_and_tensors_rejects_bad_layout  |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_value_range                          |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_wide_values                          |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_accumulate_matches                   |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_stored_layout                        |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_hybrid_coo                           |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_strided_storage                      |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_special_values                       |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_out                                  |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_out_stored_layout                    |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_dense_self                   |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_dense_mask                   |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_size_mismatch                |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_sparse_dim_mismatch          |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_invalid_accumulate           |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_dense_out                    |      1 |       0 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_dispatch                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_feature_dropout.py                                            | test_feature_dropout                                             |      1 |       0 |       0 |       0 |        1 |
| tests/test_feature_dropout.py                                            | test_feature_dropout_no_train                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_feature_dropout.py                                            | test_feature_dropout_p_zero                                      |      1 |       0 |       0 |       0 |        1 |
| tests/test_feature_dropout.py                                            | test_feature_dropout_p_one                                       |      1 |       0 |       0 |       0 |        1 |
| tests/test_feature_dropout.py                                            | test_feature_dropout_inplace                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_new_ones.py                                                   | test_new_ones                                                    |      1 |       0 |       0 |       0 |        1 |
| tests/test_new_ones.py                                                   | test_new_ones_different_size                                     |      1 |       0 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_ascend_paths                                 |      0 |       1 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_ascend_sparse_group_fallback                 |      0 |       1 |       0 |       0 |        1 |
| tests/test_nonzero_static.py                                             | test_nonzero_static_ascend_bfloat16_special_values               |      0 |       1 |       0 |       0 |        1 |
| tests/test_deepseek_v4_attention_compute_global_topk_indices_and_lens.py | test_compute_global_topk_indices_and_lens_vllm_accuracy          |      0 |       1 |       0 |       0 |        1 |
| tests/test_upsample_nearest2d_backward.py                                | test_upsample_nearest2d_backward_iluvatar_rejects_fp64           |      0 |       1 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_ascend_batched_small_lower_single_rhs        |      0 |       1 |       0 |       0 |        1 |
| tests/test_cholesky_solve.py                                             | test_cholesky_solve_ascend_blocked_single_rhs_conditioned        |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_gate_engages                                  |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_gate_defers                                   |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_gate_limits_follow_element_width              |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax                                               |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_out                                           |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_half_to_float                                 |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_neg_inf                                       |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_at_row_cap                                    |      0 |       1 |       0 |       0 |        1 |
| tests/test_rdna4_softmax.py                                              | test_rdna4_softmax_non_contiguous                                |      0 |       1 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                            | test_model_loaded_once_per_version                               |      0 |       1 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                            | test_proposer_cache_separates_outer_package_versions             |      0 |       1 |       0 |       0 |        1 |
| tests/test_inverse.py                                                    | test_inverse_singular                                            |      0 |       1 |       0 |       0 |        1 |
| tests/test_rms_norm_w8a16_int8.py                                        | test_rms_norm_w8a16_int8                                         |      0 |       1 |       0 |       0 |        1 |
| tests/test_rms_norm_w8a16_int8.py                                        | test_rms_norm_w8a16_int8_ragged_shapes                           |      0 |       1 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_uniform_condition_overlapping_output_rejected         |      0 |       1 |       0 |       0 |        1 |
| tests/test_where_self_out.py                                             | test_where_uniform_condition_graph_reloads_inputs                |      0 |       1 |       0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                                    | test_sparse_mask_projection_rejects_fp8                          |      0 |       1 |       0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                        | test_chunk_gated_delta_rule_fwd_accuracy                         |      0 |       0 |       1 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                        | test_chunk_gated_delta_rule_fwd_no_initial_state                 |      0 |       0 |       1 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                        | test_chunk_gated_delta_rule_fwd_with_cu_seqlens                  |      0 |       0 |       0 |       1 |        1 |
| TOTAL                                                                    |                                                                  |    306 |      24 |       2 |       1 |      333 |
