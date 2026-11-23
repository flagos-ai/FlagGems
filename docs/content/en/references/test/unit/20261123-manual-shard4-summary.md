|                             filepath                              |                                function                                 | passed | failed | skipped | xpassed | SUBTOTAL |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- | -----: | -----: | ------: | ------: | -------: |
| tests/core/test_reference_only.py                                 | test_benchmark_runs_original_baseline_once                              |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_backward_reference_uses_original_grad_semantics                    |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_skip_native_is_not_a_pass_or_failure                               |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_failure_is_not_passed                                    |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_capability_failures_are_recorded_individually_and_later_cases_run  |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_failure_categories_do_not_guess_api_or_dtype_support               |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_capability_failure_does_not_continue_after_sync_failure            |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_interrupt_propagates_and_leaves_unexecuted_cases         |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_capability_failure_before_invocation_keeps_zero_call_count         |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_source_skip_preserves_node_semantics_and_unexecuted_records        |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_unsupported_custom_runner_cannot_execute_candidate                 |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_conflicting_modes_are_rejected                                     |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_report_replaces_stale_data_and_preserves_case_skips      |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_unknown_case_selection_fails                             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness   |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_mode_cannot_run_correctness_pytest                       |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                 | test_reference_only_defaults_to_core_without_changing_normal_defaults   |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_model_source.py                               | ModelSourceTests.test_defaults_and_once_only                            |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_model_source.py                               | ModelSourceTests.test_explicit_sources_including_empty_are_preserved    |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_model_source.py                               | ModelSourceTests.test_manifest_schema_and_wheel_data                    |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_model_source.py                               | ModelSourceTests.test_version_preferences_preserved                     |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_status_uses_stderr_and_flush                                       |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_exception_chain_redacts_url_secrets                                |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_status_closed_stream_does_not_fail                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardWraps.test_name_preserved                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardWraps.test_doc_preserved                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardWraps.test_wrapped_attribute_points_to_original           |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_single_non_contiguous_arg                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_multiple_non_contiguous_args              |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_already_contiguous_arg_stays_contiguous   |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_single_non_contiguous_kwarg               |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_multiple_non_contiguous_kwargs            |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_mixed_args_and_kwargs_all_contiguous      |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_non_tensor_arg_passed_through_unchanged   |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_non_tensor_kwarg_passed_through_unchanged |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_contiguous_version_has_same_data          |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContiguous.test_return_value_preserved                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_output_on_correct_device                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_all_available_devices                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_current_device_matches_input                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_nullcontext_when_no_tensor                   |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_default_params                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_inplace_semantics                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_value_range                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_batched_key                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_stream_continuity                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                    | test_philox_uniform_large_offset                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_absolute.py                                            | test_absolute                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_absolute.py                                            | test_absolute_                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_absolute.py                                            | test_absolute__non_contiguous                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool2d.py                                 | test_adaptive_max_pool2d                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool2d.py                                 | test_adaptive_max_pool2d_nan_and_tie_indices                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_addcdiv_.py                                            | test_addcdiv_                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_affine_grid_generator.py                               | test_affine_grid_generator                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_any.py                                                 | test_any                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_any.py                                                 | test_any_dims                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_any.py                                                 | test_any_dim                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_arctan_.py                                             | test_arctan                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_arctan_.py                                             | test_arctan_                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_assert_async.py                                        | test_assert_async                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_avg_pool3d.py                                          | test_avg_pool3d                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_avg_pool3d.py                                          | test_avg_pool3d_backward                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims_dtypes                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims_default_dtype                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims_out                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims_rejects_bad_layout                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_with_dims.py                         | test_sparse_coo_tensor_with_dims_rejects_size_mismatch                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_bernoulli.py                                           | test_bernoulli_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_bernoulli.py                                           | test_bernoulli_various_p                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_bernoulli.py                                           | test_bernoulli                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_bitwise_and_scalar_tensor.py                           | test_bitwise_and_scalar_tensor                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_memory_format                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_device_default_cpu                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_out_matches_reference                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_requires_quantized_dtype                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_rejects_string_memory_format                |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_affine_quantized.py                              | test_empty_affine_quantized_registered_in_full_config                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_cauchy.py                                              | test_cauchy_accuracy                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cauchy.py                                              | test_cauchy_out_accuracy                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cauchy.py                                              | test_cauchy_reproducibility                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cauchy.py                                              | test_cauchy_large_tensors                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_channel_shuffle.py                                     | test_channel_shuffle                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_clamp_min.py                                           | test_clamp_min                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_clamp_min.py                                           | test_clamp_min_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_higher_rank                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_extreme                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_compute_linear_combination.py                          | test_compute_linear_combination_out                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_weight_to_int4pack.py                          | test_convert_weight_to_int4pack                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_hybrid                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_dtype                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_tensor_unsafe.py                            | test_sparse_csr_tensor_unsafe_skips_index_validation                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_value_ranges                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_sparse_self                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_non_contiguous                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_nan_inf                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_backward                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_shape_mismatch                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_rejects_dense_mask                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_mask.py                                         | test_sparse_mask_dispatch                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_dilation                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_asymmetric                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_3d                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_1d                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_no_bias                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_bias_shapes                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_boundary                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_out                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                              | test_cudnn_convolution_relu_out_no_bias                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_pos_rejected                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dual_dim_bias                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_onesided_cumseq_rejected                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_varlen_rejected                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_value_head_dim                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_cummin.py                                              | test_cummin                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cummin.py                                              | test_cummin_with_nan                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine.py           | test_fake_quantize_learnable_per_tensor_affine                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine.py           | test_fake_quantize_learnable_per_tensor_affine_grad_factor              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_autograd_chain                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_non_contiguous                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_middle_and_negative_dim                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_empty                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_scalar_input                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                     | test_cumsum_backward_use_gems_entry                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_strided                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_strided_views                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_sparse_coo                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_sparse_compressed                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_nested                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_matches_reference_table                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_aliases_agree                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_is_dispatch_free                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_dtype_and_device_independence                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_requires_grad_and_meta                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_arg_validation                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_dim.py                                                 | test_dim_with_registration_active                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default_empty                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_default_uncoalesced                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dtype                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_all                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_dtype                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_out                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_negative                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_keeps_structural_zeros                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_hybrid_dense_dims                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_uncoalesced_input                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_zero_size_dims                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_integer_promotion                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_duplicate_dim_raises                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_out_of_range_dim_raises                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dense_input_raises                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_csr_input_raises                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dim_out_dtype_mismatch_raises                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_dtype_accumulator_precision                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_complex                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_dispatch_reaches_gems_impl                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_dispatch_covers_every_overload                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                                          | test_sparse_sum_uncoalesced_matches_aten                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_elu.py                                                 | test_elu                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_elu.py                                                 | test_elu_                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_elu.py                                                 | test_elu_backward                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                     | test_embedding_spmdm_matches_embedding_bag                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                     | test_embedding_spmdm_index_dtypes                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                     | test_embedding_spmdm_noncontiguous_inputs                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                     | test_embedding_spmdm_validation                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_exp.py                                                 | test_exp                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_exp.py                                                 | test_exp_                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_exp.py                                                 | test_exp_out                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_eye.py                                                 | test_eye                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_eye.py                                                 | test_eye_m                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward                |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_mask_semantics |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_noncontiguous  |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask_backward.py | test_fake_quantize_per_channel_affine_cachemask_backward_empty          |      1 |      0 |       0 |       0 |        1 |
| tests/test_fft.py                                                 | test_fft                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py      | test_combine_topk_swa_indices_accuracy                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_float_power_.py                                        | test_float_power_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_float_power_.py                                        | test_float_power_scalar                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                               | test_fractional_max_pool2d                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                               | test_fractional_max_pool2d_backward                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_strided                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_fractional                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_requires_grad                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_complex                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_integer_limits                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Char.py                                           | test_cast_Char_negative_view                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_shapes                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_dim_forms                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_dim_forms_large                               |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_zero_nnz_and_zero_axis                        |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_noncontiguous                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_stored_offsets                                |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_out                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_uncoalesced_self                              |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_duplicate_coordinates                         |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_special_values                                |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_copy_special_values                           |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_regather_rejects_non_float_dtypes             |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_negative_dense_self                           |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_negative_out_of_range_dim                     |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_negative_aliased_dim                          |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test__sparse_sum_backward_negative_sparse_grad_for_summed_dims          |      1 |      0 |       0 |       0 |        1 |
| tests/test__sparse_sum_backward.py                                | test_sparse_sum_backward_dispatch                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_gather_block_quantized.py                              | test_gather_block_quantized                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_get_paged_mqa_logits_metadata.py                       | test_get_paged_mqa_logits_metadata                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar_byte                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal__byte                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal__byte_scalar                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar_                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                       | test_greater_equal_scalar__byte                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_strided                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_special_values                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_integer_limits                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_autograd                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_complex                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_negative_view                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_cpu                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Float.py                                          | test_cast_Float_sparse                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh_explicit                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh_out                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh_out_explicit                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh_special_values                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardtanh.py                                            | test_hardtanh_out_special_values                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_hinge_embedding_loss.py                                | test_hinge_embedding_loss                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_huber_loss.py                                          | test_huber_loss                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_huber_loss.py                                          | test_huber_loss_out                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_isreal.py                                              | test_isreal                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_isreal.py                                              | test_isreal_special_values                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_isreal.py                                              | test_isreal_non_contiguous                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_isreal.py                                              | test_isreal_empty                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_isreal.py                                              | test_isreal_conj_bit                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_select.py                                        | test_index_select                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_any_true.py                                         | test_is_any_true                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_any_true.py                                         | test_is_any_true_arbitrary_rank                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_any_true.py                                         | test_is_any_true_scalar_and_expand                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_any_true.py                                         | test_is_any_true_invalid_dtype                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_gelu_and_mul.py                                        | test_gelu_and_mul                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_2x2_kernel                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_jacobi                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_complex                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_batch_2x2_kernel                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_batch_jacobi                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_trivial                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_batch_trivial                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_2x2_low_precision                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_trivial_complex                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_batch_complex                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_uplo_upper                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_uplo_upper_complex                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_jacobi_global                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_complex_global                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_ascending_order                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_nonsquare_raises                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_linalg_eigh_non_contiguous                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_underlying_linalg_eigh                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_underlying_linalg_eigh_no_vectors                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_underlying_linalg_eigh_no_vectors_2x2                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_underlying_linalg_eigh_no_vectors_global                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                         | test_underlying_linalg_eigh_no_vectors_complex_global                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss_int_reduction                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss_broadcast                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss_empty                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss_functional                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_l1_loss.py                                             | test_l1_loss_invalid_reduction                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_leaky_relu.py                                          | test_leaky_relu                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_leaky_relu.py                                          | test_leaky_relu_                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_leaky_relu.py                                          | test_leaky_relu_out                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_leaky_relu.py                                          | test_leaky_relu_backward                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                               | test_cholesky_solve_helper                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                               | test_cholesky_solve_helper_layouts                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                               | test_cholesky_solve_helper_batch_strides                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                               | test_cholesky_solve_helper_empty                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_empty                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_noncontiguous                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_identity                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_degenerate_eigenvalues                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_ascending_order                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_trace_is_preserved                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_reads_requested_triangle                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_default_uplo_is_lower                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_batch_independence                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_invalid_shape                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_invalid_uplo                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_rejects_complex                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_fp64_preserved                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_autograd_input_is_detached                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_out                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_out_resize                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                     | test_linalg_eigvalsh_out_rejects_mismatched_dtype                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__multitensor                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__grad_scale                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__found_inf                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__tensor_lr                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__rejects_mixed_dtype_list                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__rejects_grad_dtype_mismatch                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__single_launch                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__heterogeneous_batch_accuracy                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__cuda_graph_capture                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__cuda_graph_capture_amp                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__cuda_graph_capture_tensor_lr                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_common                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_large                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_out_parameter                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_linalg_matrix_power_out                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_out_missing_raises                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_power.py                                 | test_invalid_shape_rejected                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_slogdet.py                                      | test_linalg_slogdet                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vander.py                                       | test_linalg_vander                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vander.py                                       | test_linalg_vander_with_N                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vander.py                                       | test_linalg_vander_complex                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vander.py                                       | test_linalg_vander_with_N_complex                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal.py                                          | test_log_normal                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal.py                                          | test_log_normal_params                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal.py                                          | test_log_normal_moments                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal.py                                          | test_log_normal_moments_large_std                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal.py                                          | test_log_normal_large_shapes                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_out                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_autograd_chain                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_non_contiguous                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_negative_and_middle_dim                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_wide_range                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_zero_length                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_scalar_input                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_logcumsumexp.py                                        | test_logcumsumexp_backward_use_gems_entry                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit_backward.py                                      | test_logit_backward                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit_backward.py                                      | test_logit_backward_eps                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_singular                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_singular_positions                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_empty                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_info_shape                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_info_ignores_nonfinite_pivots                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_info_reports_nan_only_as_nonzero_when_zero_absent               |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_empty_batch_dimensions                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_scatter_backward.py                             | test_accuracy_masked_scatter_backward                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_matrix_exp.py                                          | test_linalg_matrix_exp                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_matrix_exp.py                                          | test_linalg_matrix_exp_batch                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_matrix_exp.py                                          | test_linalg_matrix_exp_identity                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_matrix_exp.py                                          | test_linalg_matrix_exp_zero                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_max_pool3d_with_indices_backward.py                    | test_accuracy_max_pool3d_with_indices_backward                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_max_pool3d_with_indices_backward.py                    | test_accuracy_max_pool3d_with_indices_backward_channels_last_3d         |      1 |      0 |       0 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_post_vs_ref                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_split_sinkhorn                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_pre_vs_ref                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_mhc_bwd_vs_ref                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_hc_head_fused_kernel_vs_ref                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_csc                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_csc_batched                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_bsc                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_strided_ccol_1d                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_strided_ccol_multi_dim                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_empty                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_zero_columns                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_input_not_mutated                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_rejects_non_column_compressed           |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_coo_falls_through_to_native             |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_dtype_of_index_storage                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_dispatch_stability                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_out_device_mismatch                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_out_writes_and_returns_out              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_out_resizes                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_out_dtype_mismatch                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ccol_indices_copy.py                                   | test_accuracy_ccol_indices_copy_out_rejects_non_column_compressed       |      1 |      0 |       0 |       0 |        1 |
| tests/test_mse_loss.py                                            | test_mse_loss                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply_.py                                           | test_multiply_                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply_.py                                           | test_multiply_tensor_scalar_                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_tensor                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_scalar                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_scalar_out                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_dim_none_keepdim_shape                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_maximum_reduction_one_valid                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_large_q_is_tiled                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_gather_interpolations                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_noncontiguous_out                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_all_nan_and_path_boundaries                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_special_values_and_scalar_tensor_q                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                         | test_nanquantile_errors                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_ne.py                                                  | test_ne                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_ne.py                                                  | test_ne_scalar                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_select_backward.py                              | test_nested_select_backward                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_select_backward.py                              | test_nested_select_backward_1d_components                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py                        | test_nested_view_from_buffer_copy                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py                        | test_nested_view_from_buffer_copy_oversized_tail                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_zero_boundary                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_nan                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_inf                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_finfo_extremes                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_y                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_x                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_nan_y                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                          | test_nextafter_scalar_inf_y                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nonzero.py                                             | test_nonzero                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_not_equal_.py                                          | test_accuracy_not_equal_                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_not_equal_.py                                          | test_accuracy_not_equal_scalar_                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_correctness                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_padding_value                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_empty_error                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_high_rank_boundary                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_large_batch_correctness                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_invalid_shape                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_scalar_error                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_noncontiguous_boundary                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_mixed_device_error                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_mixed_dtype                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pad_sequence.py                                        | test_pad_sequence_zero_output                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_per_token_group_quant_fp8.py                           | test_per_token_group_quant_fp8                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_partition                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_partition_rejects_nonpositive_tiles                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_partition_rejects_nonpositive_max_grid               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_codegen_config_balance_grid_defaults_off                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_codegen_config_accepts_balance_grid_opt_in                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_partition_covers_each_tile_once                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_generated_wrapper_source_and_runtime_partition       |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_cache_isolation                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                      | test_balanced_grid_real_kernel_covers_non_power_of_two_tail             |      1 |      0 |       0 |       0 |        1 |
| tests/test_prelu.py                                               | test_prelu                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_channels_last                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_empty                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_zero_output_scale                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_bad_size                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_bad_dtype                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_mismatched_qparams                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_resizes                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_storage_offset                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_empty                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_wrong_dtype                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                                | test_quantized_batch_norm_out_wrong_device                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rand.py                                                | test_rand                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_contiguous_view_path                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_copy_path                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_copy_path_ranks                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_complex                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_lazy_conj_bit_kept_on_view_path                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_neg_materialized_value                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_autograd                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_rejects_non_tensor                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_rejects_sparse_layout                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                    | test_reflection_pad3d                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                    | test_reflection_pad3d_list_padding                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                    | test_reflection_pad3d_empty_padding                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                    | test_reflection_pad3d_4d_input                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d.py                                    | test_reflection_pad3d_out                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_renorm_.py                                             | test_renorm_                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_replication_pad3d_backward.py                          | test_replication_pad3d_backward                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_replication_pad3d_backward.py                          | test_replication_pad3d_backward_rejects_mismatched_leading_shape        |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_output.py                                       | test_resize_output                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_forward_modes                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_comprehensive_forward_shapes                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_large_hidden                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_dropout_one                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_data                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh.py                                            | test_rnn_tanh_data_dispatch                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                             | test_digamma                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                             | test_digamma_byte                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                             | test_digamma_int_rejects_inplace                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_tensor                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_rsub.py                                                | test_rsub_scalar                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py           | test_scaled_dot_product_attention_math_for_mps                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_softmax.py                                      | test_scaled_softmax_forward                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_softmax.py                                      | test_scaled_softmax_backward                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_select_scatter.py                                      | test_select_scatter                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_select_scatter.py                                      | test_select_scatter_with_self_overlapping_input                         |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_model_loaded_once_per_version                                      |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_status.py                                     | test_proposer_cache_separates_outer_package_versions                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_ctx_from_first_arg_not_second                |      0 |      0 |       1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py                      | TestInputGuardContext.test_ctx_from_first_kwarg_when_no_tensor_in_args  |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_scale                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py      | test_combine_topk_swa_indices_vllm_accuracy                             |      0 |      0 |       1 |       0 |        1 |
| tests/test_fp8_einsum.py                                          | test_accuracy_fp8_einsum                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_accuracy                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_large_weight_offset                                 |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_core_api                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_invalid_core_limit                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_core_api_unavailable                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_group_boundaries                                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_rejects_invalid_group_list_type                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_matmul.py                                      | test_grouped_matmul_rejects_invalid_metadata                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__rejects_mixed_device_list                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_fused_sgd_.py                                          | test_fused_sgd__launches_on_param_device                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_mhc_ops.py                                             | test_hc_head_fused_kernel_vs_vllm                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                        | test_grouped_topk                                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                        | test_grouped_topk_large_scale                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                        | test_grouped_topk_scaling_factor                                        |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                        | test_grouped_topk_single_token                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                        | test_grouped_topk_sigmoid                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_strided_offset_lazy_bits                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_ravel.py                                               | test_ravel_bfloat16_lazy_neg_copy                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_dropout_neg_degrades_to_zero              |      0 |      1 |       0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                            | test_cudnn_attention_backward_attn_bias_shapes                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                        | test_lu_with_info_exact_zero_pivot_position                             |      0 |      0 |       0 |       1 |        1 |
| TOTAL                                                             |                                                                         |    488 |      3 |      25 |       1 |      517 |
|                       filepath                       |                             function                              | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------- | ----------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_signbit.py                                | test_signbit                                                      |      1 |      0 |       0 |        1 |
| tests/test_signbit.py                                | test_signbit_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_pad_circular.py                           | test_pad_circular                                                 |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss                                               |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_backward                                      |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_backward_scalar_grad_output                   |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_backward_beta_zero_equal_inputs_cuda_behavior |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_broadcast                                     |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_special_values                                |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_out                                           |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_out_reduced                                   |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_functional                                    |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_negative_beta                                 |      1 |      0 |       0 |        1 |
| tests/test_smooth_l1_loss.py                         | test_smooth_l1_loss_backward_negative_beta                        |      1 |      0 |       0 |        1 |
| tests/test_softshrink_backward.py                    | test_softshrink_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                   | test_sparse_sampled_addmm                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                   | test_sparse_sampled_addmm_large                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                   | test_sparse_sampled_addmm_out                                     |      1 |      0 |       0 |        1 |
| tests/test_special_bessel_j1.py                      | test_special_bessel_j1                                            |      1 |      0 |       0 |        1 |
| tests/test_special_entr.py                           | test_special_entr                                                 |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform                                           |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn                                        |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn_scale                                  |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn_leading_dims                           |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p                                                |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p_non_tensor                                     |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p_out                                            |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p_negative                                       |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p_nan_inf                                        |      1 |      0 |       0 |        1 |
| tests/test_special_log1p.py                          | test_special_log1p_small_values                                   |      1 |      0 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py             | test_special_modified_bessel_k1                                   |      1 |      0 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py             | test_special_modified_bessel_k1_out                               |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u                       |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u_scalar_n              |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py | test_special_shifted_chebyshev_polynomial_u_                      |      1 |      0 |       0 |        1 |
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
| tests/test_special_zeta.py                           | test_special_zeta_q_gradient                                      |      1 |      0 |       0 |        1 |
| tests/test_special_zeta.py                           | test_special_zeta_x_gradient_is_not_implemented                   |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                             | test_accuracy_diag_embed                                          |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                             | test_diag_embed_backward                                          |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                             | test_diag_embed_backward_default_dims                             |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                             | test_diag_embed_backward_non_contiguous                           |      1 |      0 |       0 |        1 |
| tests/test_subtract_.py                              | test_subtract                                                     |      1 |      0 |       0 |        1 |
| tests/test_subtract_.py                              | test_subtract_                                                    |      1 |      0 |       0 |        1 |
| tests/test_sym_size.py                               | test_sym_size                                                     |      1 |      0 |       0 |        1 |
| tests/test_tanh.py                                   | test_tanh                                                         |      1 |      0 |       0 |        1 |
| tests/test_tanh.py                                   | test_tanh_                                                        |      1 |      0 |       0 |        1 |
| tests/test_tanh.py                                   | test_tanh_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_rejects_unsupported_dtype      |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward                                |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_optional_grads                 |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_noncontiguous                  |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_end_to_end                     |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_reshaped_workspace             |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_empty                          |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_special_values                 |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py          | test_thnn_fused_lstm_cell_backward_requires_no_grad               |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy_complex                              |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy                                      |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy_scalar                               |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy_non_contiguous                       |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy_same_dim_does_not_alias              |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_transpose_copy_invalid_dims                                  |      1 |      0 |       0 |        1 |
| tests/test_transpose_copy.py                         | test_accuracy_transpose_copy_float8                               |      1 |      0 |       0 |        1 |
| tests/test_triton_lang_helper.py                     | test_patch_missing_nearbyint_uses_rint                            |      1 |      0 |       0 |        1 |
| tests/test_greater_.py                               | test_greater_                                                     |      1 |      0 |       0 |        1 |
| tests/test_greater_.py                               | test_greater_fp64_precision                                       |      1 |      0 |       0 |        1 |
| tests/test_unique_consecutive.py                     | test_accuracy_unique_consecutive                                  |      1 |      0 |       0 |        1 |
| tests/test_unsafe_split_with_sizes.py                | test_unsafe_split_with_sizes                                      |      1 |      0 |       0 |        1 |
| tests/test_unsafe_split_with_sizes.py                | test_unsafe_split_with_sizes_different_dims                       |      1 |      0 |       0 |        1 |
| tests/test_unsafe_split_with_sizes.py                | test_unsafe_split_with_sizes_edge_cases                           |      1 |      0 |       0 |        1 |
| tests/test_unsafe_split_with_sizes.py                | test_unsafe_split_with_sizes_all_zero_splits                      |      1 |      0 |       0 |        1 |
| tests/test_unsafe_split_with_sizes.py                | test_unsafe_split_with_sizes_single_element_splits                |      1 |      0 |       0 |        1 |
| tests/test_upsample_bilinear2d_aa_backward.py        | test_upsample_bilinear2d_aa_backward                              |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss                                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_target_patterns                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_default_reduction                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_extra_reduction_values                |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_out                                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_noncontiguous                         |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_backward                              |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_backward_duplicate_targets            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_backward_strided_grad_out             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_backward_large_class_count            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_empty_batch                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_class_count                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_class_split_path                      |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_special_values                        |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_out_of_range_target                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_out_dtype                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_out_invalid                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_invalid_input_dtype                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_invalid_target_dtype                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_invalid_target_shape                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_invalid_reduction                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                 | test_multilabel_margin_loss_invalid_rank                          |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact3d.py               | test_upsample_nearest_exact3d                                     |      1 |      0 |       0 |        1 |
| tests/test_w8a8_block_fp8_matmul.py                  | test_w8a8_block_fp8_matmul                                        |      1 |      0 |       0 |        1 |
| tests/test_view_copy.py                              | test_view_copy                                                    |      1 |      0 |       0 |        1 |
| tests/test_weight_norm_interface.py                  | test_weight_norm_interface                                        |      1 |      0 |       0 |        1 |
| tests/test_weight_norm_interface.py                  | test_weight_norm_interface_backward                               |      1 |      0 |       0 |        1 |
| tests/test_zero.py                                   | test_zero                                                         |      1 |      0 |       0 |        1 |
| tests/test_zero.py                                   | test_zero_                                                        |      1 |      0 |       0 |        1 |
| tests/test_zero.py                                   | test_zero_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_acc_false                                          |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_acc_true                                           |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__acc_false                                         |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__acc_true                                          |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_error_all_none                                     |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__error_all_none                                    |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put_mixed_none_and_tensor                              |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__large_strided_offset                              |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__bf16_strided_boundary                             |      1 |      0 |       0 |        1 |
| tests/test_index_put.py                              | test_index_put__single_element_index                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve                                             |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_noncontiguous                               |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_ignored_entries                             |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_singular_and_near_singular                  |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_empty                                       |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out                                         |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_nonempty_resize                         |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_empty                                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_aliases_inputs                          |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_invalid_shapes                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_invalid_shapes                          |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_rejected_dtype                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_rejected_dtype                          |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_mismatched_inputs                           |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_invalid_outputs                         |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_rejects_expanded_output                 |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_rejects_autograd                        |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_autograd_broadcast                          |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_scipy_reference                             |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_wide_rhs                                    |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_large_ignored_entries                       |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_out_large_ignored_entries                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_float64_precision                           |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_finite_inverse_overflow                     |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_chunked_previous_rows                       |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_panel_wide_rhs                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_float64_offdiagonal_precision               |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean                                                         |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_empty                                                   |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_empty_explicit_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_empty_rejects_non_floating_input                        |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_empty_rejects_non_floating_dtype                        |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_dim                                                     |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_dim_empty                                               |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_dim_empty_explicit_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_dim_large_k                                             |      1 |      0 |       0 |        1 |
| tests/test_mean.py                                   | test_mean_dim_large_innerdim                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_operator_markers_are_declared                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_float                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_int                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_bool                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_complex                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_rejects_unsupported_dtype             |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_list_lengths                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_mixed_shapes                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_empty_and_zero_dim                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_mixed_dtypes                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_layouts                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_inplace                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_inplace_writes_through_views          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_inplace_rejects_overlap               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_inplace_rejects_promotion             |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_error_parity                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_out_overload                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_foreach_unary_registration_is_live                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_foreach_unary_launch_count_is_independent_of_list_length     |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_foreach_unary_launch_count_scales_with_dtype_groups          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_foreach_unary_launch_count_with_heterogeneous_numel          |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_unary_complex_large                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                          | test_accuracy_foreach_abs_complex_large                           |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max                                                    |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_                                                   |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_low_precision                                      |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_uint8_int_scalar                                   |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_int8_overflow_scalar                               |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_uint8_overflow_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_clamp_max.py                              | test_clamp_max_uint8_boundary_scalar                              |      1 |      0 |       0 |        1 |
| tests/test_outplace_fused_experts.py                 | test_outplace_fused_experts_accuracy                              |      1 |      0 |       0 |        1 |
| tests/test_outplace_fused_experts.py                 | test_outplace_fused_experts_does_not_modify_input                 |      1 |      0 |       0 |        1 |
| tests/test_silu_and_mul.py                           | test_silu_and_mul                                                 |      1 |      0 |       0 |        1 |
| tests/test_silu_and_mul.py                           | test_silu_and_mul_out                                             |      1 |      0 |       0 |        1 |
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
| tests/test_shape_utils.py                            | test_c_contiguous_stride_normal                                   |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_c_contiguous_stride_with_zero_size                           |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_f_contiguous_stride_normal                                   |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_f_contiguous_stride_with_zero_size                           |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_ordered_stride_normal                                        |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_ordered_stride_with_zero_size                                |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_stride_order                                                 |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_shape_empty                                     |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_shape1                                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_shape2                                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_stride_empty                                    |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_stride1                                         |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_the_same_stride2                                         |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_c_contiguous_empty                                       |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_all_c_contiguous1                                            |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_heuristics_for_tile_size                                     |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_heuristics_for_num_warps                                     |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_contiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_transposed_is_not_viewable                    |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_keeps_transposed_layout                       |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_unit_dims_any_stride                          |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_scalar_and_empty                              |      1 |      0 |       0 |        1 |
| tests/test_shape_utils.py                            | test_compute_stride_matches_torch_view                            |      1 |      0 |       0 |        1 |
| tests/test_sspaddmm.py                               | test_sspaddmm                                                     |      1 |      0 |       0 |        1 |
| tests/test_sspaddmm.py                               | test_sspaddmm_out                                                 |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp                                                |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out                                            |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_large_dynamic_range                            |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_mixed_infinities                               |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_nan_propagation                                |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_infinite_inputs                                |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_wide_dynamic_range_rows                        |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_fp64_precision                                 |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_complex                                        |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_complex_special_values                         |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_scalar                                         |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_scalar_invalid_dim                             |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_invalid_dim                                    |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_empty                                          |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_empty                                      |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_resizes                                    |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_noncontiguous                              |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_rejects_mismatched_dtype                   |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_rejects_mismatched_device                  |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_out_invalid_dim_does_not_touch_out             |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_noncontiguous_input                            |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_all_dims                                       |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_long_rows                                      |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_long_rows_special_values                       |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_long_row_fp64                                  |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_long_complex_rows                              |      1 |      0 |       0 |        1 |
| tests/test__logcumsumexp.py                          | test__logcumsumexp_long_complex_rows_special_values               |      1 |      0 |       0 |        1 |
| tests/test_randn.py                                  | test_randn                                                        |      1 |      0 |       0 |        1 |
| tests/test_randn.py                                  | test_randn_large_output_offsets                                   |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner                                                        |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_1d_dot                                                 |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_scalar                                                 |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_matvec                                                 |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_empty_contraction                                      |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_2d                                                     |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_dtype_mismatch                                         |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                  | test_inner_contraction_mismatch                                   |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin                                                       |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin_                                                      |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin_out_integer_rejects                                   |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin_boundaries_and_out_of_domain                          |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                 | test_arcsin_empty_and_scalar                                      |      1 |      0 |       0 |        1 |
| tests/test_mrope.py                                  | test_mrope_vs_torch                                               |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_accuracy                                  |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_zero_infinity                             |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_tensor_overload                           |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_out                                       |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_out_resize                                |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__basic                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__options                                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__grad_scale                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__found_inf                                     |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__empty                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__mixed_empty_and_nonempty                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__mismatched_list_lengths                       |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__mismatched_tensor_properties                  |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__later_entry_bad_dtype_or_device               |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__later_entry_bad_state_step                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__multi_tensor                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__transposed_parity                             |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                         | test_fused_adagrad__internal_overlap_rejected                     |      1 |      0 |       0 |        1 |
| tests/test_lstm.py                                   | test_lstm                                                         |      1 |      0 |       0 |        1 |
| tests/test_less_.py                                  | test_less_                                                        |      1 |      0 |       0 |        1 |
| tests/test_less_.py                                  | test_less_scalar_                                                 |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                 | test_wrapped_linear_prepack                                       |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                 | test_wrapped_linear_prepack_rounds_ties_to_even_and_clamps        |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py                 | test_wrapped_linear_prepack_rejects_invalid_shapes                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_unsupported_shell_input_dtype                           |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_finite_extremes                                         |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_window_promotes_low_precision_input                     |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft                                                         |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_center                                                  |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_real_output_and_complex_window_promotion                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_gradients                                               |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_padding_and_promotion_gradients                         |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_empty_signal_gradients                                  |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_lazy_views                                              |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_chunked_forward_and_backward                            |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_nonfinite                                               |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_nonfinite_window_support                                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_empty_batch_or_signal_follows_native                    |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_invalid_inputs                                          |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_center_rejects_explicit_align                           |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_default_align_odd_window_native_boundary                |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_fused_complex_window_gradients                          |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_center_fused_complex_window_gradients                   |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_empty_batch_gradients                                   |      1 |      0 |       0 |        1 |
| tests/test_stft.py                                   | test_stft_center_empty_batch_gradients                            |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_dispatch_fused_moe_kernel_matches_ref                        |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_vs_ref                                             |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_int8                                               |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_int8_w8a16                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_int4_w4a16                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_inplace                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_apply_router_weight_on_input                       |      1 |      0 |       0 |        1 |
| tests/test_topk_softmax.py                           | test_topk_softmax                                                 |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                       | test_triangular_solve_rejected_float64                            |      0 |      0 |       1 |        1 |
| tests/test_fused_indexer_q_rope_quant.py             | test_fused_indexer_q_rope_quant                                   |      0 |      0 |       1 |        1 |
| tests/test_stft.py                                   | test_stft_ascend_complex_half_promotion                           |      0 |      0 |       1 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_vs_vllm                                            |      0 |      0 |       1 |        1 |
| tests/test_fused_experts_impl.py                     | test_accuracy_fused_moe_fp8                                       |      0 |      0 |       1 |        1 |
| tests/test_fused_experts_impl.py                     | test_fused_moe_fp8_blockwise                                      |      0 |      0 |       1 |        1 |
| tests/test_stft.py                                   | test_stft_torch_dispatch                                          |      0 |      1 |       0 |        1 |
| TOTAL                                                |                                                                   |    341 |      1 |       7 |      349 |
