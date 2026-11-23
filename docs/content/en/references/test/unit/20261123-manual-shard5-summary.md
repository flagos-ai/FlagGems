|                         filepath                         |                                function                                 | passed | skipped | SUBTOTAL |
| -------------------------------------------------------- | ----------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/core/test_reference_only.py                        | test_benchmark_runs_original_baseline_once                              |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_backward_reference_uses_original_grad_semantics                    |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_skip_native_is_not_a_pass_or_failure                               |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_failure_is_not_passed                                    |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_capability_failures_are_recorded_individually_and_later_cases_run  |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_failure_categories_do_not_guess_api_or_dtype_support               |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_capability_failure_does_not_continue_after_sync_failure            |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_interrupt_propagates_and_leaves_unexecuted_cases         |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_capability_failure_before_invocation_keeps_zero_call_count         |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_source_skip_preserves_node_semantics_and_unexecuted_records        |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_unsupported_custom_runner_cannot_execute_candidate                 |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_conflicting_modes_are_rejected                                     |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_report_replaces_stale_data_and_preserves_case_skips      |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_unknown_case_selection_fails                             |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness   |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_mode_cannot_run_correctness_pytest                       |      1 |       0 |        1 |
| tests/core/test_reference_only.py                        | test_reference_only_defaults_to_core_without_changing_normal_defaults   |      1 |       0 |        1 |
| tests/flagtune/test_model_source.py                      | ModelSourceTests.test_defaults_and_once_only                            |      1 |       0 |        1 |
| tests/flagtune/test_model_source.py                      | ModelSourceTests.test_explicit_sources_including_empty_are_preserved    |      1 |       0 |        1 |
| tests/flagtune/test_model_source.py                      | ModelSourceTests.test_manifest_schema_and_wheel_data                    |      1 |       0 |        1 |
| tests/flagtune/test_model_source.py                      | ModelSourceTests.test_version_preferences_preserved                     |      1 |       0 |        1 |
| tests/flagtune/test_status.py                            | test_status_uses_stderr_and_flush                                       |      1 |       0 |        1 |
| tests/flagtune/test_status.py                            | test_exception_chain_redacts_url_secrets                                |      1 |       0 |        1 |
| tests/flagtune/test_status.py                            | test_status_closed_stream_does_not_fail                                 |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardWraps.test_name_preserved                                 |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardWraps.test_doc_preserved                                  |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardWraps.test_wrapped_attribute_points_to_original           |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_single_non_contiguous_arg                 |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_multiple_non_contiguous_args              |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_already_contiguous_arg_stays_contiguous   |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_single_non_contiguous_kwarg               |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_multiple_non_contiguous_kwargs            |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_mixed_args_and_kwargs_all_contiguous      |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_non_tensor_arg_passed_through_unchanged   |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_non_tensor_kwarg_passed_through_unchanged |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_contiguous_version_has_same_data          |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContiguous.test_return_value_preserved                    |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_output_on_correct_device                     |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_all_available_devices                        |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_current_device_matches_input                 |      1 |       0 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_nullcontext_when_no_tensor                   |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_                                                    |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_default_params                                      |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_inplace_semantics                                   |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_value_range                                         |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_batched_key                                         |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_stream_continuity                                   |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                           | test_philox_uniform_large_offset                                        |      1 |       0 |        1 |
| tests/test_abs.py                                        | test_abs                                                                |      1 |       0 |        1 |
| tests/test_abs.py                                        | test_abs_                                                               |      1 |       0 |        1 |
| tests/test_adaptive_avg_pool3d_backward.py               | test_adaptive_avg_pool3d_backward_grad_input                            |      1 |       0 |        1 |
| tests/test_addcdiv.py                                    | test_addcdiv                                                            |      1 |       0 |        1 |
| tests/test_addcdiv.py                                    | test_addcdiv_out                                                        |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args                                    |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_invalid_indices                    |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_invalid_metadata                   |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_dispatch                           |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_zero_compressed_axis               |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_zero_axis_metadata                 |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_zero_axis_invalid_indices          |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_empty_plain_stride                 |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_zero_batch_compressed_stride       |      1 |       0 |        1 |
| tests/test_validate_sparse_bsc_tensor_args.py            | test_validate_sparse_bsc_tensor_args_large_batch                        |      1 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py         | test_validate_compressed_sparse_indices                                 |      1 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py         | test_validate_compressed_sparse_indices_invalid_indices                 |      1 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py         | test_validate_compressed_sparse_indices_large_batch                     |      1 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py         | test_validate_compressed_sparse_indices_invalid_metadata                |      1 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py         | test_validate_compressed_sparse_indices_dispatch                        |      1 |       0 |        1 |
| tests/test_arctan_.py                                    | test_arctan                                                             |      1 |       0 |        1 |
| tests/test_arctan_.py                                    | test_arctan_                                                            |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_asinh                                                              |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_asinh_various_sizes                                                |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_asinh_edge_cases                                                   |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_asinh_empty_tensor                                                 |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_accuracy_asinh_out                                                 |      1 |       0 |        1 |
| tests/test_asinh.py                                      | test_asinh_                                                             |      1 |       0 |        1 |
| tests/test_avg_pool1d.py                                 | test_avg_pool1d                                                         |      1 |       0 |        1 |
| tests/test_avg_pool1d.py                                 | test_avg_pool1d_default_stride                                          |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_2d                                                |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_2d_dilation                                       |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_1d                                                |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_3d                                                |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_invalid_padding                                   |      1 |       0 |        1 |
| tests/test_convolution_mode.py                           | test_convolution_mode_same_strided_raises                               |      1 |       0 |        1 |
| tests/test_bernoulli.py                                  | test_bernoulli_                                                         |      1 |       0 |        1 |
| tests/test_bernoulli.py                                  | test_bernoulli_various_p                                                |      1 |       0 |        1 |
| tests/test_bernoulli.py                                  | test_bernoulli                                                          |      1 |       0 |        1 |
| tests/test_bitwise_and_scalar_tensor.py                  | test_bitwise_and_scalar_tensor                                          |      1 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py       | test_native_batch_norm_legit_no_training                                |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d                                                |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_projection                                     |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_out                                            |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_dispatch                                       |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_wide_tile_selection                            |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_unbatched                                      |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_negative_padding                               |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_non_square_geometry                            |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_strided_input                                  |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_channels_last                                  |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_empty                                          |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_symint_pair                                    |      1 |       0 |        1 |
| tests/test_slow_conv_dilated2d.py                        | test_slow_conv_dilated2d_error_message                                  |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                    | test_nested_view_from_buffer                                            |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                    | test_nested_view_from_buffer_aliasing                                   |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                    | test_nested_view_from_buffer_3d_components                              |      1 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                   | test_sparse_bsr_tensor_unsafe                                           |      1 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                   | test_sparse_bsr_tensor_unsafe_dtype                                     |      1 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                   | test_sparse_bsr_tensor_unsafe_skips_index_validation                    |      1 |       0 |        1 |
| tests/test_addbmm.py                                     | test_accuracy_addbmm                                                    |      1 |       0 |        1 |
| tests/test_conv_tbc_backward.py                          | test_conv_tbc_backward                                                  |      1 |       0 |        1 |
| tests/test_cos.py                                        | test_cos                                                                |      1 |       0 |        1 |
| tests/test_cos.py                                        | test_cos_                                                               |      1 |       0 |        1 |
| tests/test_elu_backward_out.py                           | test_elu_backward_out                                                   |      1 |       0 |        1 |
| tests/test_elu_backward_out.py                           | test_elu_backward_out_noncontiguous                                     |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices                                                       |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_bsr                                                   |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_batched                                               |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_int32_buffer                                          |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_alias_and_mutation                                    |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_dispatch_stability                                    |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_dispatch_sentinel                                     |      1 |       0 |        1 |
| tests/test_crow_indices.py                               | test_crow_indices_unsupported_layout_raises                             |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod                                                            |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_dtype                                                      |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_low_precision_multitile                                    |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace                                                    |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_long_row_int16                                     |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_low_precision_multitile                            |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_non_contiguous                                     |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_non_contiguous_tile_boundary                       |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_dtype_mismatch                                     |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_inplace_bool_unsupported                                   |      1 |       0 |        1 |
| tests/test_cumprod.py                                    | test_cumprod_backward                                                   |      1 |       0 |        1 |
| tests/test_cast_Long.py                                  | test_cast_Long                                                          |      1 |       0 |        1 |
| tests/test_cast_Long.py                                  | test_cast_Long_strided                                                  |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward                                        |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out_of_bound                           |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_edge_coordinates                       |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_all_out_of_bound                       |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_accumulation                           |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_non_finite_grid                        |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_non_finite_grad_output                 |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_float64                                |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_non_contiguous                         |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_empty                                  |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_invalid_inputs                         |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out                                    |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out_resize                             |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out_resize_warns                       |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out_strided                            |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_out_dtype_mismatch                     |      1 |       0 |        1 |
| tests/test_cudnn_grid_sampler_backward.py                | test_cudnn_grid_sampler_backward_dispatch                               |      1 |       0 |        1 |
| tests/test_dimV.py                                       | test_accuracy_dimV_coo                                                  |      1 |       0 |        1 |
| tests/test_dimV.py                                       | test_accuracy_dimV_matches_indices_formula                              |      1 |       0 |        1 |
| tests/test_dimV.py                                       | test_accuracy_dimV_unsupported_layouts_raise                            |      1 |       0 |        1 |
| tests/test_dimV.py                                       | test_accuracy_dimV_dispatch_stability                                   |      1 |       0 |        1 |
| tests/test_dropout_.py                                   | test_dropout_                                                           |      1 |       0 |        1 |
| tests/test_dropout_.py                                   | test_dropout__noop                                                      |      1 |       0 |        1 |
| tests/test_embedding.py                                  | test_embedding                                                          |      1 |       0 |        1 |
| tests/test_embedding.py                                  | test_embedding_backward                                                 |      1 |       0 |        1 |
| tests/test_embedding.py                                  | test_embedding_dense_backward                                           |      1 |       0 |        1 |
| tests/test_empty.py                                      | test_empty_permuted                                                     |      1 |       0 |        1 |
| tests/test_empty.py                                      | test_empty                                                              |      1 |       0 |        1 |
| tests/test_empty.py                                      | test_empty_default_dtype                                                |      1 |       0 |        1 |
| tests/test_expand.py                                     | test_expand                                                             |      1 |       0 |        1 |
| tests/test_expand.py                                     | test_expand_                                                            |      1 |       0 |        1 |
| tests/test_expand.py                                     | test_expand_invalid_sizes                                               |      1 |       0 |        1 |
| tests/test_expand.py                                     | test_expand_zero_size_singleton_stride                                  |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense                                |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_special_values                 |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_mask_layout                    |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_offset_views                   |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_unsupported_dtype              |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_invalid_input_rank             |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_invalid_input_layout           |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test__sparse_semi_structured_apply_dense_invalid_mask                   |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply_dense.py        | test_sparse_semi_structured_apply_dense_dispatch                        |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_accuracy_fake_quantize_per_tensor_affine                           |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_accuracy_fake_quantize_per_tensor_affine_tensor_qparams            |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_accuracy_fake_quantize_per_tensor_affine_half_to_even              |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_accuracy_fake_quantize_per_tensor_affine_noncontiguous             |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_accuracy_fake_quantize_per_tensor_affine_empty                     |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py            | test_fake_quantize_per_tensor_affine_invalid_qparams                    |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                              | test_fft_ifftshift                                                      |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                              | test_fft_ifftshift_int_dim                                              |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                              | test_fft_ifftshift_empty                                                |      1 |       0 |        1 |
| tests/test_fft_ifftshift.py                              | test_fft_ifftshift_scalar                                               |      1 |       0 |        1 |
| tests/test_fused_add_rms_norm.py                         | test_fused_add_rms_norm                                                 |      1 |       0 |        1 |
| tests/test_float_power_.py                               | test_float_power_                                                       |      1 |       0 |        1 |
| tests/test_float_power_.py                               | test_float_power_scalar                                                 |      1 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                      | test_fractional_max_pool2d                                              |      1 |       0 |        1 |
| tests/test_fractional_max_pool2d.py                      | test_fractional_max_pool2d_backward                                     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward                                               |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_shapes                                        |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_dim_forms                                     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_dim_forms_large                               |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_zero_nnz_and_zero_axis                        |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_noncontiguous                                 |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_stored_offsets                                |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_out                                           |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_uncoalesced_self                              |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_duplicate_coordinates                         |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_special_values                                |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_copy_special_values                           |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_regather_rejects_non_float_dtypes             |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_negative_dense_self                           |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_negative_out_of_range_dim                     |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_negative_aliased_dim                          |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test__sparse_sum_backward_negative_sparse_grad_for_summed_dims          |      1 |       0 |        1 |
| tests/test__sparse_sum_backward.py                       | test_sparse_sum_backward_dispatch                                       |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve                                                           |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_batch                                                     |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_broadcast                                                 |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_out                                                       |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_non_contiguous                                            |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_out_non_contiguous                                        |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_dtype_mismatch                                            |      1 |       0 |        1 |
| tests/test_lu_solve.py                                   | test_lu_solve_pivots_dtype                                              |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py | test_fake_quantize_learnable_per_channel_affine                         |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py | test_fake_quantize_learnable_per_channel_affine_grad_factor             |      1 |       0 |        1 |
| tests/test_atanh_backward.py                             | test_atanh_backward                                                     |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double                                                        |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_strided                                                |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_special_values                                         |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_integer_limits                                         |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_autograd                                               |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_complex                                                |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_negative_view                                          |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_cpu                                                    |      1 |       0 |        1 |
| tests/test_cast_Double.py                                | test_cast_Double_sparse                                                 |      1 |       0 |        1 |
| tests/test_grid_sampler_3d_backward.py                   | test_grid_sampler_3d_backward                                           |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal                                                      |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal_scalar                                               |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal_scalar_byte                                          |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal_                                                     |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal__byte                                                |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal__byte_scalar                                         |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal_scalar_                                              |      1 |       0 |        1 |
| tests/test_greater_equal.py                              | test_greater_equal_scalar__byte                                         |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh                                                           |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh_explicit                                                  |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh_out                                                       |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh_out_explicit                                              |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh_special_values                                            |      1 |       0 |        1 |
| tests/test_hardtanh.py                                   | test_hardtanh_out_special_values                                        |      1 |       0 |        1 |
| tests/test_hinge_embedding_loss.py                       | test_hinge_embedding_loss                                               |      1 |       0 |        1 |
| tests/test_huber_loss.py                                 | test_huber_loss                                                         |      1 |       0 |        1 |
| tests/test_huber_loss.py                                 | test_huber_loss_out                                                     |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_scalar                                                  |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_scalar_                                                 |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_tensor_value                                            |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_duplicate_index                                         |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_large_row_fill                                          |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_grid_stride                                             |      1 |       0 |        1 |
| tests/test_index_fill.py                                 | test_index_fill_noncontiguous                                           |      1 |       0 |        1 |
| tests/test_index_select.py                               | test_index_select                                                       |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial                                                           |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial_large_count                                               |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial_varying_prob                                              |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial_edge_probs                                                |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial_zero_count                                                |      1 |       0 |        1 |
| tests/test_binomial.py                                   | test_binomial_out                                                       |      1 |       0 |        1 |
| tests/test_is_nonzero.py                                 | test_is_nonzero                                                         |      1 |       0 |        1 |
| tests/test_is_nonzero.py                                 | test_is_nonzero_exception                                               |      1 |       0 |        1 |
| tests/test_isposinf.py                                   | test_isposinf                                                           |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss                                                            |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss_int_reduction                                              |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss_broadcast                                                  |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss_empty                                                      |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss_functional                                                 |      1 |       0 |        1 |
| tests/test_l1_loss.py                                    | test_l1_loss_invalid_reduction                                          |      1 |       0 |        1 |
| tests/test_less_.py                                      | test_less_                                                              |      1 |       0 |        1 |
| tests/test_less_.py                                      | test_less_scalar_                                                       |      1 |       0 |        1 |
| tests/test_linalg_householder_product.py                 | test_linalg_householder_product                                         |      1 |       0 |        1 |
| tests/test_linalg_householder_product.py                 | test_linalg_householder_product_batched                                 |      1 |       0 |        1 |
| tests/test_linalg_lu_solve.py                            | test_linalg_lu_solve                                                    |      1 |       0 |        1 |
| tests/test_linalg_lu_solve.py                            | test_linalg_lu_solve_batched                                            |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_common                                                             |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_large                                                              |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_out_parameter                                                      |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_linalg_matrix_power_out                                            |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_out_missing_raises                                                 |      1 |       0 |        1 |
| tests/test_linalg_matrix_power.py                        | test_invalid_shape_rejected                                             |      1 |       0 |        1 |
| tests/test_linalg_slogdet.py                             | test_linalg_slogdet                                                     |      1 |       0 |        1 |
| tests/test_linalg_vander.py                              | test_linalg_vander                                                      |      1 |       0 |        1 |
| tests/test_linalg_vander.py                              | test_linalg_vander_with_N                                               |      1 |       0 |        1 |
| tests/test_linalg_vander.py                              | test_linalg_vander_complex                                              |      1 |       0 |        1 |
| tests/test_linalg_vander.py                              | test_linalg_vander_with_N_complex                                       |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_                                                               |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_large_inplace                                                  |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_special_values                                                 |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_noncontiguous                                                  |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_empty                                                          |      1 |       0 |        1 |
| tests/test_log_.py                                       | test_log_unsupported_dtype_raises                                       |      1 |       0 |        1 |
| tests/test_logaddexp2.py                                 | test_logaddexp2                                                         |      1 |       0 |        1 |
| tests/test_logaddexp2.py                                 | test_logaddexp2_out                                                     |      1 |       0 |        1 |
| tests/test_logaddexp2.py                                 | test_logaddexp2_non_finite                                              |      1 |       0 |        1 |
| tests/test_logit.py                                      | test_logit                                                              |      1 |       0 |        1 |
| tests/test_logit.py                                      | test_logit_                                                             |      1 |       0 |        1 |
| tests/test_logit.py                                      | test_logit_out                                                          |      1 |       0 |        1 |
| tests/test_logit.py                                      | test_logit_out_non_contiguous_out                                       |      1 |       0 |        1 |
| tests/test_linalg_lu.py                                  | test_linalg_lu                                                          |      1 |       0 |        1 |
| tests/test_linalg_lu.py                                  | test_linalg_lu_out                                                      |      1 |       0 |        1 |
| tests/test_masked_scale.py                               | test_masked_scale                                                       |      1 |       0 |        1 |
| tests/test_matmul_bias_activation.py                     | test_matmul_bias_activation                                             |      1 |       0 |        1 |
| tests/test_max_pool2d.py                                 | test_max_pool2d_with_indices                                            |      1 |       0 |        1 |
| tests/test_max_pool2d.py                                 | test_max_pool2d_backward                                                |      1 |       0 |        1 |
| tests/test_max_pool2d.py                                 | test_max_pool2d_with_indices_backward                                   |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim                                                         |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_keepdim_shapes                                          |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_ties                                                    |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_strided_nonlast_reduction                               |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_even_count_picks_lower_middle                           |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_propagates_nan                                          |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_scalar_input                                            |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_empty_reduction_dim_raises                              |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_empty_but_nonzero_reduction                             |      1 |       0 |        1 |
| tests/test_median_dim.py                                 | test_median_dim_named_dim_preserves_names                               |      1 |       0 |        1 |
| tests/test_mish.py                                       | test_mish                                                               |      1 |       0 |        1 |
| tests/test_mish.py                                       | test_mish_                                                              |      1 |       0 |        1 |
| tests/test_mish.py                                       | test_mish_backward                                                      |      1 |       0 |        1 |
| tests/test_moe_sum.py                                    | test_moe_sum                                                            |      1 |       0 |        1 |
| tests/test_dreglu.py                                     | test_dreglu                                                             |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_tensor                                                 |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_scalar                                                 |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_out                                                    |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_scalar_out                                             |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_dim_none_keepdim_shape                                 |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_maximum_reduction_one_valid                            |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_large_q_is_tiled                                       |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_gather_interpolations                                  |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_noncontiguous_out                                      |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_all_nan_and_path_boundaries                            |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_special_values_and_scalar_tensor_q                     |      1 |       0 |        1 |
| tests/test_nanquantile.py                                | test_nanquantile_errors                                                 |      1 |       0 |        1 |
| tests/test_ne.py                                         | test_ne                                                                 |      1 |       0 |        1 |
| tests/test_ne.py                                         | test_ne_scalar                                                          |      1 |       0 |        1 |
| tests/test_nested_select_backward.py                     | test_nested_select_backward                                             |      1 |       0 |        1 |
| tests/test_nested_select_backward.py                     | test_nested_select_backward_1d_components                               |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py               | test_nested_view_from_buffer_copy                                       |      1 |       0 |        1 |
| tests/test_nested_view_from_buffer_copy.py               | test_nested_view_from_buffer_copy_oversized_tail                        |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_                                                         |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_zero_boundary                                            |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_nan                                                      |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_inf                                                      |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_finfo_extremes                                           |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_scalar_y                                                 |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_scalar_x                                                 |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_scalar_nan_y                                             |      1 |       0 |        1 |
| tests/test_nextafter_.py                                 | test_nextafter_scalar_inf_y                                             |      1 |       0 |        1 |
| tests/test_nonzero.py                                    | test_nonzero                                                            |      1 |       0 |        1 |
| tests/test_not_equal_.py                                 | test_accuracy_not_equal_                                                |      1 |       0 |        1 |
| tests/test_not_equal_.py                                 | test_accuracy_not_equal_scalar_                                         |      1 |       0 |        1 |
| tests/test_outer.py                                      | test_outer                                                              |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_correctness                                           |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_padding_value                                         |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_empty_error                                           |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_high_rank_boundary                                    |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_large_batch_correctness                               |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_invalid_shape                                         |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_scalar_error                                          |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_noncontiguous_boundary                                |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_mixed_device_error                                    |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_mixed_dtype                                           |      1 |       0 |        1 |
| tests/test_pad_sequence.py                               | test_pad_sequence_zero_output                                           |      1 |       0 |        1 |
| tests/test_per_token_group_quant_fp8.py                  | test_per_token_group_quant_fp8                                          |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_partition                                            |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_partition_rejects_nonpositive_tiles                  |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_partition_rejects_nonpositive_max_grid               |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_codegen_config_balance_grid_defaults_off                           |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_codegen_config_accepts_balance_grid_opt_in                         |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_partition_covers_each_tile_once                      |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_generated_wrapper_source_and_runtime_partition       |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_cache_isolation                                      |      1 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py             | test_balanced_grid_real_kernel_covers_non_power_of_two_tail             |      1 |       0 |        1 |
| tests/test_prelu.py                                      | test_prelu                                                              |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm                                               |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_channels_last                                 |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_empty                                         |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_zero_output_scale                             |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_bad_size                                      |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_bad_dtype                                     |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out                                           |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_mismatched_qparams                        |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_resizes                                   |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_storage_offset                            |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_empty                                     |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_wrong_dtype                               |      1 |       0 |        1 |
| tests/test_quantized_batch_norm.py                       | test_quantized_batch_norm_out_wrong_device                              |      1 |       0 |        1 |
| tests/test_rand_like.py                                  | test_rand_like                                                          |      1 |       0 |        1 |
| tests/test_reglu.py                                      | test_reglu                                                              |      1 |       0 |        1 |
| tests/test_repeat_interleave.py                          | test_repeat_interleave_self_int                                         |      1 |       0 |        1 |
| tests/test_repeat_interleave.py                          | test_repeat_interleave_self_int_non_contiguous                          |      1 |       0 |        1 |
| tests/test_repeat_interleave.py                          | test_repeat_interleave_tensor                                           |      1 |       0 |        1 |
| tests/test_repeat_interleave.py                          | test_repeat_interleave_self_tensor                                      |      1 |       0 |        1 |
| tests/test_reshape_and_cache.py                          | test_reshape_and_cache                                                  |      1 |       0 |        1 |
| tests/test_resolve_neg.py                                | test_accuracy_resolve_neg                                               |      1 |       0 |        1 |
| tests/test_roll.py                                       | test_roll_single_dim                                                    |      1 |       0 |        1 |
| tests/test_roll.py                                       | test_roll_multi_dims                                                    |      1 |       0 |        1 |
| tests/test_roll.py                                       | test_roll_flatten                                                       |      1 |       0 |        1 |
| tests/test_roll.py                                       | test_roll_with_non_dense_input                                          |      1 |       0 |        1 |
| tests/test_rrelu_.py                                     | test_rrelu_                                                             |      1 |       0 |        1 |
| tests/test_rrelu_.py                                     | test_rrelu__custom_bounds                                               |      1 |       0 |        1 |
| tests/test_rrelu_.py                                     | test_rrelu__inplace                                                     |      1 |       0 |        1 |
| tests/test_rrelu_.py                                     | test_rrelu__training                                                    |      1 |       0 |        1 |
| tests/test_rrelu_.py                                     | test_rrelu__non_contiguous                                              |      1 |       0 |        1 |
| tests/test_rwkv_mm_sparsity.py                           | test_rwkv_mmsparsity                                                    |      1 |       0 |        1 |
| tests/test_scaled_dot_product_efficient_attention.py     | test_scaled_dot_product_efficient_attention                             |      1 |       0 |        1 |
| tests/test_scaled_dot_product_efficient_attention.py     | test_scaled_dot_product_efficient_attention_causal                      |      1 |       0 |        1 |
| tests/test_scaled_dot_product_efficient_attention.py     | test_scaled_dot_product_efficient_attention_logsumexp                   |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_prod_launch_boundary                                |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce                                                     |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_                                                    |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out                                                 |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_high_dim                                            |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce__high_dim                                           |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out_high_dim                                        |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_high_dim_active_prefix                              |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce__high_dim_active_prefix                             |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out_high_dim_active_prefix                          |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_5d_canonical_gate                                   |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_empty                                               |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce__empty                                              |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out_empty                                           |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_noncontiguous                                       |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_high_contention                                     |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_nan                                                 |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_prod_nan                                            |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_invalid_reduce                                      |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_invalid_dim                                         |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out_dtype_mismatch                                  |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_prod_partial_blocks                                 |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_inplace_prod_partial_blocks                         |      1 |       0 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_out_prod_partial_blocks                             |      1 |       0 |        1 |
| tests/test_set_.py                                       | test_set_source_tensor                                                  |      1 |       0 |        1 |
| tests/test_set_.py                                       | test_set_source_tensor_storage_offset                                   |      1 |       0 |        1 |
| tests/test_set_.py                                       | test_set_default                                                        |      1 |       0 |        1 |
| tests/flagtune/test_status.py                            | test_model_loaded_once_per_version                                      |      0 |       1 |        1 |
| tests/flagtune/test_status.py                            | test_proposer_cache_separates_outer_package_versions                    |      0 |       1 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_ctx_from_first_arg_not_second                |      0 |       1 |        1 |
| tests/test_FLA/test_fla_utils_input_guard.py             | TestInputGuardContext.test_ctx_from_first_kwarg_when_no_tensor_in_args  |      0 |       1 |        1 |
| tests/test_convolution_double_backward.py                | test_convolution_double_backward                                        |      0 |       1 |        1 |
| tests/test_fp8_einsum.py                                 | test_accuracy_fp8_einsum                                                |      0 |       1 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine.py | test_fake_quantize_learnable_per_channel_affine_transposed              |      0 |       1 |        1 |
| tests/test_fp8_fp4_mega_moe.py                           | test_fp8_fp4_mega_moe                                                   |      0 |       1 |        1 |
| tests/test_fp8_fp4_mega_moe.py                           | test_fp8_fp4_mega_moe_torch_ref                                         |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_gate_engages                                         |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_gate_defers                                          |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_gate_limits_follow_element_width                     |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax                                                      |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_out                                                  |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_half_to_float                                        |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_neg_inf                                              |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_at_row_cap                                           |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                              | test_rdna4_softmax_non_contiguous                                       |      0 |       1 |        1 |
| tests/test_reshape_and_cache.py                          | test_reshape_and_cache_vllm_accuracy                                    |      0 |       1 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_inplace_extrema_special_values                      |      0 |       1 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_sum_contention                                      |      0 |       1 |        1 |
| tests/test_scatter_reduce.py                             | test_scatter_reduce_amin_nan_layout                                     |      0 |       1 |        1 |
| TOTAL                                                    |                                                                         |    437 |      22 |      459 |
|                       filepath                       |                               function                               | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------- | -------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_silu_and_mul_with_clamp.py                | test_silu_and_mul_with_clamp                                         |      1 |      0 |       0 |        1 |
| tests/test_silu_and_mul_with_clamp.py                | test_silu_and_mul_with_clamp_out                                     |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_scalar_tensor.py              | test_bitwise_xor_scalar_tensor                                       |      1 |      0 |       0 |        1 |
| tests/test_sobol_engine_initialize_state_.py         | test_sobol_engine_initialize_state_                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                    | test_sparse_broadcast_to_identity                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                    | test_sparse_broadcast_to_prepend                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_broadcast_to.py                    | test_sparse_broadcast_to_expand_sparse_dim                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear_with_bias                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear_n_ne_k                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear_activation                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear_with_bias_shapes                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_semi_structured_linear.py          | test_sparse_semi_structured_linear_non_contiguous_weight             |      1 |      0 |       0 |        1 |
| tests/test_special_bessel_y1.py                      | test_special_bessel_y1                                               |      1 |      0 |       0 |        1 |
| tests/test_special_erfcx.py                          | test_special_erfcx                                                   |      1 |      0 |       0 |        1 |
| tests/test_special_i0.py                             | test_special_i0                                                      |      1 |      0 |       0 |        1 |
| tests/test_special_i0.py                             | test_special_i0_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_special_log_softmax.py                    | test_special_log_softmax                                             |      1 |      0 |       0 |        1 |
| tests/test_special_log_softmax.py                    | test_special_log_softmax_large_n                                     |      1 |      0 |       0 |        1 |
| tests/test_special_ndtri.py                          | test_special_ndtri                                                   |      1 |      0 |       0 |        1 |
| tests/test_special_ndtri.py                          | test_special_ndtri_edge_values                                       |      1 |      0 |       0 |        1 |
| tests/test_special_ndtri.py                          | test_special_ndtri_non_contiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_special_ndtri.py                          | test_special_ndtri_unsupported_dtype                                 |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_w.py | test_special_shifted_chebyshev_polynomial_w                          |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_w.py | test_special_shifted_chebyshev_polynomial_w_n_out_of_range           |      1 |      0 |       0 |        1 |
| tests/test_split_with_sizes_copy.py                  | test_split_with_sizes_copy                                           |      1 |      0 |       0 |        1 |
| tests/test_split_with_sizes_copy.py                  | test_split_with_sizes_copy_different_dims                            |      1 |      0 |       0 |        1 |
| tests/test_split_with_sizes_copy.py                  | test_split_with_sizes_copy_edge_cases                                |      1 |      0 |       0 |        1 |
| tests/test_sum_to_size.py                            | test_sum_to_size                                                     |      1 |      0 |       0 |        1 |
| tests/test_sum_to_size.py                            | test_sum_to_size_extreme                                             |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | test_linspace                                                        |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_dtype_and_endpoints                    |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_cpu_allocation_options                 |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_default_and_meta_device                |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_invalid_zero_endpoints                 |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_existing_step_rejection                |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_no_fill_launch_and_fresh_output        |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_meta_endpoints                         |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_meta_endpoint_rank_validation          |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_meta_allocation_options                |      1 |      0 |       0 |        1 |
| tests/test_linspace.py                               | TestGenericLinspaceEmpty.test_meta_sparse_layout_rejected            |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                         | test_typed_pointer                                                   |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                         | test_typed_pointer_reinterpret_with_offset                           |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                         | test_typed_pointer_as_is                                             |      1 |      0 |       0 |        1 |
| tests/test_tensor_wrapper.py                         | test_strided_buffer_slice                                            |      1 |      0 |       0 |        1 |
| tests/test_threshold.py                              | test_threshold                                                       |      1 |      0 |       0 |        1 |
| tests/test_threshold.py                              | test_threshold_backward                                              |      1 |      0 |       0 |        1 |
| tests/test_topk_softplus_sqrt.py                     | test_topk_softplus_sqrt                                              |      1 |      0 |       0 |        1 |
| tests/test_topk_softplus_sqrt.py                     | test_topk_softplus_sqrt_hash                                         |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz                                                           |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_edge_cases                                                |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_large_reduction                                           |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_empty_input                                               |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_single_length_dim                                         |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_invalid_dim                                               |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_non_contiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_int_promotion                                             |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_fp64                                                      |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_complex                                                   |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_bool                                                      |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_invalid_dx                                                |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_empty_dim_ordering                                        |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_single_length_dim_ordering                                |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_bool_complex_ordering                                     |      1 |      0 |       0 |        1 |
| tests/test_trapz.py                                  | test_trapz_backward                                                  |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc                                                           |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_                                                          |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_special_values                                            |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_always_rounds_towards_zero                                |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_integer_input                                             |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_non_contiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_trunc.py                                  | test_trunc_empty                                                     |      1 |      0 |       0 |        1 |
| tests/test_unfold_copy.py                            | test_unfold_copy                                                     |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bin_ct                                       |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bin_ct_with_range                            |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bin_ct_with_weight                           |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bin_ct_with_density                          |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bins_tensor                                  |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                              | test_accuracy_histogram_bins_tensor_nonuniform                       |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d.py                     | test_upsample_bicubic2d                                              |      1 |      0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                   | test_slow_conv2d_backward                                            |      1 |      0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                   | test_slow_conv2d_backward_dispatches_to_gems                         |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                  | test_slow_conv_transpose3d                                           |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                  | test_slow_conv_transpose3d_bias                                      |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                  | test_slow_conv_transpose3d_strided_bias                              |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                  | test_slow_conv_transpose3d_output_padding_dilation                   |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                  | test_slow_conv_transpose3d_scalar_stride                             |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py     | test_value_selecting_reduction_backward                              |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py     | test_value_selecting_reduction_backward_max_dim                      |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py     | test_value_selecting_reduction_backward_min_dim                      |      1 |      0 |       0 |        1 |
| tests/test_bitwise_left_shift_.py                    | test_bitwise_left_shift_                                             |      1 |      0 |       0 |        1 |
| tests/test_xor.py                                    | test_xor                                                             |      1 |      0 |       0 |        1 |
| tests/test_xor.py                                    | test_xor_scalar                                                      |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put                                                             |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_dtypes                                                      |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_accumulate_duplicates                                       |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_negative_index                                              |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_index_bounds                                                |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_index_out_of_range                                          |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_non_contiguous                                              |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_non_contiguous_high_rank                                    |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_storage_offset                                              |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_non_contiguous_index_and_source                             |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_empty_index                                                 |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_zero_sized_self                                             |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_does_not_mutate_self                                        |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_device_mismatch                                             |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_invalid_dtypes                                              |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_index_source_diff_shapes                                    |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_out_non_contiguous_out                                      |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_out_invalid                                                 |      1 |      0 |       0 |        1 |
| tests/test_put.py                                    | test_put_out_error_leaves_out_untouched                              |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform                                              |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn                                           |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn_scale                                     |      1 |      0 |       0 |        1 |
| tests/test_hadamard_transform.py                     | test_hadamard_transform_mn_leading_dims                              |      1 |      0 |       0 |        1 |
| tests/test_bitwise_right_shift_.py                   | test_bitwise_right_shift_                                            |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized                              |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_axis                         |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_float_zero_points            |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_device_default_cpu           |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_memory_format                |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_requires_quantized_dtype     |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_rejects_bad_qparams          |      1 |      0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py     | test_empty_per_channel_affine_quantized_rejects_string_memory_format |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution                                      |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_kernel_size                          |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_padding_stride                       |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_default_stride                       |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_bias                                 |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_out                                  |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_noncontiguous                        |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_backward                             |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_special_values                       |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_unsupported_dtype                    |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_invalid_input_shape                  |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_invalid_weight_shape                 |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_negative_padding                     |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_negative_stride                      |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_non_tensor_weight                    |      1 |      0 |       0 |        1 |
| tests/test_nnpack_spatial_convolution.py             | test_nnpack_spatial_convolution_nonsquare_kernel                     |      1 |      0 |       0 |        1 |
| tests/test_silu_and_mul.py                           | test_silu_and_mul                                                    |      1 |      0 |       0 |        1 |
| tests/test_silu_and_mul.py                           | test_silu_and_mul_out                                                |      1 |      0 |       0 |        1 |
| tests/test_negative.py                               | test_negative                                                        |      1 |      0 |       0 |        1 |
| tests/test_ones_like.py                              | test_ones_like                                                       |      1 |      0 |       0 |        1 |
| tests/test_ones_like.py                              | test_ones_like_empty                                                 |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_shape                          |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_unsupported_dtype                  |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_padding                        |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_stride                         |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_groups                         |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_bad_input_size                     |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_none_operand                       |      1 |      0 |       0 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_dense_operand                      |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_src                                                     |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_reduce_add                                              |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_add_                                                    |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_add_2d_noncontiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_add                                                     |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_add_2d                                                  |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_reduce_multiply                                         |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_src_                                                    |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_reduce_add_                                             |      1 |      0 |       0 |        1 |
| tests/test_scatter.py                                | test_scatter_reduce_multiply_                                        |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                            | test_hash_tensor                                                     |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                            | test_hash_tensor_int_dtypes                                          |      1 |      0 |       0 |        1 |
| tests/test_hash_tensor.py                            | test_hash_tensor_multidim                                            |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm                                                           |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_beta_zero_ignores_bias                                    |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_vector_bias_shapes                                        |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_scalar_bias                                               |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_padded_vector_bias                                        |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_beta_zero_ignores_bias                                |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_vector_bias_shapes                                    |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_noncontiguous                                         |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_broadcast_bias                                            |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_fp32_accum                                          |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_beta_zero_ignores_bias                              |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_out_fp32_accum                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                  | test_addmm_dtype_out_beta_zero_ignores_bias                          |      1 |      0 |       0 |        1 |
| tests/test_mv.py                                     | test_mv                                                              |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_accuracy                                     |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_zero_infinity                                |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_tensor_overload                              |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_out                                          |      1 |      0 |       0 |        1 |
| tests/test__ctc_loss_backward.py                     | test__ctc_loss_backward_out_resize                                   |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add                                                             |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add_                                                            |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add_tensor_scalar                                               |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add_tensor_scalar_                                              |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add_scalar_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test_add.py                                    | test_add_scalar_scalar                                               |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                  | test_cholesky_solve_helper                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                  | test_cholesky_solve_helper_layouts                                   |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                  | test_cholesky_solve_helper_batch_strides                             |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                  | test_cholesky_solve_helper_empty                                     |      1 |      0 |       0 |        1 |
| tests/test_eq.py                                     | test_eq                                                              |      1 |      0 |       0 |        1 |
| tests/test_eq.py                                     | test_eq_scalar                                                       |      1 |      0 |       0 |        1 |
| tests/test_eq.py                                     | test_eq_integer_precision                                            |      1 |      0 |       0 |        1 |
| tests/test_eq.py                                     | test_eq_scalar_integer_precision                                     |      1 |      0 |       0 |        1 |
| tests/test_eq.py                                     | test_eq_scalar_float_scalar                                          |      1 |      0 |       0 |        1 |
| tests/test_lift_fresh.py                             | test_lift_fresh                                                      |      1 |      0 |       0 |        1 |
| tests/test_lift_fresh.py                             | test_lift_fresh_empty                                                |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d                                             |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_public                                      |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_out                                         |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_4d_input                                    |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_dispatch_kernels                            |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_out_4d_input                                |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_public_out_overload                         |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_public_out_overload_4d_input                |      1 |      0 |       0 |        1 |
| tests/test__adaptive_avg_pool3d.py                   | test_adaptive_avg_pool3d_public_backward                             |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_backward_large_m                                     |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_backward_output_mask                                 |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_backward_from_forward                                |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_empty_rows                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_explicit_size                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_rejects_bad_layout                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap     |      1 |      0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                       | test_linalg_ldl_solve                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                       | test_linalg_ldl_solve_batched                                        |      1 |      0 |       0 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py      | test_stage_deepseek_v4_mega_moe_inputs_accuracy                      |      0 |      0 |       1 |        1 |
| tests/test_stage_deepseek_v4_mega_moe_inputs.py      | test_stage_deepseek_v4_mega_moe_inputs_rejects_bad_hidden_size       |      0 |      0 |       1 |        1 |
| tests/test_topk_softplus_sqrt.py                     | test_topk_softplus_sqrt_vs_vllm                                      |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight                                    |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_size_inferred                      |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_params                             |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_out                                |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_special_values                     |      0 |      0 |       1 |        1 |
| tests/test_mkldnn_reorder_conv2d_weight.py           | test_mkldnn_reorder_conv2d_weight_rejects_backward                   |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_large_matrix                                              |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_large_matrix_alpha_beta                                   |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_small_square_large                                        |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_out_large_matrix                                          |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_bounded_workspace                                         |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_fallback_autotuning                                       |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_beta_cache_transitions                             |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_empty                                       |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_out_resize                                  |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_beta_zero                                   |      0 |      0 |       1 |        1 |
| tests/test_addmm.py                                  | test_addmm_ascend_review_out_validation                              |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mm_w8a8_int8                                                    |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mm_w8a8_int8_strides                                            |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mm_w8a8_int8_updates                                            |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mm_w8a8_int8_empty                                              |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mm_w8a8_int8_reject_float                                       |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_scaled_bias                                                     |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_scaled_empty_bias                                               |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_scaled_graph_bias_updates                                       |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_scaled_validation                                               |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_tensor_scales_without_bias                                      |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_scaled_unaligned_weight                                         |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_prequantized                                                    |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_long_k_overflow                                                 |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_mixed_scales                                                    |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_graph_updates                                                   |      0 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                           | test_invalid                                                         |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_backward_mthreads_vendor_dispatch                    |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_contract                                              |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_strides                                               |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_special_values                                        |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_real_shape                                            |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_errors                                                |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_exact_segments                                        |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_core_shapes                                           |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layernorm_large_normalized_extents                              |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_saved_statistics_backward                            |      0 |      0 |       1 |        1 |
| tests/test_weight_norm.py                            | test_weight_norm                                                     |      0 |      1 |       0 |        1 |
| tests/test_put.py                                    | test_put_complex32_rejected                                          |      0 |      1 |       0 |        1 |
| tests/test_cdist_backward.py                         | test_cdist_backward                                                  |      0 |      1 |       0 |        1 |
| tests/test_cdist_backward.py                         | test_cdist_backward_zero_distance                                    |      0 |      1 |       0 |        1 |
| tests/test_cdist_backward.py                         | test_cdist_backward_padded_lanes_no_nan                              |      0 |      1 |       0 |        1 |
| tests/test_cdist_backward.py                         | test_cdist_backward_p_zero                                           |      0 |      1 |       0 |        1 |
| tests/test_cdist_backward.py                         | test_cdist_backward_partial_zero_diff                                |      0 |      1 |       0 |        1 |
| tests/test_hash_tensor.py                            | test_hash_tensor_scalar                                              |      0 |      1 |       0 |        1 |
| tests/test_add.py                                    | test_add_complex                                                     |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_native_layer_norm                                               |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm                                                      |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_fp32_affine                                          |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_native_layer_norm_statistics                                    |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_native_layer_norm_fp32_statistics                               |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_nvidia_paths                                         |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                             | test_layer_norm_fp32_affine_noncontiguous                            |      0 |      1 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py               | test_sparse_compressed_tensor_infer_size                             |      0 |      1 |       0 |        1 |
| TOTAL                                                |                                                                      |    224 |     17 |      46 |      287 |
