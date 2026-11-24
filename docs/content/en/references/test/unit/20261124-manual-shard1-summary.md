|                             filepath                             |                                           function                                            | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/core/test_dynamic_override.py                              | test_override_call_count_snapshots_and_all_skip_restore                                       |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_basic_override                                                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_multiple_overrides                                                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_restore_single_op                                                  |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_restore_all                                                        |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_context_manager                                                    |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_list_overrides                                                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_override_from_file                                                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_override_batch_from_files                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_override_nonexistent_file                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_override_missing_function                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_restore_without_override                                           |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_operator_with_kwargs                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_concurrent_registries                                              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestDynamicOpOverride.test_override_special_ops                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_parse_override_spec_colon                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_parse_override_spec_equals                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_parse_override_spec_invalid                                              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_load_override_config_yaml                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_load_override_config_json                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestCLIOverride.test_load_override_config_missing_file                                        |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestIntegrationScenarios.test_ab_testing_scenario                                             |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestIntegrationScenarios.test_progressive_override                                            |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestPytestIntegration.test_override_option_registered                                         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestPytestIntegration.test_override_applied_through_cli                                       |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestPytestIntegration.test_override_config_file_applied_through_cli                           |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestPytestIntegration.test_no_override_leaves_default_implementation                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestPytestIntegration.test_missing_operator_is_available_during_collection                    |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_pytest_parser_accepts_the_plugin                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_missing_candidate_aborts                             |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_missing_run_aborts                                   |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_unknown_unused_operator_fails_and_is_removed         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_existing_registration_uses_candidate                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_unused_candidate_makes_pytest_fail                   |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestRegistrarLiveOverrideResolution.test_direct_call_and_restore_work                         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestUnimplementedOperatorOverride.test_repeated_override_restores_original_state              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestUnimplementedOperatorOverride.test_exception_restores_existing_and_removes_added_operator |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                              | TestUnimplementedOperatorOverride.test_file_candidate_for_missing_operator                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_computes_requested_metrics_and_preserves_duplicate_shapes                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_reads_v2_rows_and_rounds_only_serialized_metrics                                         |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_private_rows_map_platform_key_to_public_model_column                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_v3_requires_identical_benchmark_protocols                                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_v3_protocol_is_preserved_in_comparison_output                                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_rejects_ambiguous_or_incompatible_source_rows                                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_keeps_invalid_measurements_but_omits_derived_metrics                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_cli_writes_baseline_and_ours_columns                                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_rejects_cross_platform_comparison                                                        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_comparison_requires_platform_identity                                                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_non_mm_dimensions_are_preserved_in_csv_and_jsonl                                         |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                           | test_rejects_different_workload_dimension_schemas                                             |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_stable_state_skips_mode_resolution_and_device_changes_recheck                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_expired_auto_failure_invalidates_dispatch_even_after_idle                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_environment_changes_recheck_and_removal_restores_state                                   |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_changed_mode_clears_compiled_entries                                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_failed_resolution_is_not_cached                                                          |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_api_include_state_and_registry_changes_invalidate                                        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_live_mode_switches_preserve_default_cost_and_expanded                                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_invalid_live_setting_raises_until_fixed                                                  |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_explicit_default_ignores_irrelevant_model_and_include_changes                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_model_source_is_not_a_mode_selection_dependency                                          |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py                      | test_auto_fuse_invalidates_all_entries_for_the_same_model                                     |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py          | test_fused_recurrent_gated_delta_rule_fwd_accuracy                                            |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                            | test_unflatten_dense_tensors                                                                  |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                            | test_unflatten_dense_tensors_metadata                                                         |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                            | test_unflatten_dense_tensors_strided_flat                                                     |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                            | test_unflatten_dense_tensors_exceeds_flat                                                     |      1 |      0 |       0 |        1 |
| tests/test__weight_norm.py                                       | test_underscore_weight_norm                                                                   |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d_backward.py                       | test_adaptive_avg_pool2d_backward                                                             |      1 |      0 |       0 |        1 |
| tests/test_affine_grid_generator.py                              | test_affine_grid_generator                                                                    |      1 |      0 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py                 | test_validate_compressed_sparse_indices                                                       |      1 |      0 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py                 | test_validate_compressed_sparse_indices_invalid_indices                                       |      1 |      0 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py                 | test_validate_compressed_sparse_indices_large_batch                                           |      1 |      0 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py                 | test_validate_compressed_sparse_indices_invalid_metadata                                      |      1 |      0 |       0 |        1 |
| tests/test_validate_compressed_sparse_indices.py                 | test_validate_compressed_sparse_indices_dispatch                                              |      1 |      0 |       0 |        1 |
| tests/test_arctan_.py                                            | test_arctan                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_arctan_.py                                            | test_arctan_                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_various_sizes                                                                      |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_edge_cases                                                                         |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_empty_tensor                                                                       |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_accuracy_asinh_out                                                                       |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_avg_pool2d.py                                         | test_avg_pool2d                                                                               |      1 |      0 |       0 |        1 |
| tests/test_avg_pool2d.py                                         | test_avg_pool2d_backward                                                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_2d                                                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_2d_dilation                                                             |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_1d                                                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_3d                                                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_invalid_padding                                                         |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                                   | test_convolution_mode_same_strided_raises                                                     |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli_                                                                               |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli_various_p                                                                      |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli                                                                                |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_scalar_tensor.py                          | test_bitwise_and_scalar_tensor                                                                |      1 |      0 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py               | test_native_batch_norm_legit_no_training                                                      |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_generic_and_cached_binding_agree                                                         |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_current_tensors_and_unspecialized_values_are_not_frozen                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_defaults_and_explicit_keyword_values_share_correct_plan                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_unspecialized_integer_width_matches_generic                                              |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_alignment_and_dtype_changes_do_not_reuse_wrong_entry                                     |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_cached_binding_preserves_backend_tensor_specialization                                   |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_custom_constexpr_uses_generic_fallback                                                   |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_hooks_receive_current_arguments                                                          |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_replaced_kernel_cache_entry_refreshes_launch_constants                                   |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_launch_plan_does_not_retain_input_tensors                                                |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_offline_benchmark_clear_also_releases_launch_plans                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_cached_binding_requires_explicit_opt_in                                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_unsupported_triton_uses_generic_binding                                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_disabled_binding_does_not_enter_fast_path                                                |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_cold_launch_uses_generic_path                                                            |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_cached_binding_preserves_custom_specialization_key                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_binding_plans_are_scoped_to_device                                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry_cached_binding.py                            | test_mode_change_clears_cached_binding_plans                                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                            | test_nested_view_from_buffer                                                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                            | test_nested_view_from_buffer_aliasing                                                         |      1 |      0 |       0 |        1 |
| tests/test_nested_view_from_buffer.py                            | test_nested_view_from_buffer_3d_components                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe                                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe_dtype                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe_skips_index_validation                                          |      1 |      0 |       0 |        1 |
| tests/test_conj.py                                               | test_conj                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_conj.py                                               | test__conj                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward                                                                   |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_weighted_empty_mapping                                            |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_native                                                            |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_zero_dimension                                                    |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_sparse_chunk_boundaries                                           |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_sparse_mean_rounding                                              |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_empty                                                             |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_sparse_frequency_error                                            |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_invalid                                                           |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_deterministic                                                     |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_empty_table                                                       |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution                                                                              |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_empty_batch                                                                  |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_noncontiguous_input                                                          |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_noncontiguous_weight                                                         |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_spatial_rank                                                                 |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_depthwise                                                                    |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_transposed                                                                   |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_bias_is_optional                                                             |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_non_positive_groups                                                          |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_input_channel_mismatch                                                       |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_groups_do_not_divide_channels                                                |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_empty_weight                                                                 |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_bias_shape_mismatch                                                          |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_rank_mismatch                                                                |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_non_positive_stride                                                          |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_non_positive_dilation                                                        |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_negative_padding                                                             |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_output_padding_too_large                                                     |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_kernel_larger_than_padded_input                                              |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_dilated_kernel_larger_than_input                                             |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_rejects_complex                                                              |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                        | test_convolution_autograd_input_is_detached                                                   |      1 |      0 |       0 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor                                                                      |      1 |      0 |       0 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor_lazy_bits                                                            |      1 |      0 |       0 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor_gradient                                                             |      1 |      0 |       0 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor_invalid_shape                                                        |      1 |      0 |       0 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor_tiled_copy                                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler                                                                       |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_non_contiguous                                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_nan_inf_grid                                                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_special_values                                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_invalid_grid                                                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_out                                                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_out_resize                                                            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_out_strided                                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_grid_sampler.py                                 | test_cudnn_grid_sampler_out_dtype_mismatch                                                    |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dropout_pos_rejected                                            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dual_dim_bias                                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_onesided_cumseq_rejected                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_varlen_rejected                                                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_value_head_dim                                                  |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_posdim                                                                     |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_posdim_with_nan                                                            |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_posdim_bool                                                                |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_negdim                                                                     |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_negdim_with_nan                                                            |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_negdim_bool                                                                |      1 |      0 |       0 |        1 |
| tests/test_cummax_helper.py                                      | test_cummax_helper_dim0                                                                       |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine_backward.py | test_fake_quantize_learnable_per_tensor_affine_backward                                       |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine_backward.py | test_fake_quantize_learnable_per_tensor_affine_backward_clamped                               |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu                                                                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_dilation                                                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_asymmetric                                                        |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_3d                                                                |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_1d                                                                |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_no_bias                                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_bias_shapes                                                       |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_boundary                                                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_out                                                               |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_relu.py                             | test_cudnn_convolution_relu_out_no_bias                                                       |      1 |      0 |       0 |        1 |
| tests/test_digmma.py                                             | test_digamma_                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_value_ranges                                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_sparse_self                                                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_non_contiguous                                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_out                                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_nan_inf                                                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_backward                                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_shape_mismatch                                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_rejects_dense_mask                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                        | test_sparse_mask_dispatch                                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_default                                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_default_empty                                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_default_uncoalesced                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dtype                                                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim                                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_all                                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_dtype                                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_out                                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_negative                                                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_keeps_structural_zeros                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_hybrid_dense_dims                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_uncoalesced_input                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_zero_size_dims                                                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_integer_promotion                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_duplicate_dim_raises                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_out_of_range_dim_raises                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dense_input_raises                                                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_csr_input_raises                                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dim_out_dtype_mismatch_raises                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_dtype_accumulator_precision                                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_complex                                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_dispatch_reaches_gems_impl                                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_dispatch_covers_every_overload                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                         | test_sparse_sum_uncoalesced_matches_aten                                                      |      1 |      0 |       0 |        1 |
| tests/test_embedding_sparse_backward.py                          | test_embedding_sparse_backward                                                                |      1 |      0 |       0 |        1 |
| tests/test_eye.py                                                | test_eye                                                                                      |      1 |      0 |       0 |        1 |
| tests/test_eye.py                                                | test_eye_m                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py         | test_accuracy_fake_quantize_per_channel_affine_cachemask                                      |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py         | test_accuracy_fake_quantize_per_channel_affine_cachemask_half_to_even                         |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py         | test_accuracy_fake_quantize_per_channel_affine_cachemask_noncontiguous                        |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py         | test_accuracy_fake_quantize_per_channel_affine_cachemask_out                                  |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py         | test_accuracy_fake_quantize_per_channel_affine_cachemask_empty                                |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                    | test_feature_dropout                                                                          |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                    | test_feature_dropout_no_train                                                                 |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                    | test_feature_dropout_p_zero                                                                   |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                    | test_feature_dropout_p_one                                                                    |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                    | test_feature_dropout_inplace                                                                  |      1 |      0 |       0 |        1 |
| tests/test_fix_.py                                               | test_fix_                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_accuracy_flipud                                                                          |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_accuracy_flipud_non_float                                                                |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_accuracy_flipud_complex                                                                  |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_accuracy_flipud_autograd                                                                 |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_accuracy_flipud_noncontiguous                                                            |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                             | test_flipud_rejects_zero_dimensional_input                                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_fast_path                                                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_fallback_cases                                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_int_fallback                                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_writes_through_view                                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_registration_is_live                                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_rejects_empty_list                                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_fast_path                                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_half_to_even                                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_large_magnitude                                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_fallback_cases                                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_writes_through_view                                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_registration_is_live                                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_rejects_empty_list                                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_fast_path                                                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_fallback_cases                                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_int_fallback                                                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_writes_through_view                                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_registration_is_live                                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_rejects_empty_list                                                  |      1 |      0 |       0 |        1 |
| tests/test_frac.py                                               | test_frac                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_functional_sym_constrain_range.py                     | test_functional_sym_constrain_range                                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rng_bridge_build_failure                                                   |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_python_reservation_matches_native                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_python_reservation_contract                                                |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout                                                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_layout                                                                     |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_replay_and_independence                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_word_increment                                                   |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_counter_carry                                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_offset_overflow_leaves_state_unchanged                           |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_default_generator_replay                                                   |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_empty_default_rng                                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_nonfloating_dtype                                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_wrong_generator_device                                             |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_concurrent_generator_reservations                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_reservations_interoperate_with_native_rand                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_aten_autograd_uses_gems                                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_checkpoint_replays_default_rng                                             |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_graph_capture_without_consuming_rng                                |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_special_values                                                             |      1 |      0 |       0 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py     | test_combine_topk_swa_indices_accuracy                                                        |      1 |      0 |       0 |        1 |
| tests/test_ge.py                                                 | test_ge                                                                                       |      1 |      0 |       0 |        1 |
| tests/test_ge.py                                                 | test_ge_scalar                                                                                |      1 |      0 |       0 |        1 |
| tests/test_glu_backward_jvp.py                                   | test_glu_backward_jvp                                                                         |      1 |      0 |       0 |        1 |
| tests/test_glu_backward_jvp.py                                   | test_glu_backward_jvp_broadcast_dgrad                                                         |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                    | test_grid_sampler_2d                                                                          |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                    | test_grid_sampler_2d_oh_ne_iw                                                                 |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                    | test_grid_sampler_2d_nearest_half_integer                                                     |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                    | test_grid_sampler_2d_dim_one_reflection                                                       |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                                 | test_gt                                                                                       |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                                 | test_gt_scalar                                                                                |      1 |      0 |       0 |        1 |
| tests/test_hardswish_backward.py                                 | test_hardswish_backward                                                                       |      1 |      0 |       0 |        1 |
| tests/test_heaviside_.py                                         | test_heaviside_                                                                               |      1 |      0 |       0 |        1 |
| tests/test_hstack.py                                             | test_accuracy_hstack                                                                          |      1 |      0 |       0 |        1 |
| tests/test_hstack.py                                             | test_exception_hstack                                                                         |      1 |      0 |       0 |        1 |
| tests/test_igammac_.py                                           | test_special_gammaincc                                                                        |      1 |      0 |       0 |        1 |
| tests/test_igammac_.py                                           | test_igammac_                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_irshift.py                                            | test_irshift                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_irshift.py                                            | test_irshift_edge_cases                                                                       |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy                                                                           |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy_int                                                                       |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy_noncontiguous                                                             |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy_out                                                                       |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy_high_rank                                                                 |      1 |      0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                     | test_fw_primal_copy_out_high_rank                                                             |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                         | test_less_equal                                                                               |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                         | test_less_equal_scalar                                                                        |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                         | test_less_equal_byte                                                                          |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                         | test_less_equal_scalar_byte                                                                   |      1 |      0 |       0 |        1 |
| tests/test_isneginf.py                                           | test_isneginf                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_isneginf.py                                           | test_isneginf_out                                                                             |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_result_is_independent_of_input_storage                                          |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_out                                                                             |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_out_resizes_buffer                                                              |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_out_writes_through_strided_buffer                                               |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_reads_layout                                                                    |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_per_channel_quantization                                                        |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_extreme_qparams                                                                 |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_storage_boundaries                                                              |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_quantized_special_values                                                        |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_degenerate_shape                                                                |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_out_degenerate_shape                                                            |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_rejects_non_quantized_dtype                                                     |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_rejects_non_tensor_argument                                                     |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                           | test_int_repr_out_rejects_mismatched_dtype                                                    |      1 |      0 |       0 |        1 |
| tests/test_leaky_relu.py                                         | test_leaky_relu                                                                               |      1 |      0 |       0 |        1 |
| tests/test_leaky_relu.py                                         | test_leaky_relu_                                                                              |      1 |      0 |       0 |        1 |
| tests/test_leaky_relu.py                                         | test_leaky_relu_out                                                                           |      1 |      0 |       0 |        1 |
| tests/test_leaky_relu.py                                         | test_leaky_relu_backward                                                                      |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag                                                                            |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_forward_only                                                               |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_single_bag                                                                 |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_single_index                                                               |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_strided_weighted                                                           |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_mixed_indices_negative_padding                                             |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_native_abi                                                                 |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_large_float64                                                              |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_nonfinite_and_ties                                                         |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_empty_table                                                                |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_invalid_values                                                             |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_invalid_after_warmup                                                       |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_invalid_metadata                                                           |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_default                                                                    |      1 |      0 |       0 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_default_empty                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh                                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_empty                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_noncontiguous                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_identity                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_degenerate_eigenvalues                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_ascending_order                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_trace_is_preserved                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_reads_requested_triangle                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_default_uplo_is_lower                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_batch_independence                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_invalid_shape                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_invalid_uplo                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_rejects_complex                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_fp64_preserved                                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_autograd_input_is_detached                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_out                                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_out_resize                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                    | test_linalg_eigvalsh_out_rejects_mismatched_dtype                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_gems_path_active                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_vector                                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_matrix                                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_batched                                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_underdetermined                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_underdetermined_blocked                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_underdetermined_blocked_fp64                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_underdetermined_matrix                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_broadcast                                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_tall_monolithic_edge                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_tall_blocked                                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_tall_blocked_fp64                                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_rank_deficient                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_near_singular                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_fp64                                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_driver_rejected                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_degenerate                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_complex_fallback                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_square_wy                                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_square_wy_fp64                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_lstsq.py                                       | test_linalg_lstsq_underdetermined_wy                                                          |      1 |      0 |       0 |        1 |
| tests/test_nan_to_num_.py                                        | test_nan_to_num_                                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr                                                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_out_invalid                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_non_contiguous                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_invalid_mode                                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_input_not_mutated                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_zero_column                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_rank_deficient                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_empty                                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_transposed_input                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_unsupported_dtype                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_dim_lt_2                                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve                                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_empty                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_noncontiguous                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_noncontiguous_multidim_lead                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_recovers_known_solution                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_dims                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_dims_defaults_to_none                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_dims_out_of_range                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_dims_duplicate                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_shape_mismatch                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_residual_is_small                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_singular_does_not_hang_or_crash                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_rejects_complex                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_fp64_preserved                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_fp64_accuracy_is_double_grade                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorsolve.py                                 | test_linalg_tensorsolve_autograd_input_is_detached                                            |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward                                                                           |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward_expanded_grad                                                             |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward_special_values                                                            |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward_matches_aten_rounding                                                     |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward_via_torch_dispatch                                                        |      1 |      0 |       0 |        1 |
| tests/test_log1p.py                                              | test_log1p_backward_fused_kernel_count                                                        |      1 |      0 |       0 |        1 |
| tests/test_logaddexp.py                                          | test_logaddexp                                                                                |      1 |      0 |       0 |        1 |
| tests/test_logaddexp.py                                          | test_logaddexp_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_logical_xor_.py                                       | test_logical_xor_                                                                             |      1 |      0 |       0 |        1 |
| tests/test_logical_xor_.py                                       | test_logical_xor_bool_scalar                                                                  |      1 |      0 |       0 |        1 |
| tests/test_logical_xor_.py                                       | test_logical_xor_broadcast                                                                    |      1 |      0 |       0 |        1 |
| tests/test_logical_xor_.py                                       | test_logical_xor_noncontiguous                                                                |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack                                                                                |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_unpack_data_false                                                              |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_unpack_pivots_false                                                            |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_batched                                                                        |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_backward                                                                       |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_backward_entry                                                                 |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_backward_batched                                                               |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_backward_nondifferentiable_outputs                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                          | test_lu_unpack_backward_dispatch                                                              |      1 |      0 |       0 |        1 |
| tests/test_flash_mla.py                                          | test_flash_mla                                                                                |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                                    | test_matmul_backward                                                                          |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                                    | test_matmul_backward_self_only                                                                |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                                    | test_matmul_backward_other_only                                                               |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                                      | test_diagonal_copy                                                                            |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                                      | test_diagonal_copy_int                                                                        |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                                      | test_diagonal_copy_low_precision                                                              |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                                      | test_diagonal_copy_empty                                                                      |      1 |      0 |       0 |        1 |
| tests/test_diagonal_copy.py                                      | test_diagonal_copy_single_element                                                             |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim                                                                            |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim                                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_nan                                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_lastdim_sort                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_lastdim_sort_nan                                                           |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_direct_flat_nan                                                            |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_tie_lower_median                                                                  |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_duplicate_indices_select_value                                             |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_reduction_boundary_dim0                                                           |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_public_shapes_dim0                                                         |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_nan_first_index_dim0                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_non_contiguous_dim0                                                        |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_named_dim0_preserves_names                                                 |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_no_dim                                                                      |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_extra_no_dim_dtypes                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim                                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_full_registration                                                     |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_large                                                                 |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_beyond_old_flat_limit                                                 |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_dim_count_selects_first_index                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_full_registration_nan_semantics                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_complex_no_dim                                                              |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_complex_nonempty_errors                                                           |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_reduced_dim_raises                                                          |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_output_unsupported_dtype                                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_scalar_dim                                                                        |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_non_contiguous                                                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_high_dim_semantics                                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width                                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_float64_key_select                                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_float64_key_select_nan_first_index                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_extended_lastdim_width                                                            |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_lastdim_sort_unique_exact_index                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_lastdim_sort_1536_unique_exact_index                                              |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_boundaries                                                        |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_unique_exact_index                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_nan_first_index                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_duplicates_infinities_and_zeros                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_signed_zero_bits                                                  |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_boundaries                                                         |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp16_key_select_unique_exact_index                                                |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_nan_first_index                                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_width640_keepdim                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_duplicates_infinities_and_zeros                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_signed_zero_index_bits                                             |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_width640_nonlast_and_no_dim                                        |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width_nan_first_index                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_fallback_nan                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_non_lastdim_fallback_nan_first_index                                              |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_large_reduction_semantics                                         |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_large_reduction_out                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_named_dim_preserves_names                                         |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width_duplicate_indices_select_value                                        |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_boundaries                                                     |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_unique_exact_index                                             |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_all_equal                                                      |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_out                                                                               |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_out_error_paths                                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out                                                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out_python_api                                                         |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out_wrong_device                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_error_paths                                                                       |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim                                                                   |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_ties_and_special_values                                                  |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_nan                                                                      |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_non_contiguous                                                    |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_is_a_graph_node                                                          |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim                                                                      |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_ties_follow_saved_index                                              |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_nan_follows_saved_index                                              |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_requires_grad_inputs                                                     |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_large                                                             |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_large_duplicate_median                                            |      1 |      0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_non_contiguous                                                       |      1 |      0 |       0 |        1 |
| tests/test_miopen_batch_norm_backward.py                         | test_miopen_batch_norm_backward                                                               |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_moe_load_balance_loss_matches_reference                                                  |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_tle_path_with_irregular_large_shape                                                      |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_noncontiguous_logits_and_mask                                                            |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_all_masked_tokens_return_zero                                                            |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_default_top_k_is_two                                                                     |      1 |      0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_moe_load_balance_loss_validation                                                         |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward                                                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_empty_batch                                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_noncontiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_nan_and_inf                                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_watchdog_path                                 |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_full_target                                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_duplicate                                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_duplicate_tail                                |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_empty_prefix                                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_large_class_bucket_boundaries                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_many_rows_target_length_reduce                            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_invalid_target_values_raise                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_invalid_after_distant_sentinel_raises                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_invalid_target_metadata_raises                            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py                     | test_multilabel_margin_loss_forward_invalid_input_shape_raises                                |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian                                                                                |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_large_flat                                                                     |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_hygon_flat_above_block_limit                                                   |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim                                                                            |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_large_radix_path                                                               |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_all_nan_rows                                                                   |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_non_contiguous                                                                 |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_values                                                                     |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_values_large_int                                                           |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_values_non_contiguous_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_empty                                                                          |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_out_empty                                                                      |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_empty                                                                      |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_bool_unsupported                                                               |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_byte_boundaries                                                                |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_out_byte_boundaries                                                            |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_byte_boundaries                                                            |      1 |      0 |       0 |        1 |
| tests/test_nanmedian.py                                          | test_nanmedian_dim_values_byte_boundaries                                                     |      1 |      0 |       0 |        1 |
| tests/test_native_norm.py                                        | test_native_norm                                                                              |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_tensor.py                          | test_nested_from_padded_tensor_3d                                                             |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_tensor.py                          | test_nested_from_padded_tensor_2d                                                             |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_tensor.py                          | test_nested_from_padded_tensor_sum_s                                                          |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_tensor.py                          | test_nested_from_padded_tensor_cache_invalidation                                             |      1 |      0 |       0 |        1 |
| tests/test_nested_from_padded_tensor.py                          | test_nested_from_padded_tensor_ragged_idx                                                     |      1 |      0 |       0 |        1 |
| tests/test_nested_to_padded_tensor.py                            | test_nested_to_padded_tensor                                                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_to_padded_tensor.py                            | test_nested_to_padded_tensor_padding                                                          |      1 |      0 |       0 |        1 |
| tests/test_nested_to_padded_tensor.py                            | test_nested_to_padded_tensor_dispatch                                                         |      1 |      0 |       0 |        1 |
| tests/test_nested_to_padded_tensor.py                            | test_nested_to_padded_tensor_output_size                                                      |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter                                                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_                                                                               |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_out_none                                                                       |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_inplace_is_self                                                                |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_rejects_out                                                                    |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_edge_values                                                                    |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_nan                                                                            |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_infinity                                                                       |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                                          | test_nextafter_small_values                                                                   |      1 |      0 |       0 |        1 |
| tests/test_nonzero.py                                            | test_nonzero                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_not_equal_.py                                         | test_accuracy_not_equal_                                                                      |      1 |      0 |       0 |        1 |
| tests/test_not_equal_.py                                         | test_accuracy_not_equal_scalar_                                                               |      1 |      0 |       0 |        1 |
| tests/test_outer.py                                              | test_outer                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_correctness                                                                 |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_padding_value                                                               |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_empty_error                                                                 |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_high_rank_boundary                                                          |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_large_batch_correctness                                                     |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_invalid_shape                                                               |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_scalar_error                                                                |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_noncontiguous_boundary                                                      |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_mixed_device_error                                                          |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_mixed_dtype                                                                 |      1 |      0 |       0 |        1 |
| tests/test_pad_sequence.py                                       | test_pad_sequence_zero_output                                                                 |      1 |      0 |       0 |        1 |
| tests/test_per_token_group_quant_fp8.py                          | test_per_token_group_quant_fp8                                                                |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_partition                                                                  |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_partition_rejects_nonpositive_tiles                                        |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_partition_rejects_nonpositive_max_grid                                     |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_codegen_config_balance_grid_defaults_off                                                 |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_codegen_config_accepts_balance_grid_opt_in                                               |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_partition_covers_each_tile_once                                            |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_generated_wrapper_source_and_runtime_partition                             |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_cache_isolation                                                            |      1 |      0 |       0 |        1 |
| tests/test_pointwise_dynamic_balance_grid.py                     | test_balanced_grid_real_kernel_covers_non_power_of_two_tail                                   |      1 |      0 |       0 |        1 |
| tests/test_prelu.py                                              | test_prelu                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm                                                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_channels_last                                                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_empty                                                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_zero_output_scale                                                   |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_bad_size                                                            |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_bad_dtype                                                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out                                                                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_mismatched_qparams                                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_resizes                                                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_storage_offset                                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_empty                                                           |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_wrong_dtype                                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_batch_norm.py                               | test_quantized_batch_norm_out_wrong_device                                                    |      1 |      0 |       0 |        1 |
| tests/test_rand_like.py                                          | test_rand_like                                                                                |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d_backward.py                          | test_reflection_pad3d_backward                                                                |      1 |      0 |       0 |        1 |
| tests/test_repeat.py                                             | test_repeat                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_reshape_alias_copy.py                                 | test_accuracy_reshape_alias_copy                                                              |      1 |      0 |       0 |        1 |
| tests/test_reshape_alias_copy.py                                 | test_accuracy_reshape_alias_copy_noncontiguous_input                                          |      1 |      0 |       0 |        1 |
| tests/test_reshape_alias_copy.py                                 | test_accuracy_reshape_alias_copy_storage_offset                                               |      1 |      0 |       0 |        1 |
| tests/test_reshape_alias_copy.py                                 | test_accuracy_reshape_alias_copy_independent_storage                                          |      1 |      0 |       0 |        1 |
| tests/test_reshape_alias_copy.py                                 | test_accuracy_reshape_alias_copy_zero_dim                                                     |      1 |      0 |       0 |        1 |
| tests/test_resolve_conj.py                                       | test_resolve_conj                                                                             |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell                                                                   |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_non_contiguous                                                    |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_zero_sized                                                        |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_invalid_input                                                     |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_complex_rejected                                                  |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_autograd_graph                                                    |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                      | test_accuracy_rnn_tanh_cell_keyword_arguments                                                 |      1 |      0 |       0 |        1 |
| tests/test_rrelu.py                                              | test_rrelu                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_rrelu.py                                              | test_rrelu_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_rwkv_ka_fusion.py                                     | test_accuracy_rwkv_kafusion                                                                   |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py                 | test_scaled_dot_product_cudnn_attention                                                       |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py                 | test_scaled_dot_product_cudnn_attention_with_attn_bias                                        |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy                                                                     |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_non_contiguous                                                      |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_empty                                                               |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_empty_non_contiguous                                                |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_zero_dim                                                            |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_single_element                                                      |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_autograd_contract                                                   |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_dispatch_stability                                                  |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_neg_view                                                            |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_neg_view_complex                                                    |      1 |      0 |       0 |        1 |
| tests/test_detach_copy.py                                        | test_accuracy_detach_copy_complex_layouts                                                     |      1 |      0 |       0 |        1 |
| tests/test_selu.py                                               | test_selu                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_selu.py                                               | test_selu_                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py          | test_fused_recurrent_gated_delta_rule_matches_vllm                                            |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_hygon_alignment                                                   |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_legacy_ascend_empty_table_invalid                                 |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_legacy_ascend_sparse_zero_dim_prefix                              |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_iluvatar_max                                                      |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag_backward.py                             | test_embedding_bag_backward_iluvatar_masked_indices                                           |      0 |      0 |       1 |        1 |
| tests/test_reshape_from_tensor.py                                | test_reshape_from_tensor_multi_device                                                         |      0 |      0 |       1 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_scale                                                           |      0 |      0 |       1 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_large_64bit_indexing                                                       |      0 |      0 |       1 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py     | test_combine_topk_swa_indices_vllm_accuracy                                                   |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_hygon_alignment                                                            |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_ascend_short_bags                                                          |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_mthreads_launch                                                            |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_iluvatar_max                                                               |      0 |      0 |       1 |        1 |
| tests/test_embedding_bag.py                                      | test_embedding_bag_iluvatar_masked_indices                                                    |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_gate_engages                                                               |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_gate_defers                                                                |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_gate_limits_follow_element_width                                           |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax                                                                            |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_out                                                                        |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_half_to_float                                                              |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_neg_inf                                                                    |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_at_row_cap                                                                 |      0 |      0 |       1 |        1 |
| tests/test_rdna4_softmax.py                                      | test_rdna4_softmax_non_contiguous                                                             |      0 |      0 |       1 |        1 |
| tests/test_addbmm_.py                                            | test_accuracy_addbmm_                                                                         |      0 |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward                                                                 |      0 |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dropout_neg_degrades_to_zero                                    |      0 |      1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_attn_bias_shapes                                                |      0 |      1 |       0 |        1 |
| tests/test_euclidean_dist.py                                     | test_euclidean_dist                                                                           |      0 |      1 |       0 |        1 |
| TOTAL                                                            |                                                                                               |    660 |      5 |      24 |      689 |
|                                filepath                                |                                function                                | passed | skipped | SUBTOTAL |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_conv2d.py                                                   | test_conv2d                                                            |      1 |       0 |        1 |
| tests/test_conv2d.py                                                   | test_conv2d_padding                                                    |      1 |       0 |        1 |
| tests/test_pad_circular.py                                             | test_pad_circular                                                      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss                                                    |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_backward                                           |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_backward_scalar_grad_output                        |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_backward_beta_zero_equal_inputs_cuda_behavior      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_broadcast                                          |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_special_values                                     |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_out                                                |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_out_reduced                                        |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_functional                                         |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_negative_beta                                      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                                           | test_smooth_l1_loss_backward_negative_beta                             |      1 |       0 |        1 |
| tests/test_softshrink_backward.py                                      | test_softshrink_backward                                               |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                                     | test_sparse_sampled_addmm                                              |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                                     | test_sparse_sampled_addmm_large                                        |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                                     | test_sparse_sampled_addmm_out                                          |      1 |       0 |        1 |
| tests/test_special_bessel_j1.py                                        | test_special_bessel_j1                                                 |      1 |       0 |        1 |
| tests/test_special_entr.py                                             | test_special_entr                                                      |      1 |       0 |        1 |
| tests/test_batch_norm.py                                               | test_native_batch_norm                                                 |      1 |       0 |        1 |
| tests/test_batch_norm.py                                               | test_batch_norm                                                        |      1 |       0 |        1 |
| tests/test_batch_norm.py                                               | test_batch_norm_backward                                               |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p                                                     |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p_non_tensor                                          |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p_out                                                 |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p_negative                                            |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p_nan_inf                                             |      1 |       0 |        1 |
| tests/test_special_log1p.py                                            | test_special_log1p_small_values                                        |      1 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py                               | test_special_modified_bessel_k1                                        |      1 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py                               | test_special_modified_bessel_k1_out                                    |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py                   | test_special_shifted_chebyshev_polynomial_u                            |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py                   | test_special_shifted_chebyshev_polynomial_u_scalar_n                   |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py                   | test_special_shifted_chebyshev_polynomial_u_                           |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_broadcast                                            |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_out                                                  |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_tensor_scalar                                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_tensor_scalar_out                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_scalar_tensor                                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_scalar_tensor_out                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_special_values                                       |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_numerical_regimes                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_negative_noninteger_q                                |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_integer_promotion                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_half_bfloat16_promotion                              |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_rejects_native_low_precision                         |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_out_exact_alias                                      |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_out_partial_overlap_rejected                         |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_out_casts_after_float_compute                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_q_gradient                                           |      1 |       0 |        1 |
| tests/test_special_zeta.py                                             | test_special_zeta_x_gradient_is_not_implemented                        |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                                 | test_rnn_relu                                                          |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                                 | test_rnn_relu_direct_wrapper                                           |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                                 | test_rnn_relu_direct_backward                                          |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                                 | test_rnn_relu_large_hidden                                             |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_tensor                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_scalar                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_tensor_                                                |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_scalar_                                                |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_scalar_tensor                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_scalar_scalar                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_complex                                               |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_float_tensor                                          |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_int_tensor                                            |      1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_int_scalar                                            |      1 |       0 |        1 |
| tests/test_sym_size.py                                                 | test_sym_size                                                          |      1 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh                                                              |      1 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh_                                                             |      1 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh_backward                                                     |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_rejects_unsupported_dtype           |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward                                     |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_optional_grads                      |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_noncontiguous                       |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_end_to_end                          |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_reshaped_workspace                  |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_empty                               |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_special_values                      |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py                            | test_thnn_fused_lstm_cell_backward_requires_no_grad                    |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy_complex                                   |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy                                           |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy_scalar                                    |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy_non_contiguous                            |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy_same_dim_does_not_alias                   |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_transpose_copy_invalid_dims                                       |      1 |       0 |        1 |
| tests/test_transpose_copy.py                                           | test_accuracy_transpose_copy_float8                                    |      1 |       0 |        1 |
| tests/test_triu.py                                                     | test_triu                                                              |      1 |       0 |        1 |
| tests/test_triu.py                                                     | test_triu_noncontiguous                                                |      1 |       0 |        1 |
| tests/test_triu.py                                                     | test_triu_                                                             |      1 |       0 |        1 |
| tests/test_triu.py                                                     | test_triu_inplace_noncontiguous                                        |      1 |       0 |        1 |
| tests/test_unfold.py                                                   | test_unfold                                                            |      1 |       0 |        1 |
| tests/test_unique_dim.py                                               | test_unique_dim_int                                                    |      1 |       0 |        1 |
| tests/test_unique_dim.py                                               | test_unique_dim_float                                                  |      1 |       0 |        1 |
| tests/test_unique_dim.py                                               | test_unique_dim_returns_three_tensors                                  |      1 |       0 |        1 |
| tests/test_conv1d.py                                                   | test_conv1d                                                            |      1 |       0 |        1 |
| tests/test_conv1d.py                                                   | test_conv1d_padding                                                    |      1 |       0 |        1 |
| tests/test_conv1d.py                                                   | test_conv1d_dilation                                                   |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward                                      |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_grad_input                           |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_edges                                |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_grad_input_resize                    |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_grad_input_errors                    |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_invalid                              |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_scale_contract                       |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_deterministic                        |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_clamped_scale_cpu_reference          |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_grad_input_channels_last             |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_channels_last_cache_reuse            |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_grad_input_channels_last_cache_reuse |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_fp64_opmath                          |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_nonfinite_cpu_reference              |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_extreme_positive_scales              |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod                                                    |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_int_bool                                           |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_sequence_form                                      |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_non_contiguous                                     |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_row_order                                          |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_backward                                           |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_nan_inf                                            |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_rejects_empty_list                                 |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_rejects_multidim_input                             |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_rejects_mixed_device                               |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_rejects_mixed_dtype                                |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_rejects_non_tensor                                 |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                                           | test_cartesian_prod_backward_mixed_requires_grad                       |      1 |       0 |        1 |
| tests/test_upsample_trilinear3d_backward.py                            | test_upsample_trilinear3d_backward                                     |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_fresh_tensor                                              |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_bumps_on_inplace_ops                                      |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_inplace_under_no_grad                                     |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_views_share_base_counter                                  |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_noncontiguous_and_strided_views                           |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_dtype_sweep                                               |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_sparse_layouts                                            |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_inference_mode_raises                                     |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_meta_tensor                                               |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_matches_reference_table                                   |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_dispatch_sentinel_composite_key                           |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_disjoint_from_native_neighbors                            |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_non_tensor_input                                          |      1 |       0 |        1 |
| tests/test_version.py                                                  | test_version_bump_sequence_matches_native                              |      1 |       0 |        1 |
| tests/test_weight_int4pack_mm_with_scales_and_zeros.py                 | test_weight_int4pack_mm_with_scales_and_zeros                          |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy                                                             |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_out                                                         |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_special_values                                              |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_tensor_scalar                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_tensor_scalar_out                                           |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_scalar_tensor                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                                                    | test_xlogy_scalar_tensor_out                                           |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                                           | test_frobenius_norm                                                    |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                                           | test_frobenius_norm_zero_size                                          |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                                           | test_frobenius_norm_duplicate_dim                                      |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_uses_flag_gems_view_path                                     |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_complex_view_semantics                                       |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_noncontiguous_complex_view                                   |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_non_complex_returns_input                                    |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_float8_returns_input                                         |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_autograd_view_relation                                       |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_conjugate_view_version_tracking                              |      1 |       0 |        1 |
| tests/test_real.py                                                     | test_real_conjugate_nonleaf_inplace_view_replay                        |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve                                                    |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_larger_shapes                                      |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_rhs_boundaries                                     |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_upper                                              |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_transpose_contiguous_factor                        |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_blocked_single_rhs                                 |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_small_gather_single_rhs                            |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_fp64_blocked                                       |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_fp32_blocked_upper                                 |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_fp32_blocked_lower                                 |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_batch                                              |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_broadcast_batch                                    |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_noncontiguous_inputs                               |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_scaled_inputs                                      |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_conditioned_matrix                                 |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_accuracy                                           |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_direct                                             |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex                                            |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_noncontiguous                              |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_broadcast                                  |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_small_gather_conditioned                   |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_blocked                                    |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_blocked_conditioned                        |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_lazy_conjugate_upper                       |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_complex_lazy_conjugate_with_all_gems               |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out                                                |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out_resizes_for_broadcast_result                   |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out_noncontiguous_and_alias                        |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out_safe_dtype_cast                                |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out_rejects_incompatible_dtype                     |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_out_rejects_different_device                       |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_empty_input                                        |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_invalid_inputs                                     |      1 |       0 |        1 |
| tests/test_masked_fill.py                                              | test_masked_fill                                                       |      1 |       0 |        1 |
| tests/test_masked_fill.py                                              | test_masked_fill_                                                      |      1 |       0 |        1 |
| tests/test_masked_fill.py                                              | test_masked_fill_scalar                                                |      1 |       0 |        1 |
| tests/test_masked_fill.py                                              | test_masked_fill_scalar_                                               |      1 |       0 |        1 |
| tests/test_fractional_max_pool3d_backward.py                           | test_accuracy_fractional_max_pool3d_backward                           |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_operator_markers_are_declared                                     |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_float                                      |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_int                                        |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_bool                                       |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_complex                                    |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_rejects_unsupported_dtype                  |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_list_lengths                               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_mixed_shapes                               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_empty_and_zero_dim                         |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_mixed_dtypes                               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_layouts                                    |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_inplace                                    |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_inplace_writes_through_views               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_inplace_rejects_overlap                    |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_inplace_rejects_promotion                  |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_error_parity                               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_out_overload                               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_foreach_unary_registration_is_live                                |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_foreach_unary_launch_count_is_independent_of_list_length          |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_foreach_unary_launch_count_scales_with_dtype_groups               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_foreach_unary_launch_count_with_heterogeneous_numel               |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_unary_complex_large                              |      1 |       0 |        1 |
| tests/test_foreach_unary.py                                            | test_accuracy_foreach_abs_complex_large                                |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py      | test_fake_quantize_learnable_per_channel_affine_backward               |      1 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_channel_affine_backward.py      | test_fake_quantize_learnable_per_channel_affine_backward_extreme       |      1 |       0 |        1 |
| tests/test_zeros.py                                                    | test_zeros                                                             |      1 |       0 |        1 |
| tests/test_batch_norm_backward_elemt.py                                | test_batch_norm_backward_elemt                                         |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_tensor                                                 |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_tensor_                                                |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_float                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_int                                               |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_float_                                            |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_mode_int_                                              |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_float                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_int                                               |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_float_                                            |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_mode_int_                                              |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_scalar                                                 |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_scalar_                                                |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_                                                       |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_tensor                                                 |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_scalar_scalar                                                 |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_complex                                               |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_float_tensor                                          |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_tensor_int                                                    |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_complex_int_scalar                                            |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_tensor_tensor                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_tensor_scalar                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_out_scalar_tensor                                             |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_tensor                                                   |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_scalar                                                   |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_tensor_                                                  |      1 |       0 |        1 |
| tests/test_div.py                                                      | test_div_mode_scalar_                                                  |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_tensor_qparams_rounding                             |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_tensor_qparams_extreme_values                       |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams          |      1 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_disabled |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet                                                           |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_complex                                                   |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_near_singular_scaled                                      |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_scaled_identity                                           |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_singular                                                  |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_1x1                                                       |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_empty_matrix                                              |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_empty_batch                                               |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_multi_dim_batch                                           |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_non_contiguous                                            |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_nan_inf                                                   |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_large                                                     |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_large_1025                                                |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_invalid_inputs                                            |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_backward                                                  |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_batched_backward                                          |      1 |       0 |        1 |
| tests/test_slogdet.py                                                  | test_slogdet_identity                                                  |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_tensor                                                       |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_scalar                                                       |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_tensor_inplace                                               |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_scalar_inplace                                               |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_scalar_byte                                                  |      1 |       0 |        1 |
| tests/test_fmod.py                                                     | test_fmod_scalar_inplace_byte                                          |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy                                     |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_dtype_mapping                       |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_values                              |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_noncontiguous_batch                 |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_transposed                          |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_high_rank_noncontiguous             |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_expanded                            |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_storage_offset                      |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_independent_storage                 |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_empty                               |      1 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                     | test_accuracy_view_as_complex_copy_error_paths                         |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH                                                                |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH_non_contiguous                                                 |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH_complex                                                        |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH_write_through_view                                             |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH_1d_raises                                                      |      1 |       0 |        1 |
| tests/test_mH.py                                                       | test_mH_0d_warns_and_conjugates                                        |      1 |       0 |        1 |
| tests/test_mrope.py                                                    | test_mrope_vs_torch                                                    |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_real_dtypes                                               |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_complex_dtypes                                            |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_0d                                                        |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_1d_matches_native_error                                   |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_non_contiguous                                            |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_mutation_through_view                                     |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_conjugate_bit_input                                       |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_dispatch_stability                                        |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_matches_native_view_table                                 |      1 |       0 |        1 |
| tests/test_adjoint.py                                                  | test_adjoint_diagnostics_parity                                        |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange                                                            |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange_start_dtype_inference                                      |      1 |       0 |        1 |
| tests/test_arange_start.py                                             | test_arange_start_float_args_follow_default_dtype                      |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index                                                             |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_nonleading_adjacent_tensor_indices                          |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_with_none_basic_indexing                                    |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_with_none_and_tensor                                        |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_boolean_mask                                                |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_empty_tensor                                                |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_1d_special_case                                             |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_error_empty_indices                                         |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_error_too_many_indices                                      |      1 |       0 |        1 |
| tests/test_index.py                                                    | test_index_int32_offset_overflow                                       |      1 |       0 |        1 |
| tests/test_norm.py                                                     | test_norm                                                              |      1 |       0 |        1 |
| tests/test_norm.py                                                     | test_norm_scalar                                                       |      1 |       0 |        1 |
| tests/test_norm.py                                                     | test_norm_scalaropt_dim                                                |      1 |       0 |        1 |
| tests/test_sum.py                                                      | test_sum                                                               |      1 |       0 |        1 |
| tests/test_sum.py                                                      | test_sum_dim                                                           |      1 |       0 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_out                                                       |      1 |       0 |        1 |
| tests/test_sum.py                                                      | test_sum_out                                                           |      1 |       0 |        1 |
| tests/test_gelu.py                                                     | test_gelu                                                              |      1 |       0 |        1 |
| tests/test_gelu.py                                                     | test_gelu_backward                                                     |      1 |       0 |        1 |
| tests/test_gelu.py                                                     | test_gelu_                                                             |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__basic                                              |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__options                                            |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__grad_scale                                         |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__found_inf                                          |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__empty                                              |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__mixed_empty_and_nonempty                           |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__mismatched_list_lengths                            |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__mismatched_tensor_properties                       |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__later_entry_bad_dtype_or_device                    |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__later_entry_bad_state_step                         |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__multi_tensor                                       |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__transposed_parity                                  |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                                           | test_fused_adagrad__internal_overlap_rejected                          |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                           | test_upsample_lanczos2d_aa_backward                                    |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                           | test_upsample_lanczos2d_aa_backward_dtypes                             |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                           | test_upsample_lanczos2d_aa_backward_precomputed_path                   |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                           | test_upsample_lanczos2d_aa_backward_out                                |      1 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                           | test_upsample_lanczos2d_aa_backward_noncontiguous_grad                 |      1 |       0 |        1 |
| tests/test_topk_softmax.py                                             | test_topk_softmax                                                      |      0 |       1 |        1 |
| tests/test_upsample_bilinear2d_backward.py                             | test_upsample_bilinear2d_backward_unsupported_fp64                     |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_ascend_batched_small_lower_single_rhs              |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                                           | test_cholesky_solve_ascend_blocked_single_rhs_conditioned              |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_inplace_boundaries_exact                                     |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_overwrites_special_values_exact                              |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_noncontiguous_storage_exact                                  |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_repeated_calls_exact                                         |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zeros_factory_poisoned_allocation_exact                           |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_launch_core_dim_limit                                        |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_real_large_input_exact                                       |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zero_capped_grid_covers_all_tiles_exact                           |      0 |       1 |        1 |
| tests/test_zeros.py                                                    | test_zeros_factory_capped_grid_exact                                   |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8                                                       |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_prequantized                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_large_k_scales                               |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_skinny_epilogue                              |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_dense_reuse                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_graph_updates                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_empty                                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_output_dtype                                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_independent_and_invalid_out                           |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_broadcast                                    |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_invalid_inputs_and_scales                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_scalar_shapes                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_scaled_mm_interface                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_default_dtype                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_invalid_epilogue                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_mthreads_strided_epilogue                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_hopper_external_scales                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_hopper_torch_interface                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                                              | test_mm_w8a8_fp8_hopper_torch_rowwise                                  |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_full_reduction_fp16_partial_overflow                      |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_large_ascend_reduction_tile                                   |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_ascend_grid_stride                                            |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_launch_grid_limit                                             |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_empty_output_does_not_launch                              |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_real_core_dim_limit                                       |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_forced_grid_stride                                        |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_grid_stride_noncontiguous_dtype                           |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_broadcast_gradient_core_dim                               |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_full_reduction_forced_grid_stride                             |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_dim_empty_review                                              |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_empty_review                                                  |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_out_noncontiguous_review                                      |      0 |       1 |        1 |
| tests/test_sum.py                                                      | test_sum_partial_precision_review                                      |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_ascend_config                                                |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_tile_boundaries                                              |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_finite_edge_accuracy                                         |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_layout_and_storage                                           |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_public_api_accuracy                                          |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_forward_backward_against_autograd                            |      0 |       1 |        1 |
| tests/test_gelu.py                                                     | test_gelu_more_than_65535_tiles                                        |      0 |       1 |        1 |
| TOTAL                                                                  |                                                                        |    345 |      53 |      398 |
