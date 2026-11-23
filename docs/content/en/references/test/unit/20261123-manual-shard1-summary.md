|                        filepath                         |                                           function                                            | passed | failed | skipped | SUBTOTAL |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/core/test_dynamic_override.py                     | test_override_call_count_snapshots_and_all_skip_restore                                       |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_basic_override                                                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_multiple_overrides                                                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_restore_single_op                                                  |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_restore_all                                                        |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_context_manager                                                    |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_list_overrides                                                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_override_from_file                                                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_override_batch_from_files                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_override_nonexistent_file                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_override_missing_function                                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_restore_without_override                                           |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_operator_with_kwargs                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_concurrent_registries                                              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestDynamicOpOverride.test_override_special_ops                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_parse_override_spec_colon                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_parse_override_spec_equals                                               |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_parse_override_spec_invalid                                              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_load_override_config_yaml                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_load_override_config_json                                                |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestCLIOverride.test_load_override_config_missing_file                                        |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestIntegrationScenarios.test_ab_testing_scenario                                             |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestIntegrationScenarios.test_progressive_override                                            |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestPytestIntegration.test_override_option_registered                                         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestPytestIntegration.test_override_applied_through_cli                                       |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestPytestIntegration.test_override_config_file_applied_through_cli                           |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestPytestIntegration.test_no_override_leaves_default_implementation                          |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestPytestIntegration.test_missing_operator_is_available_during_collection                    |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_pytest_parser_accepts_the_plugin                     |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_missing_candidate_aborts                             |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_missing_run_aborts                                   |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_unknown_unused_operator_fails_and_is_removed         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_existing_registration_uses_candidate                 |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_unused_candidate_makes_pytest_fail                   |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestRegistrarLiveOverrideResolution.test_direct_call_and_restore_work                         |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestUnimplementedOperatorOverride.test_repeated_override_restores_original_state              |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestUnimplementedOperatorOverride.test_exception_restores_existing_and_removes_added_operator |      1 |      0 |       0 |        1 |
| tests/core/test_dynamic_override.py                     | TestUnimplementedOperatorOverride.test_file_candidate_for_missing_operator                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_computes_requested_metrics_and_preserves_duplicate_shapes                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_reads_v2_rows_and_rounds_only_serialized_metrics                                         |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_private_rows_map_platform_key_to_public_model_column                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_v3_requires_identical_benchmark_protocols                                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_v3_protocol_is_preserved_in_comparison_output                                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_rejects_ambiguous_or_incompatible_source_rows                                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_keeps_invalid_measurements_but_omits_derived_metrics                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_cli_writes_baseline_and_ours_columns                                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_rejects_cross_platform_comparison                                                        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_comparison_requires_platform_identity                                                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_non_mm_dimensions_are_preserved_in_csv_and_jsonl                                         |      1 |      0 |       0 |        1 |
| tests/flagtune/test_compare_pretune.py                  | test_rejects_different_workload_dimension_schemas                                             |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_stable_state_skips_mode_resolution_and_device_changes_recheck                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_expired_auto_failure_invalidates_dispatch_even_after_idle                                |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_environment_changes_recheck_and_removal_restores_state                                   |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_changed_mode_clears_compiled_entries                                                     |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_failed_resolution_is_not_cached                                                          |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_api_include_state_and_registry_changes_invalidate                                        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_live_mode_switches_preserve_default_cost_and_expanded                                    |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_invalid_live_setting_raises_until_fixed                                                  |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_explicit_default_ignores_irrelevant_model_and_include_changes                            |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_model_source_is_not_a_mode_selection_dependency                                          |      1 |      0 |       0 |        1 |
| tests/flagtune/test_dispatch_state_cache.py             | test_auto_fuse_invalidates_all_entries_for_the_same_model                                     |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py | test_fused_recurrent_gated_delta_rule_fwd_accuracy                                            |      1 |      0 |       0 |        1 |
| tests/test__linalg_slogdet.py                           | test__linalg_slogdet                                                                          |      1 |      0 |       0 |        1 |
| tests/test__thnn_differentiable_gru_cell_backward.py    | test_thnn_differentiable_gru_cell_backward                                                    |      1 |      0 |       0 |        1 |
| tests/test__thnn_differentiable_gru_cell_backward.py    | test_thnn_differentiable_gru_cell_backward_noncontiguous                                      |      1 |      0 |       0 |        1 |
| tests/test__thnn_differentiable_gru_cell_backward.py    | test_thnn_differentiable_gru_cell_backward_empty                                              |      1 |      0 |       0 |        1 |
| tests/test__thnn_differentiable_gru_cell_backward.py    | test_thnn_differentiable_gru_cell_backward_invalid_shape                                      |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                       | test_accuracy_adaptive_avg_pool2d_forward                                                     |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                       | test_accuracy_adaptive_avg_pool2d_boundary_layout                                             |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                       | test_accuracy_adaptive_avg_pool2d_out                                                         |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                       | test_accuracy_adaptive_avg_pool2d_out_layout                                                  |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                       | test_accuracy_adaptive_avg_pool2d_out_resizes                                                 |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                    | test_accuracy_addbmm                                                                          |      1 |      0 |       0 |        1 |
| tests/test_addmv_.py                                    | test_addmv_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py      | test_native_batch_norm_legit_no_training                                                      |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                                   | test_arccosh                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                                   | test_arccosh_special_values                                                                   |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                                   | test_arccosh_out                                                                              |      1 |      0 |       0 |        1 |
| tests/test_arm_w4a8_g128.py                             | test_w4a8_g128_packer_rejects_invalid_contracts                                               |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                  | test_attention_mask_broadcast                                                                 |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                  | test_attention_mask_broadcast_graph                                                           |      1 |      0 |       0 |        1 |
| tests/test_attention_mask_broadcast.py                  | test_attention_mask_broadcast_gqa                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod                                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_ignores_unstored_zeros                                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_stored_zero_wins                                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_empty_input                                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_dtype                                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_unsorted_columns                                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_empty_rows_and_columns                                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_out                                                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                           | test_sparse_csr_prod_rejects_bad_request                                                      |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update                                                                   |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update_non_contiguous_params                                             |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update_requires_running_stats                                            |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update_backward                                                          |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update_param_size_mismatch                                               |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_with_update.py                    | test_batch_norm_with_update_empty_dims                                                        |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits.py          | test_binary_cross_entropy_with_logits                                                         |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits.py          | test_binary_cross_entropy_with_logits_mean                                                    |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits.py          | test_binary_cross_entropy_with_logits_sum                                                     |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits.py          | test_binary_cross_entropy_with_logits_weight                                                  |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits.py          | test_binary_cross_entropy_with_logits_pos_weight                                              |      1 |      0 |       0 |        1 |
| tests/test_bitwise_or_scalar_tensor.py                  | test_bitwise_or_scalar_tensor                                                                 |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                        | test_as_strided_scatter                                                                       |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                        | test_as_strided_scatter_preserves_storage_geometry                                            |      1 |      0 |       0 |        1 |
| tests/test_as_strided_scatter.py                        | test_as_strided_scatter_noncontiguous_self                                                    |      1 |      0 |       0 |        1 |
| tests/test_ceil.py                                      | test_ceil                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_ceil.py                                      | test_ceil_                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_ceil.py                                      | test_ceil_out                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_conv_depthwise2d.py                          | test_conv_depthwise2d                                                                         |      1 |      0 |       0 |        1 |
| tests/test_col2im.py                                    | test_col2im                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                           | test_constant_pad_nd                                                                          |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                           | test_constant_pad_nd_non_contiguous                                                           |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                           | test_constant_pad_nd_zero_value                                                               |      1 |      0 |       0 |        1 |
| tests/test_constant_pad_nd.py                           | test_constant_pad_nd_partial_dims                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_infer_size                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_conversion_and_strides                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_scalar_values                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_invalid_inputs                                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_coalesced                                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_dispatch                                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_registration_scope                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_mismatched_devices                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_autograd                                                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_pin_memory                                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_pin_memory_rejection                                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_large_reduction                                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_size_overflow                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_cpu_transfer                                                           |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_invariant_check                                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_explicit_size                                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_explicit_validation                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_explicit_pin_memory                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_nonempty                                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_empty                                                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_hybrid_input                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_uncoalesced_input                                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_keeps_old_buffers_intact                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_size_mismatch_raises                                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_non_coo_rejected                                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_and_clear_.py                  | test_sparse_resize_and_clear_leaf_requires_grad_rejected                                      |      1 |      0 |       0 |        1 |
| tests/test_cosine_embedding_loss.py                     | test_cosine_embedding_loss                                                                    |      1 |      0 |       0 |        1 |
| tests/test_elu_backward_out.py                          | test_elu_backward_out                                                                         |      1 |      0 |       0 |        1 |
| tests/test_elu_backward_out.py                          | test_elu_backward_out_noncontiguous                                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_rnn_backward.py                        | test_cudnn_rnn_backward                                                                       |      1 |      0 |       0 |        1 |
| tests/test_cumulative_trapezoid.py                      | test_cumulative_trapezoid                                                                     |      1 |      0 |       0 |        1 |
| tests/test_cumulative_trapezoid.py                      | test_cumulative_trapezoid_large                                                               |      1 |      0 |       0 |        1 |
| tests/test_cumulative_trapezoid.py                      | test_cumulative_trapezoid_dim                                                                 |      1 |      0 |       0 |        1 |
| tests/test_cumulative_trapezoid.py                      | test_cumulative_trapezoid_dx                                                                  |      1 |      0 |       0 |        1 |
| tests/test_dense_dim.py                                 | test_dense_dim_strided                                                                        |      1 |      0 |       0 |        1 |
| tests/test_dense_dim.py                                 | test_dense_dim_sparse_coo                                                                     |      1 |      0 |       0 |        1 |
| tests/test_dense_dim.py                                 | test_dense_dim_sparse_compressed                                                              |      1 |      0 |       0 |        1 |
| tests/test_dense_dim.py                                 | test_dense_dim_matches_reference_table                                                        |      1 |      0 |       0 |        1 |
| tests/test_dense_dim.py                                 | test_dense_dim_disjoint_from_native_neighbors                                                 |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter                                                                         |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter_offset                                                                  |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter_negative_offset                                                         |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter_non_last_dims                                                           |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter_large_offset                                                            |      1 |      0 |       0 |        1 |
| tests/test_diagonal_scatter.py                          | test_diagonal_scatter_non_square                                                              |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_accuracy                                                                            |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_broadcast                                                                           |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_non_contiguous                                                                      |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_negative_p                                                                          |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_fp64                                                                                |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_empty                                                                               |      1 |      0 |       0 |        1 |
| tests/test_dist.py                                      | test_dist_empty_raises                                                                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor                                                                      |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_views                                                                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_invalid_size                                                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_negative_dimension_message                                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_pin_memory_ignored                                                   |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_non_strided_layout_rejected                                          |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dtype_matrix                                                     |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_resizes_mismatched_out                                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_non_contiguous_gapped                                            |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_transposed_view                                                  |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_complex_gapped                                                   |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_zero_size                                                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_cpu                                                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_cpu_non_contiguous                                               |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_zero_tensor_rejected                                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_requires_grad_rejected                                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_meta                                                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_negative_size                                                    |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_registered_dispatch_path                                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_registered_dispatch_path                                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dispatch_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dispatch_complex                                                 |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dispatch_complex_non_contiguous                                  |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dispatch_zero_tensor_rejected                                    |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_downstream_dispatch                                                  |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_inplace_view_rejected                                                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                       | test_efficientzerotensor_out_dispatch_resize_growth                                           |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward                                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_extreme                                             |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_non_contiguous                                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_out_non_contiguous                                  |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                       | test_special_erf                                                                              |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                       | test_erf                                                                                      |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                       | test_erf_                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze_                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__int_dtypes                                                                      |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__non_contiguous                                                                  |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__expanded_stride_zero                                                            |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__storage_offset                                                                  |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__rejects_out_of_range_dim                                                        |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__rejects_out_of_range_dims                                                       |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__rejects_duplicate_dims                                                          |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__rejects_inplace_on_grad_leaf                                                    |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__backward                                                                        |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__dim                                                                             |      1 |      0 |       0 |        1 |
| tests/test_squeeze_.py                                  | test_squeeze__dims                                                                            |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout                                                                    |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_identity                                                           |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_p_one_special_values                                               |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_empty_returns_input                                                |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_noncontiguous                                                      |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_invalid_probability                                                |      1 |      0 |       0 |        1 |
| tests/test_feature_alpha_dropout.py                     | test_feature_alpha_dropout_requires_two_dimensions                                            |      1 |      0 |       0 |        1 |
| tests/test_fliplr.py                                    | test_accuracy_fliplr                                                                          |      1 |      0 |       0 |        1 |
| tests/test_fliplr.py                                    | test_accuracy_fliplr_non_float                                                                |      1 |      0 |       0 |        1 |
| tests/test_fliplr.py                                    | test_accuracy_fliplr_noncontiguous                                                            |      1 |      0 |       0 |        1 |
| tests/test_fliplr.py                                    | test_fliplr_rejects_inputs_with_fewer_than_two_dims                                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_fast_path                                                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_fallback_cases                                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_int_fallback                                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_writes_through_view                                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_rejects_empty_list                                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_fast_path                                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_half_to_even                                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_large_magnitude                                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_fallback_cases                                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_writes_through_view                                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_rejects_empty_list                                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_fast_path                                                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_fallback_cases                                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_int_fallback                                                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_writes_through_view                                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_rejects_empty_list                                                  |      1 |      0 |       0 |        1 |
| tests/test_functional_assert_async.py                   | test_functional_assert_async_pass                                                             |      1 |      0 |       0 |        1 |
| tests/test_functional_assert_async.py                   | test_functional_assert_async_float                                                            |      1 |      0 |       0 |        1 |
| tests/test_functional_assert_async.py                   | test_functional_assert_async_multi_element_error                                              |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_channelwise                                                        |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_groupwise                                                          |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_noncontiguous                                                      |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_rounding_and_nibbles                                               |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_invalid                                                            |      1 |      0 |       0 |        1 |
| tests/test_dyn_quant_matmul_4bit.py                     | test_dyn_quant_matmul_4bit_compile                                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                            | test_fused_rms_norm                                                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                            | test_fused_rms_norm_no_weight                                                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                            | test_fused_rms_norm_aten_dispatch                                                             |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                            | test_fused_rms_norm_backward                                                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_rms_norm.py                            | test_fused_rms_norm_backward_no_weight                                                        |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_3d.py                           | test_grid_sampler_3d                                                                          |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_3d.py                           | test_grid_sampler_3d_varied_output_sizes                                                      |      1 |      0 |       0 |        1 |
| tests/test_glu_jvp.py                                   | test_glu_jvp                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_glu_jvp.py                                   | test_glu_jvp_non_contiguous                                                                   |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double                                                                              |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_strided                                                                      |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_special_values                                                               |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_integer_limits                                                               |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_autograd                                                                     |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_complex                                                                      |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_cpu                                                                          |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                        | test_gt                                                                                       |      1 |      0 |       0 |        1 |
| tests/test_gt.py                                        | test_gt_scalar                                                                                |      1 |      0 |       0 |        1 |
| tests/test_hardsigmoid_backward.py                      | test_hardsigmoid_backward                                                                     |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_mixed                                                                       |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_scalar_tensor                                                               |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_float                                                                       |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_float_                                                                      |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_int                                                                         |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_int_                                                                        |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_scalar_scalar                                                               |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_scalar_inplace_float                                                        |      1 |      0 |       0 |        1 |
| tests/test_floor_divide.py                              | test_floor_divide_scalar_inplace_int                                                          |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                    | test_hsplit_int                                                                               |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                    | test_hsplit_array                                                                             |      1 |      0 |       0 |        1 |
| tests/test_hsplit.py                                    | test_hsplit_array_out_of_range                                                                |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_uses_platform_appropriate_lock                                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_flagtune_environment_controls_operator_selection                                         |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_tuning_mode_environment_matrix                                                           |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_tuning_mode_defers_missing_package_handling_to_policy                                    |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_legacy_fallback_honors_operator_include                                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_expanded_request_does_not_probe_platform_package                                         |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_adapted_operator_rejects_invalid_cost_model_setting                                      |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_disabled_flagtune_does_not_probe_platform_package                                        |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_adapted_tuning_mode_prefers_cost_model_when_both_enabled                                 |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_tuning_mode_rejects_non_binary_environment_values                                        |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_adapted_libtuner_switches_default_expanded_and_cost_model                                |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libtuner_apply_defers_model_availability_to_policy                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_official_triton_libtuner_uses_unadapted_routes                                           |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_decorator_cascade                                                                        |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_pass_kernel_arg_via_kw                                                                   |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_kernel_arg_apply_default                                                                 |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_threadsafety                                                                             |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_descriptor_cache_key_uses_resolved_type                                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_descriptor_cache_key_without_descriptor_module                                           |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_model_bundle_missing_error_supports_older_flagtree                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_key_without_vendor_hook                                                         |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_key_with_hygon_backend                                                          |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_key_normalizes_descriptor_arguments                                             |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_run_normalizes_keyword_descriptors                                              |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_key_orders_specialization_then_dns_then_constexpr                               |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_key_buckets_integers_by_triton_width                                            |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_launch_options_key_selects_compile_options                                      |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libentry_run_separates_compile_options                                                   |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_hash_generation                                                                          |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_hash_changes_when_dependency_modified                                                    |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_flagtree_policy_is_bypassed_when_expanded_flagtune_is_enabled                            |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_official_triton_treats_model_annotation_as_unadapted                                     |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_operator_fuse_invalidates_existing_dispatch_caches                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_legacy_fallback_prunes_and_restores_tuner_state                                          |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_flagtree_policy_is_bypassed_when_cost_model_is_disabled                                  |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_benchmark_success_count_tracks_finite_uncached_benchmarks                                |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_benchmark_key_preserves_raw_shape_and_scopes_timing_protocol                             |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_ordinary_libtuner_keeps_legacy_benchmark_arguments                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_cost_model_protocol_stays_explicit                                                       |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_replay_fallback_reuses_event_config_cache_namespace                                      |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_config_cache_namespace_separates_replay_protocols                                        |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_adapted_config_cache_namespace_separates_tuning_modes                                    |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_benchmark_config_reuses_kernel_context_and_bypasses_caches                               |      1 |      0 |       0 |        1 |
| tests/test_libentry.py                                  | test_libcache_vllm_signal_scenario                                                            |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo                                                          |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_flags                                                    |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_empty                                                    |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_strided_inputs                                           |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_zero_dim                                         |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_rank_mismatch                                    |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_batch_mismatch                                   |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_unsupported_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_positional_flags                                 |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_invalid_flag_type                                |      1 |      0 |       0 |        1 |
| tests/test_convert_indices_from_csr_to_coo.py           | test_convert_indices_from_csr_to_coo_rejects_out_dtype_mismatch                               |      1 |      0 |       0 |        1 |
| tests/test_is_all_true.py                               | test_is_all_true                                                                              |      1 |      0 |       0 |        1 |
| tests/test_is_all_true.py                               | test_accuracy_is_all_true_empty                                                               |      1 |      0 |       0 |        1 |
| tests/test_isnan.py                                     | test_isnan                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_gt_tensor_.py                                | test_gt_tensor_                                                                               |      1 |      0 |       0 |        1 |
| tests/test_gt_tensor_.py                                | test_gt_tensor__byte                                                                          |      1 |      0 |       0 |        1 |
| tests/test_igamma_.py                                   | test_igamma_                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv                                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_empty                                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_noncontiguous                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_round_trip                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_row_swap_pivoting                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_batch_independence                                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_binds_single_argument                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_invalid_shape                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_rejects_low_precision                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_singular_does_not_hang_or_crash                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_rejects_complex                                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_autograd_input_is_detached                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                              | test_linalg_cross                                                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                              | test_linalg_cross_noncontiguous_input_and_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                              | test_linalg_cross_rejects_different_input_ranks                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_cross.py                              | test_linalg_cross_conjugated_view                                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_batch                                                                           |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_broadcast                                                                       |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_out                                                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_non_contiguous                                                                  |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_out_non_contiguous                                                              |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_dtype_mismatch                                                                  |      1 |      0 |       0 |        1 |
| tests/test_lu_solve.py                                  | test_lu_solve_pivots_dtype                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_2d                                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_3d                                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_broadcast                                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_zero_sized                                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_mixed_dtype                                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_complex                                                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_grad                                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv                                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_batched                                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_large                                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_large_batched                                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_out                                                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_rejects_complex                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_out_resizes_and_warns                                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_out_rejects_unsafe_cast                                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_pinv.py                               | test_linalg_pinv_out_noncontiguous                                                            |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_out                                                                                |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_special_values                                                                     |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_empty                                                                              |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_noncontiguous                                                                      |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_int_promotes_to_float                                                              |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_backward                                                                           |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_backward_expanded_grad                                                             |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                     | test_log10_backward_special_values                                                            |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                               | test_log_softmax                                                                              |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                               | test_accuracy_log_softmax_out                                                                 |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                               | test_log_softmax_backward_data                                                                |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                               | test_accuracy_log_softmax_backward_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_logical_xor.py                               | test_logical_xor                                                                              |      1 |      0 |       0 |        1 |
| tests/test_mixed_dtypes_linear.py                       | test_mixed_dtypes_linear                                                                      |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose3d.py                          | test_conv_transpose3d                                                                         |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose3d.py                          | test_conv_transpose3d_bias                                                                    |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose3d.py                          | test_conv_transpose3d_dilation                                                                |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose3d.py                          | test_conv_transpose3d_groups                                                                  |      1 |      0 |       0 |        1 |
| tests/test_conv_transpose3d.py                          | test_conv_transpose3d_output_padding                                                          |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                           | test_matmul_backward                                                                          |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                           | test_matmul_backward_self_only                                                                |      1 |      0 |       0 |        1 |
| tests/test_matmul_backward.py                           | test_matmul_backward_other_only                                                               |      1 |      0 |       0 |        1 |
| tests/test_max_pool2d.py                                | test_max_pool2d_with_indices                                                                  |      1 |      0 |       0 |        1 |
| tests/test_max_pool2d.py                                | test_max_pool2d_backward                                                                      |      1 |      0 |       0 |        1 |
| tests/test_max_pool2d.py                                | test_max_pool2d_with_indices_backward                                                         |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim                                                                               |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_keepdim_shapes                                                                |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_ties                                                                          |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_strided_nonlast_reduction                                                     |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_even_count_picks_lower_middle                                                 |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_propagates_nan                                                                |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_scalar_input                                                                  |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_empty_reduction_dim_raises                                                    |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_empty_but_nonzero_reduction                                                   |      1 |      0 |       0 |        1 |
| tests/test_median_dim.py                                | test_median_dim_named_dim_preserves_names                                                     |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                      | test_mish                                                                                     |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                      | test_mish_                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_mish.py                                      | test_mish_backward                                                                            |      1 |      0 |       0 |        1 |
| tests/test_moe_sum.py                                   | test_moe_sum                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward                                                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_empty_batch                                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_noncontiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_nan_and_inf                                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_watchdog_path                                 |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_full_target                                   |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_duplicate                                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_duplicate_tail                                |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_empty_prefix                                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_large_class_bucket_boundaries                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_many_rows_target_length_reduce                            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_invalid_target_values_raise                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_invalid_after_distant_sentinel_raises                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_invalid_target_metadata_raises                            |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss_forward.py            | test_multilabel_margin_loss_forward_invalid_input_shape_raises                                |      1 |      0 |       0 |        1 |
| tests/test_nan_to_num_.py                               | test_nan_to_num_                                                                              |      1 |      0 |       0 |        1 |
| tests/test_native_channel_shuffle.py                    | test_native_channel_shuffle                                                                   |      1 |      0 |       0 |        1 |
| tests/test_negative_.py                                 | test_negative_                                                                                |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros                                                                                |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_inherits_metadata                                                              |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_source_is_not_read_or_mutated                                                  |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_dtype_override                                                                 |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_explicit_device                                                                |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_sparse_layout                                                                  |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_size_argument_forms                                                            |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_bool_extent                                                                    |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_out_resizes_buffer                                                             |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_out_writes_through_strided_buffer                                              |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_empty_source_still_supplies_defaults                                           |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_rejects_non_tensor_self                                                        |      1 |      0 |       0 |        1 |
| tests/test_new_zeros.py                                 | test_new_zeros_rejects_non_sequence_size                                                      |      1 |      0 |       0 |        1 |
| tests/test_nll_loss_forward.py                          | test_nll_loss_forward                                                                         |      1 |      0 |       0 |        1 |
| tests/test_normal_functional.py                         | test_normal_functional                                                                        |      1 |      0 |       0 |        1 |
| tests/test_or_scalar.py                                 | test_or_scalar                                                                                |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod                                                                           |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_int_bool                                                                  |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_sequence_form                                                             |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_non_contiguous                                                            |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_row_order                                                                 |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_backward                                                                  |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_nan_inf                                                                   |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_rejects_empty_list                                                        |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_rejects_multidim_input                                                    |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_rejects_mixed_device                                                      |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_rejects_mixed_dtype                                                       |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_rejects_non_tensor                                                        |      1 |      0 |       0 |        1 |
| tests/test_cartesian_prod.py                            | test_cartesian_prod_backward_mixed_requires_grad                                              |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist                                                                                    |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist_p1                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist_pinf                                                                               |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist_p_general                                                                          |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist_p0                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_pdist.py                                     | test_pdist_p_large                                                                            |      1 |      0 |       0 |        1 |
| tests/test_pixel_shuffle.py                             | test_pixel_shuffle                                                                            |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                                 | test_polygamma                                                                                |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                                 | test_polygamma_wide_domain                                                                    |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                                 | test_polygamma_out                                                                            |      1 |      0 |       0 |        1 |
| tests/test_polygamma.py                                 | test_polygamma_                                                                               |      1 |      0 |       0 |        1 |
| tests/test_quantile.py                                  | test_quantile                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_quantile.py                                  | test_quantile_dim                                                                             |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d                                                                     |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_channels_last                                                       |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_4d_output_is_channels_last                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_matches_native_cuda                                                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_3d_output_stays_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_zero_batch                                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_extreme_values                                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_padding_sentinel                                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_default_stride                                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_runs_triton_kernel                                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out                                                                 |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_resizes                                                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_preserves_offset_and_strides                                    |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_channels_last_out                                               |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_aliasing_input                                                  |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_mismatched_qparams                                              |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_rejects_per_channel_out                                         |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_out_rejects_mismatched_out                                          |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_invalid_params                                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_invalid_shapes                                                      |      1 |      0 |       0 |        1 |
| tests/test_quantized_max_pool2d.py                      | test_quantized_max_pool2d_rejects_per_channel_input                                           |      1 |      0 |       0 |        1 |
| tests/test_randn_like.py                                | test_randn_like                                                                               |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad1d_backward.py                 | test_reflection_pad1d_backward                                                                |      1 |      0 |       0 |        1 |
| tests/test_relu6_.py                                    | test_relu6_                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_replication_pad2d.py                         | test_replication_pad2d                                                                        |      1 |      0 |       0 |        1 |
| tests/test_replication_pad2d.py                         | test_replication_pad2d_out                                                                    |      1 |      0 |       0 |        1 |
| tests/test_replication_pad2d.py                         | test_replication_pad2d_3d_input                                                               |      1 |      0 |       0 |        1 |
| tests/test_resize.py                                    | test_resize                                                                                   |      1 |      0 |       0 |        1 |
| tests/test_resize.py                                    | test_resize_                                                                                  |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat                                                                                 |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_int_bool                                                                        |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_element_order                                                                   |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_non_contiguous                                                                  |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_transposed                                                                      |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_nan_inf                                                                         |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_default_offset                                                                  |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_backward                                                                        |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_backward_sum                                                                    |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_backward_non_contiguous_grad                                                    |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_large_output_is_deterministic                                                   |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_rejects_non_int_offset                                                          |      1 |      0 |       0 |        1 |
| tests/test_rrelu_with_noise_functional.py               | test_rrelu_with_noise_functional                                                              |      1 |      0 |       0 |        1 |
| tests/test_scalar_tensor.py                             | test_scalar_tensor                                                                            |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                         | test_scaled_grouped_mm                                                                        |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                         | test_scaled_grouped_mm_fp8_core                                                               |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                         | test_scaled_grouped_mm_fp8_encodings                                                          |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                         | test_scaled_grouped_mm_scale_before_output_cast                                               |      1 |      0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                         | test_scaled_grouped_mm_reduction_addressing                                                   |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_lengths                                                                   |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_offsets                                                                   |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_backward_lengths                                                          |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_lengths_out                                                               |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_offsets_out                                                               |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_backward_lengths_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_backward_offsets_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_backward_offsets                                                          |      1 |      0 |       0 |        1 |
| tests/test_segment_reduce.py                            | test_segment_reduce_indices_unsupported                                                       |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_fused_recurrent_gated_delta_rule.py | test_fused_recurrent_gated_delta_rule_matches_vllm                                            |      0 |      0 |       1 |        1 |
| tests/test_arm_w4a8_g128.py                             | test_w4a8_g128_matches_reference                                                              |      0 |      0 |       1 |        1 |
| tests/test_arm_w4a8_g128.py                             | test_w4a8_g128_decode_is_repeatable                                                           |      0 |      0 |       1 |        1 |
| tests/test_sparse_coo_tensor.py                         | test_sparse_coo_tensor_device_index                                                           |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8                                                     |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_deepgemm                                            |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_int8_extremes                                       |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_pipeline                                            |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_changing_expert_rows                                |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_async                                               |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestPPUGroupedGEMM.test_grouped_gemm_w8a8_gemv_cache_tail                                     |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8                                                   |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_int8_extremes                                     |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_changing_expert_rows                              |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_dtypes_tails                                      |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_fp8_encodings                                     |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_int8_cancellation                                 |      0 |      0 |       1 |        1 |
| tests/test_grouped_gemm_w8a8.py                         | TestMetaXGroupedGEMM.test_grouped_gemm_w8a8_sparse_changing_groups                            |      0 |      0 |       1 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_fp8                                                       |      0 |      0 |       1 |        1 |
| tests/test_fp8_paged_mqa_logits_ops.py                  | test_accuracy_fp8_paged_mqa_logits                                                            |      0 |      0 |       1 |        1 |
| tests/test_fp8_paged_mqa_logits_ops.py                  | test_accuracy_fp8_paged_mqa_logits_param                                                      |      0 |      0 |       1 |        1 |
| tests/test_functional_assert_async.py                   | test_functional_assert_async_fail                                                             |      0 |      0 |       1 |        1 |
| tests/test_indexer_k_quant_and_cache.py                 | test_indexer_k_quant_and_cache_matches_reference                                              |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_flagtree_policy_uses_cost_model_by_default_for_adapted_operator                          |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_enabled_flagtree_policy_propagates_contract_failures                                     |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_cost_model_failures_fuse_only_auto                                                       |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_auto_source_failure_warns_once_retries_and_recovers                                      |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_cost_model_singleton_does_not_load_or_swallow_legacy_failure                             |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_flagtree_proposer_cache_tracks_resolved_model_version                                    |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_hopper_mm_config_compiles_without_runtime_registration                                   |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_mul_config_compiles_and_binds_runtime_kernels                                            |      0 |      0 |       1 |        1 |
| tests/test_libentry.py                                  | test_libcache_concurrent_write_on_signal                                                      |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_int8.py                       | test_rms_norm_w8a16_int8                                                                      |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_int8.py                       | test_rms_norm_w8a16_int8_ragged_shapes                                                        |      0 |      0 |       1 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_unit_dim                                            |      0 |      1 |       0 |        1 |
| tests/test_cudnn_affine_grid_generator_backward.py      | test_cudnn_affine_grid_generator_backward_out_dispatch                                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_value_range                                                       |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_wide_values                                                       |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_accumulate_matches                                                |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_stored_layout                                                     |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_hybrid_coo                                                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_strided_storage                                                   |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_special_values                                                    |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_out                                                               |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_out_stored_layout                                                 |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_dense_self                                                |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_dense_mask                                                |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_size_mismatch                                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_sparse_dim_mismatch                                       |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_invalid_accumulate                                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_rejects_dense_out                                                 |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                   | test_sparse_mask_projection_dispatch                                                          |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_abs_inplace_registration_is_live                                                 |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_round_inplace_registration_is_live                                               |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                    | test_foreach_sign_inplace_registration_is_live                                                |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_negative_view                                                                |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                               | test_cast_Double_sparse                                                                       |      0 |      1 |       0 |        1 |
| tests/test_linalg_inv.py                                | test_linalg_inv_fp64_preserved                                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_matmul.py                             | test_linalg_matmul_double_grad                                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                            | test_linalg_svdvals                                                                           |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                            | test_linalg_svdvals_batch                                                                     |      0 |      1 |       0 |        1 |
| tests/test_linalg_svdvals.py                            | test_linalg_svdvals_non_contiguous                                                            |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_softmax_with_shape.py          | test_nested_tensor_softmax_with_shape                                                         |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_softmax_with_shape.py          | test_nested_tensor_softmax_with_shape_full_length                                             |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                  | test_diagflat_backward_complex_sum                                                            |      0 |      1 |       0 |        1 |
| TOTAL                                                   |                                                                                               |    542 |     31 |      34 |      607 |
|                     filepath                      |                                function                                | passed | skipped | SUBTOTAL |
| ------------------------------------------------- | ---------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_linalg_eig.py                          | test_linalg_eig                                                        |      1 |       0 |        1 |
| tests/test_linalg_eig.py                          | test_linalg_eig_batched                                                |      1 |       0 |        1 |
| tests/test_is_same_size.py                        | test_is_same_size                                                      |      1 |       0 |        1 |
| tests/test_is_same_size.py                        | test_is_same_size_non_contiguous                                       |      1 |       0 |        1 |
| tests/test_is_same_size.py                        | test_is_same_size_scalar_tensor                                        |      1 |       0 |        1 |
| tests/test_slow_conv_transpose2d.py               | test_slow_conv_transpose2d                                             |      1 |       0 |        1 |
| tests/test_slow_conv_transpose2d.py               | test_slow_conv_transpose2d_out_variant                                 |      1 |       0 |        1 |
| tests/test_softmax.py                             | test_softmax_wide_inner_row                                            |      1 |       0 |        1 |
| tests/test_softmax.py                             | test_softmax                                                           |      1 |       0 |        1 |
| tests/test_softmax.py                             | test_softmax_out                                                       |      1 |       0 |        1 |
| tests/test_softmax.py                             | test_softmax_backward_out                                              |      1 |       0 |        1 |
| tests/test_softmax.py                             | test_softmax_backward                                                  |      1 |       0 |        1 |
| tests/test_bitwise_right_shift_.py                | test_bitwise_right_shift_                                              |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_single_diagonal                                           |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_multiple_diagonals                                        |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_various_offsets                                           |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_non_square                                                |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_empty                                                     |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_layouts                                                   |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_1d_scalar_promotion                                       |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_offsets_dtype                                             |      1 |       0 |        1 |
| tests/test_spdiags.py                             | test_spdiags_duplicate_offsets                                         |      1 |       0 |        1 |
| tests/test_special_chebyshev_polynomial_v.py      | test_special_chebyshev_polynomial_v                                    |      1 |       0 |        1 |
| tests/test_special_expm1.py                       | test_accuracy_special_expm1                                            |      1 |       0 |        1 |
| tests/test_special_i1e.py                         | test_special_i1e                                                       |      1 |       0 |        1 |
| tests/test_special_i1e.py                         | test_special_i1e_out                                                   |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i0.py          | test_special_modified_bessel_i0                                        |      1 |       0 |        1 |
| tests/test_special_modified_bessel_i0.py          | test_special_modified_bessel_i0_out                                    |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0                                 |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_out                             |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_strided                         |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_special_values                  |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_edge_values                     |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_empty                           |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_rejects_unsupported_dtype       |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_rejects_non_float_out_buffer    |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_rejects_positional_out          |      1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py   | test_special_scaled_modified_bessel_k0_rejects_non_tensor_input        |      1 |       0 |        1 |
| tests/test_special_spherical_bessel_j0.py         | test_special_spherical_bessel_j0                                       |      1 |       0 |        1 |
| tests/test_sqrt.py                                | test_sqrt                                                              |      1 |       0 |        1 |
| tests/test_sqrt.py                                | test_sqrt_                                                             |      1 |       0 |        1 |
| tests/test_std.py                                 | test_std                                                               |      1 |       0 |        1 |
| tests/test_swiglu.py                              | test_swiglu                                                            |      1 |       0 |        1 |
| tests/test_take.py                                | test_take                                                              |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_out                                                          |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_negative_index                                               |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_out_of_bounds                                                |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_noncontiguous                                                |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_out_noncontiguous                                            |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_out_resizes_empty_tensor                                     |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_out_rejects_mismatched_dtype                                 |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_empty                                                        |      1 |       0 |        1 |
| tests/test_take.py                                | test_take_multidim_index                                               |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell                                      |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_noncontiguous                        |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_empty                                |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_hidden_bias_without_input_bias       |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_rejects_input_bias_only              |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_rejects_unsupported_dtype                     |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_out                                  |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_accuracy_thnn_fused_gru_cell_out_resize                           |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_out_aliases_input                             |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_out_rejects_invalid_dtype                     |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_out_rejects_invalid_device                    |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_out_rejects_overlapping_outputs               |      1 |       0 |        1 |
| tests/test_thnn_fused_gru_cell.py                 | test_thnn_fused_gru_cell_backward_end_to_end                           |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_dtype                                                          |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_dtype_cast                                                |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_preserve_strides                                          |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_float_to_float                                            |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_float_to_int                                              |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_int_to_float                                              |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_int_to_int                                                |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_float_to_uint8                                            |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_uint8_to_float                                            |      1 |       0 |        1 |
| tests/test_to_copy.py                             | test_to_copy_uint8_to_int                                              |      1 |       0 |        1 |
| tests/test_trace_backward.py                      | test_trace_backward                                                    |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril                                                              |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_noncontiguous                                                |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_wide_exact_row_dispatch                                      |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_tiny_batched_tile_dispatch                                   |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_tiny_batched_tile_correctness                                |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out                                                          |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_resizes                                                  |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_aliases_input                                            |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_noncontiguous_out                                        |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_sliced_leading_batch_out                                 |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_extreme_diagonal_noncontiguous_out                       |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_out_strided_dispatch_guards                                  |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace                                                      |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_noncontiguous                                        |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_strided_batched_view                                 |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_expanded_view                                        |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_overlapping_as_strided_view                          |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_empty                                                        |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_empty                                                |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_extreme_diagonal                                     |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_invalid_rank                                                 |      1 |       0 |        1 |
| tests/test_tril.py                                | test_tril_inplace_invalid_rank                                         |      1 |       0 |        1 |
| tests/test_unbind.py                              | test_unbind                                                            |      1 |       0 |        1 |
| tests/test_unbind.py                              | test_unbind_default_dim                                                |      1 |       0 |        1 |
| tests/test_unbind.py                              | test_unbind_negative_dim                                               |      1 |       0 |        1 |
| tests/test_uniform_.py                            | test_uniform_                                                          |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_acc_false                                        |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_acc_true                                         |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_index_dtype                                      |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_error_all_none                                   |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_error_too_many_indices                           |      1 |       0 |        1 |
| tests/test_unsafe_index_put.py                    | test_unsafe_index_put_mixed_none_and_tensor                            |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward                                       |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_grad_input                            |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_layout                                |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_impulse                               |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_resize                                |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_invalid                               |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_out_error                             |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_deterministic                         |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_nonfinite_cpu_reference               |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_fp64_opmath                           |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_extreme_positive_scale                |      1 |       0 |        1 |
| tests/test_upsample_nearest1d.py                  | test_upsample_nearest1d                                                |      1 |       0 |        1 |
| tests/test_ones.py                                | test_ones                                                              |      1 |       0 |        1 |
| tests/test_ones.py                                | test_ones_empty                                                        |      1 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py  | test_value_selecting_reduction_backward                                |      1 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py  | test_value_selecting_reduction_backward_max_dim                        |      1 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py  | test_value_selecting_reduction_backward_min_dim                        |      1 |       0 |        1 |
| tests/test_view_as_complex.py                     | test_view_as_complex_accuracy                                          |      1 |       0 |        1 |
| tests/test_view_as_complex.py                     | test_view_as_complex_2d_accuracy                                       |      1 |       0 |        1 |
| tests/test_view_as_complex.py                     | test_view_as_complex_3d_accuracy                                       |      1 |       0 |        1 |
| tests/test_weight_int8pack_mm.py                  | test_weight_int8pack_mm                                                |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy                                                             |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_out                                                         |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_special_values                                              |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_tensor_scalar                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_tensor_scalar_out                                           |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_scalar_tensor                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                               | test_xlogy_scalar_tensor_out                                           |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations                                                      |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_int_bool                                             |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_row_order                                            |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_non_contiguous                                       |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_nan_inf                                              |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_default_arguments                                    |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_rejects_negative_r                                   |      1 |       0 |        1 |
| tests/test_combinations.py                        | test_combinations_rejects_non_1d                                       |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test_indices_copy                                                      |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test_indices_copy_empty                                                |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test_indices_copy_out                                                  |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test__indices_copy                                                     |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test__indices_copy_empty                                               |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test__indices_copy_out                                                 |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test__indices_copy_out_noncontiguous_out                               |      1 |       0 |        1 |
| tests/test_indices_copy.py                        | test__indices_copy_noncoo_layout_raises_not_implemented                |      1 |       0 |        1 |
| tests/test_affine_grid_generator_backward.py      | test_affine_grid_generator_backward_2d                                 |      1 |       0 |        1 |
| tests/test_affine_grid_generator_backward.py      | test_affine_grid_generator_backward_3d                                 |      1 |       0 |        1 |
| tests/test_zeros.py                               | test_zeros                                                             |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward                                          |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward_dim_sweep                                |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward_cpu_oracle                               |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward_native_dropout                           |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward_empty_gems_cpu_oracle                    |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_scaled_dot_product_cudnn_attention_backward                       |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_efficient_attention_backward                                      |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_scaled_dot_product_efficient_attention_backward                   |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py            | test_scaled_dot_product_flash_attention_backward                       |      1 |       0 |        1 |
| tests/test_conv3d.py                              | test_conv3d                                                            |      1 |       0 |        1 |
| tests/test_conv3d.py                              | test_conv3d_padding                                                    |      1 |       0 |        1 |
| tests/test_lift.py                                | test_lift                                                              |      1 |       0 |        1 |
| tests/test_lift.py                                | test_lift_out                                                          |      1 |       0 |        1 |
| tests/test_lift.py                                | test_lift_low_precision                                                |      1 |       0 |        1 |
| tests/test_lift.py                                | test_lift_out_low_precision                                            |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul                                                      |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_non_contiguous                                       |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_out                                                  |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_nan_inf                                              |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_backward                                             |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_empty_list                                   |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_non_tensor                                   |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_1d_matrix                                    |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_3d_tensor                                    |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_mismatched_dims                              |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_rejects_int_dtype                                    |      1 |       0 |        1 |
| tests/test_chain_matmul.py                        | test_chain_matmul_out_rejects_wrong_dtype                              |      1 |       0 |        1 |
| tests/test__linalg_det.py                         | test_accuracy_linalg_det                                               |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce                                                          |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_large_key_space                                          |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_coalesced_input                                          |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_value_ranges                                             |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_nan_inf                                                  |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_rejects_dense_input                                      |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_rejects_csr_input                                        |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test_coalesce_rejects_fp8_input                                        |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce                                                         |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_value_ranges                                            |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_nan_inf                                                 |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_out                                                     |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_dense_dims                                              |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_out_dense_dims                                          |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_rejects_coalesced_input                                 |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_out_rejects_wrong_dtype_and_shape                       |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_rejects_dense_input                                     |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_rejects_csr_input                                       |      1 |       0 |        1 |
| tests/test_coalesce.py                            | test__coalesce_rejects_fp8_input                                       |      1 |       0 |        1 |
| tests/test_mv.py                                  | test_mv                                                                |      1 |       0 |        1 |
| tests/test_accuracy_utils.py                      | test_gems_log_prefix                                                   |      1 |       0 |        1 |
| tests/test_accuracy_utils.py                      | test_gems_log_logger                                                   |      1 |       0 |        1 |
| tests/test_values.py                              | test_accuracy_values_coo                                               |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_mutation_through_view                                      |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_strided_raises                                             |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_sparse_compressed                                          |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_unsupported_layouts_reject_like_native                     |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_non_float_dtypes_alias                                     |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_int_hybrid_alias                                           |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_requires_grad_nondifferentiable_contract                   |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_sparse_requires_grad_view_contract                         |      1 |       0 |        1 |
| tests/test_values.py                              | test_values_dispatch_stability                                         |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_aminmax                                                           |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_accuracy_aminmax_no_dim                                           |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax                                                          |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax_zero                                                     |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax_inf                                                      |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax_int                                                      |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax_scalar                                                   |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test__aminmax_out                                                      |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_underscore_aminmax_nan                                            |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_underscore_aminmax_non_contiguous                                 |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_underscore_aminmax_out_resizes_and_aliases                        |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_underscore_aminmax_out_rejects_mismatch                           |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_underscore_aminmax_empty_matches_aten                             |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_aminmax_coordinate_view                                           |      1 |       0 |        1 |
| tests/test_aminmax.py                             | test_aminmax_no_dim_strided                                            |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add                                                               |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_complex                                                       |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_                                                              |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_tensor_scalar                                                 |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_tensor_scalar_                                                |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_scalar_tensor                                                 |      1 |       0 |        1 |
| tests/test_add.py                                 | test_add_scalar_scalar                                                 |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_unpack_seq_accuracy_2d                                            |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_pack_unpack_roundtrip_2d                                          |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_pack_unpack_roundtrip_3d                                          |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_unpack_seq_single_batch                                           |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_unpack_seq_short_sequences                                        |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_unpack_seq_3d_edge_cases                                          |      1 |       0 |        1 |
| tests/test_unpack_seq.py                          | test_pack_unpack_int8_roundtrip                                        |      1 |       0 |        1 |
| tests/test_arctan2.py                             | test_arctan2                                                           |      1 |       0 |        1 |
| tests/test_arctan2.py                             | test_arctan2_                                                          |      1 |       0 |        1 |
| tests/test_arctan2.py                             | test_arctan2__integer_rejects                                          |      1 |       0 |        1 |
| tests/test_arctan2.py                             | test_arctan2_special_values                                            |      1 |       0 |        1 |
| tests/test_arctan2.py                             | test_arctan2_broadcast                                                 |      1 |       0 |        1 |
| tests/test_concat_and_cache_mla.py                | test_concat_and_cache_mla                                              |      1 |       0 |        1 |
| tests/test_mm.py                                  | test_mm                                                                |      1 |       0 |        1 |
| tests/test_mm.py                                  | test_mm_broadcast_stride_zero                                          |      1 |       0 |        1 |
| tests/test_mm.py                                  | test_mm_out_vllm_tma_column_major_weight                               |      1 |       0 |        1 |
| tests/test_mm.py                                  | test_mm_self_transpose                                                 |      1 |       0 |        1 |
| tests/test_mm.py                                  | test_mm_out_self_transpose                                             |      1 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py            | test_sparse_compressed_tensor_explicit_size                            |      1 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py            | test_sparse_compressed_tensor_infer_size                               |      1 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py            | test_sparse_compressed_tensor_rejects_bad_layout                       |      1 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py            | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap       |      1 |       0 |        1 |
| tests/test_batch_norm_gather_stats_with_counts.py | test_batch_norm_gather_stats_with_counts                               |      1 |       0 |        1 |
| tests/test_lt.py                                  | test_lt                                                                |      1 |       0 |        1 |
| tests/test_lt.py                                  | test_lt_scalar                                                         |      1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_iluvatar_fp64_rejected                |      0 |       1 |        1 |
| tests/test_upsample_bicubic2d_backward.py         | test_upsample_bicubic2d_backward_bfloat16_scalar_rounding              |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8                                                       |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_prequantized                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_large_k_scales                               |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_skinny_epilogue                              |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_dense_reuse                                  |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_graph_updates                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_empty                                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_output_dtype                                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_independent_and_invalid_out                           |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_broadcast                                    |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_invalid_inputs_and_scales                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_scalar_shapes                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_scaled_mm_interface                          |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_default_dtype                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_invalid_epilogue                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_mthreads_strided_epilogue                             |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_hopper_external_scales                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_hopper_torch_interface                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_fp8.py                         | test_mm_w8a8_fp8_hopper_torch_rowwise                                  |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_inplace_boundaries_exact                                     |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_overwrites_special_values_exact                              |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_noncontiguous_storage_exact                                  |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_repeated_calls_exact                                         |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zeros_factory_poisoned_allocation_exact                           |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_launch_core_dim_limit                                        |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_real_large_input_exact                                       |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zero_capped_grid_covers_all_tiles_exact                           |      0 |       1 |        1 |
| tests/test_zeros.py                               | test_zeros_factory_capped_grid_exact                                   |      0 |       1 |        1 |
| tests/test_flash_attention_backward.py            | test_flash_attention_backward_saved_state_arithmetic_range             |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mm_w8a8_int8                                                      |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mm_w8a8_int8_strides                                              |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mm_w8a8_int8_updates                                              |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mm_w8a8_int8_empty                                                |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mm_w8a8_int8_reject_float                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_scaled_bias                                                       |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_scaled_empty_bias                                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_scaled_graph_bias_updates                                         |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_scaled_validation                                                 |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_tensor_scales_without_bias                                        |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_scaled_unaligned_weight                                           |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_prequantized                                                      |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_long_k_overflow                                                   |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_mixed_scales                                                      |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_graph_updates                                                     |      0 |       1 |        1 |
| tests/test_mm_w8a8_int8.py                        | test_invalid                                                           |      0 |       1 |        1 |
| tests/test_unpack_seq.py                          | test_pack_unpack_fp8_roundtrip                                         |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_large_shapes                                            |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_transposed_inputs_and_k_tails                           |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_large_shape_entrypoints                                 |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_empty                                            |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_out_resize                                       |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_capped_routes                                    |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_narrow_real_core_dim                             |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_panel_loops                                      |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_broadcast_materialize                            |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_specialized_launch_cap                           |      0 |       1 |        1 |
| tests/test_mm.py                                  | test_mm_ascend_review_out_validation                                   |      0 |       1 |        1 |
| TOTAL                                             |                                                                        |    263 |      60 |      323 |
