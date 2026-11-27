|                     filepath                     |                              function                               | passed | failed | error | skipped | xfailed | SUBTOTAL |
| ------------------------------------------------ | ------------------------------------------------------------------- | -----: | -----: | ----: | ------: | ------: | -------: |
| tests/test_dense_dim.py                          | test_dense_dim_strided                                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_dense_dim.py                          | test_dense_dim_sparse_coo                                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_dense_dim.py                          | test_dense_dim_sparse_compressed                                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_dense_dim.py                          | test_dense_dim_matches_reference_table                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_dense_dim.py                          | test_dense_dim_disjoint_from_native_neighbors                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_unused_overrides_lists_injected_but_never_called               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_unused_overrides_empty_once_the_candidate_runs                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_unused_overrides_ignores_programmatic_overrides                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_report_is_withheld_when_an_override_name_did_not_match         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_report_writer_is_untouched_when_every_override_matched         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/core/test_override_name_mismatch.py        | test_non_timing_modes_do_not_treat_overrides_as_mistakes            |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test__addmm_activation.py                  | test_addmm_activation                                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test__addmm_activation.py                  | test_addmm_activation_out                                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test__weight_norm.py                       | test_underscore_weight_norm                                         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_addmv_.py                             | test_addmv_                                                         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_default_storage_offset                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_out_noncontiguous                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_out_resizes                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_out_aliases_input                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_out_dtype_mismatch_raises             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize                                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_int                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_out                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_identity_and_metadata                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_same_tensor_noop                          |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_cast                                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_strides                                   |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_out_alias                                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy_from_and_resize.py               | test_copy_from_and_resize_out_options                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_arccosh_.py                           | test_arccosh_                                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_unsafe.py           | test_sparse_coo_tensor_unsafe                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_unsafe.py           | test_sparse_coo_tensor_unsafe_hybrid                                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_unsafe.py           | test_sparse_coo_tensor_unsafe_skips_index_validation                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor_unsafe.py           | test_sparse_coo_tensor_unsafe_uncoalesced_flag                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_batch_norm_impl_index_backward.py     | test_batch_norm_impl_index_backward_eval                            |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_batch_norm_impl_index_backward.py     | test_batch_norm_impl_index_backward_empty                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_zero_size_dims                                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_duplicate_dim_raises                                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_out_of_range_dim_raises                             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dense_input_raises                                  |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_csr_input_raises                                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_out_dtype_mismatch_raises                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_dispatch_reaches_gems_impl                                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                  | test_bitwise_or_tensor                                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                  | test_bitwise_or_tensor_                                             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max                                                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_                                                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_low_precision                                        |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_uint8_int_scalar                                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_uint8_boundary_scalar                                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_min.py                          | test_clamp_min                                                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_clamp_min.py                          | test_clamp_min_                                                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_concatenate.py                        | test_concatenate                                                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_concatenate.py                        | test_concatenate_empty_tensor                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_conv_tbc_backward.py                  | test_conv_tbc_backward                                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                           | test_corrcoef                                                       |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                           | test_corrcoef_1d                                                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                           | test_corrcoef_non_contiguous                                        |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cross_entropy_loss.py                 | test_cross_entropy_loss_indices                                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cross_entropy_loss.py                 | test_cross_entropy_loss_probabilities                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cudnn_rnn_backward.py                 | test_cudnn_rnn_backward                                             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_accuracy_fake_quantize_per_tensor_affine_empty                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_fake_quantize_per_tensor_affine_invalid_qparams                |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_tile_plan_rejects_inexpressible                     |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_fliplr.py                             | test_fliplr_rejects_inputs_with_fewer_than_two_dims                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_dtype_mapping                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_error_paths                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_arcsinh.py                            | test_arcsinh_out                                                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_rejects_empty_list                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_rejects_multidim_input                          |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_rejects_mixed_device                            |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_rejects_mixed_dtype                             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_rejects_non_tensor                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_hstack.py                             | test_exception_hstack                                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py   | test_accuracy_has_compatible_shallow_copy_type                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py   | test_accuracy_has_compatible_shallow_copy_type_dense_family         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py   | test_accuracy_has_compatible_shallow_copy_type_sparse_family        |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py   | test_accuracy_has_compatible_shallow_copy_type_opaque               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_has_compatible_shallow_copy_type.py   | test_accuracy_has_compatible_shallow_copy_type_cross_family         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__empty_reinterpret                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__rejects_non_coo                                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_underscore_aminmax_out_rejects_mismatch                        |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_underscore_aminmax_empty_matches_aten                          |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_global_empty_matches_aten                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_import_enumerates_all_cases_without_device_calls               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_ascend_known_fnuz_conversion_is_unsupported                    |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_unrelated_compiler_errors_are_not_skipped                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_metax_known_native_rowwise_rejection                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_other_native_errors_are_not_skipped                            |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_resource_and_unknown_errors_propagate                          |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_unselected_tests_do_not_probe                                  |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test__stack_dim_out_of_range                                        |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test__stack_dim_out_of_range_single_input                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test__stack_zero_sized                                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_empty_exact                                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_across_devices_is_refused_when_re_entered                 |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_flat_grid_never_exceeds_core_dim                               |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__version_counter                                      |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__version_counter_identity                             |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__invalid_dims                                         |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_tensor_wrapper.py                     | test_strided_buffer_slice                                           |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/test_uniform.py                            | test_uniform_rejects_inverted_interval                              |      1 |      0 |     0 |       0 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_report_conversion_requires_platform_identity                   |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_environment_snapshot_records_model_resolution_controls         |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_operator_yaml_compiles_shape_dispatch_and_benchmark_contract   |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mul_operator_yaml_compiles_broadcast_and_scalar_variants       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mul_shape_loader_aligns_optional_rhs_before_count              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mul_executor_uses_variant_specific_tensor_and_scalar_arguments |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mul_keeps_legacy_contract_dispatch                             |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_route_platform_matches_default_manifest_catalog                |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mm_executor_resolves_route_before_tuner_lookup                 |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mm_executor_keeps_non_adapted_route_out_of_model_binding       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_worker_routes_actual_tensors_once_before_binding               |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_hopper_atomic_splitk_is_a_model_route                          |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_operator_yaml_rejects_device_placement_policy                  |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_operator_yaml_rejects_arbitrary_invocation                     |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_shared_safe_references_cover_fields_shapes_and_invoke_args     |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_shared_safe_references_reject_forward_dependencies_and_calls   |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_operator_yaml_rejects_unknown_identity_namespace               |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_public_operator_resolution_reports_missing_callable            |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_load_new_shape_spec_with_optional_count                        |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_load_legacy_shape_desc_without_count                           |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_shape_validation_is_strict                                     |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_variant_resolution_follows_public_mm_priority                  |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mm_variant_filter_is_deferred_until_tensor_routing             |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_mm_parent_percentage_limit_applies_to_raw_recipes              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_percentage_shape_limit_rejects_zero_resolved_shapes            |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_max_shapes_parser_accepts_counts_and_percentages               |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_max_shapes_parser_rejects_invalid_limits                       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_count_sort_requires_count                                      |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_random_sort_is_reproducible                                    |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_cli_defaults_and_sort_parser                                   |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_intermediate_cleanup_is_strictly_scoped_to_run_directory       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_merge_sqlite_shards_preserves_existing_rows                    |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_write_outputs_keeps_structured_jsonl_and_flat_csv              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_worker_success_and_failure_rows_use_platform_key               |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_generic_scheduler_prepares_cases_from_operator_yaml            |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_public_batch_api_returns_input_order_and_fail_fast_state       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_public_batch_api_rejects_unknown_or_misaligned_input_dtypes    |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/flagtune/test_pretune.py                   | test_sqlite_url_classification                                      |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_float8_byte_path                      |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_as_strided_copy.py                    | test_accuracy_as_strided_copy_out_float8_byte_path                  |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_embedding.py                          | test_embedding_dense_backward                                       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_fp8_einsum.py                         | test_accuracy_fp8_einsum                                            |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values_out_wrong_device                             |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num_hygon_special_values                                |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num_hygon_storage_offset                                |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num_hygon_fallback                                      |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num_hygon_scalar_delegates                              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num_hygon_metadata_delegates                            |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_float64_preserves_bits          |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_gate_engages                                     |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_gate_defers                                      |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_gate_limits_follow_element_width                 |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax                                                  |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_out                                              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_half_to_float                                    |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_neg_inf                                          |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_at_row_cap                                       |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_rdna4_softmax.py                      | test_rdna4_softmax_non_contiguous                                   |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm_descriptor_dtype_switch                              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py | test_scaled_dot_product_cudnn_attention                             |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_scaled_dot_product_cudnn_attention.py | test_scaled_dot_product_cudnn_attention_with_attn_bias              |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_float8_e8m0fnu                                    |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_float8_e8m0fnu_to_float32                         |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_float8                                         |      0 |      0 |     0 |       1 |       0 |        1 |
| tests/test_batch_norm_impl_index_backward.py     | test_batch_norm_impl_index_backward_train                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy.py               | test_binary_cross_entropy                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy.py               | test_binary_cross_entropy_weight                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy.py               | test_binary_cross_entropy_out                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy.py               | test_binary_cross_entropy_boundary                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_default                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_default_empty                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_default_uncoalesced                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dtype                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_all                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_dtype                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_out                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_negative                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dim_keeps_structural_zeros                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_hybrid_dense_dims                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_uncoalesced_input                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_integer_promotion                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_dtype_accumulator_precision                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_complex                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_dispatch_covers_every_overload                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sum.py                         | test_sparse_sum_uncoalesced_matches_aten                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_int8_overflow_scalar                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_clamp_max.py                          | test_clamp_max_uint8_overflow_scalar                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_chalf.py                              | test_chalf_real                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_chalf.py                              | test_chalf_complex                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_diag_embed_backward.py                | test_diag_embed_backward                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_diag_embed_backward.py                | test_diag_embed_backward_bridge                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_diag_embed_backward.py                | test_diag_embed_backward_use_gems_entry                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_digamma.py                            | test_digamma                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_digamma.py                            | test_digamma_byte                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_digamma.py                            | test_digamma_int_rejects_inplace                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_asinh                                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_asinh_various_sizes                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_asinh_edge_cases                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_asinh_empty_tensor                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_accuracy_asinh_out                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_asinh.py                              | test_asinh_                                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_embedding.py                          | test_embedding                                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_embedding.py                          | test_embedding_backward                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq_.py                                | test_eq_                                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq_.py                                | test_eq_scalar_                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py             | test_unique_dim_consecutive_int                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py             | test_unique_dim_consecutive_float                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py             | test_unique_dim_consecutive_edge                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py             | test_unique_dim_consecutive_wide_rows                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_accuracy_fake_quantize_per_tensor_affine                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_accuracy_fake_quantize_per_tensor_affine_tensor_qparams        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_accuracy_fake_quantize_per_tensor_affine_half_to_even          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine.py    | test_accuracy_fake_quantize_per_tensor_affine_noncontiguous         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_non_contiguous                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_view_strided                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_materialized_copy                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_special                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_backward                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_other_has_no_gradient                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_complex_copy                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_dispatch                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reshape_as.py                         | test_reshape_as_tile_kernel                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fliplr.py                             | test_accuracy_fliplr                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fliplr.py                             | test_accuracy_fliplr_non_float                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_fliplr.py                             | test_accuracy_fliplr_noncontiguous                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_full.py                               | test_full                                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_values                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_noncontiguous_batch              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_transposed                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_high_rank_noncontiguous          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_expanded                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_storage_offset                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_independent_storage              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py               | test_accuracy_view_as_complex_copy_empty                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_bsr                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_batched                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_int32_buffer                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_alias_and_mutation                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_dispatch_stability                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_dispatch_sentinel                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_crow_indices.py                       | test_crow_indices_unsupported_layout_raises                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_arcsinh.py                            | test_arcsinh                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_int_bool                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_sequence_form                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_non_contiguous                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_row_order                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_backward                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_nan_inf                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_cartesian_prod.py                     | test_cartesian_prod_backward_mixed_requires_grad                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hadamard_transform.py                 | test_hadamard_transform                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hadamard_transform.py                 | test_hadamard_transform_mn                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hadamard_transform.py                 | test_hadamard_transform_mn_scale                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hadamard_transform.py                 | test_hadamard_transform_mn_leading_dims                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hardtanh_.py                          | test_hardtanh_                                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hardtanh_.py                          | test_hardtanh__explicit                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hardtanh_.py                          | test_hardtanh__special_values                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_hstack.py                             | test_accuracy_hstack                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_argwhere.py                           | test_argwhere                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_argwhere.py                           | test_argwhere_edge_shapes                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_argwhere.py                           | test_argwhere_special_inputs                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_index_select.py                       | test_index_select                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_modified_bessel_i1.py         | test_special_modified_bessel_i1                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_modified_bessel_i1.py         | test_special_modified_bessel_i1_out                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_miopen_batch_norm.py                  | test_miopen_batch_norm                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_miopen_batch_norm.py                  | test_miopen_batch_norm_eval                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                        | test_linalg_cond                                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                        | test_linalg_cond_with_p                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                        | test_linalg_cond_svd                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_minimum.py                            | test_minimum                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__identity                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__sparse_dims_grow                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__hybrid_sparse_dims_grow                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__dense_dims_grow                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__dense_dims_grow_complex                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_resize_.py                     | test_sparse_resize__non_empty_rejects                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_max_pool2d.py                         | test_max_pool2d_with_indices                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_max_pool2d.py                         | test_max_pool2d_backward                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_max_pool2d.py                         | test_max_pool2d_with_indices_backward                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_logical_and.py                        | test_logical_and                                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_logical_and.py                        | test_logical_and_                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_accuracy_aminmax_no_dim                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax_zero                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax_inf                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax_int                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax_scalar                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test__aminmax_out                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_underscore_aminmax_nan                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_underscore_aminmax_non_contiguous                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_underscore_aminmax_out_resizes_and_aliases                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_coordinate_view                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_no_dim_strided                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_global_logical_nan_contract                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_global_storage_hole_nans                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_aminmax.py                            | test_aminmax_global_finite_controls                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_logspace.py                           | test_logspace                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_singular                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_singular_positions                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_empty                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_info_shape                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_info_ignores_nonfinite_pivots                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_info_reports_nan_only_as_nonzero_when_zero_absent           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_empty_batch_dimensions                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                  | test_sparse_bsc_tensor_explicit_size                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                  | test_sparse_bsc_tensor_infer_size                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                  | test_sparse_bsc_tensor_blocksize                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_bsc_tensor.py                  | test_sparse_bsc_tensor_rejects_wrong_layout                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_bsc                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_batched                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_int32_buffer                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_alias_and_mutation                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_unsupported_layout_raises                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_ccol_indices.py                       | test_ccol_indices_dense_and_coo_rejected                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values_keepdim_shapes                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values_ties                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values_strided_nonlast_reduction                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_median_dim_values.py                  | test_median_dim_values_out_rejects_bad_index_dtype                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_keeps_layout_not_values                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_explicit_zeros                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_empty_input                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_dtype                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_unsorted_columns                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_out                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                     | test_sparse_csr_sum_rejects_bad_request                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_flip.py                               | test_flip_backward                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_flip.py                               | test_flip_backward_empty_or_singleton                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_flip.py                               | test_flip                                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_flip.py                               | test_flip_with_non_dense_input                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_nan_to_num.py                         | test_nan_to_num                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_native_batch_norm_legit.py            | test_native_batch_norm_legit                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_native_batch_norm_legit.py            | test_native_batch_norm_legit_no_stats                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_native_batch_norm_legit.py            | test_native_batch_norm_legit_out                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_contiguous                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_non_contiguous                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_transposed_and_expanded                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_levels                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_mutation_through_view                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_dispatch_stability                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_inference_tensor                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_scalar_and_empty                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test__fw_primal.py                         | test_accuracy__fw_primal_dtype_dispatch                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_linalg_vecdot.py                      | test_linalg_vecdot                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_linalg_vecdot.py                      | test_linalg_vecdot_out                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_nll_loss2d_backward.py                | test_nll_loss2d_backward                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_norm_except_dim.py                    | test_norm_except_dim                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_norm_except_dim.py                    | test_norm_except_dim_dim                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_norm_except_dim.py                    | test_norm_except_dim_pow                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_pack_padded_sequence.py               | test_pack_padded_sequence                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_pack_padded_sequence.py               | test_pack_padded_sequence_ragged_lengths                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_single                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_multi                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_infers_terminal_total_L         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_empty_output                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_noncontiguous_dense             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_float16_preserves_bits          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_validation                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py     | test_padded_dense_to_jagged_forward_revalidates_mutated_offsets     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_permute_copy.py                       | test_permute_copy                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_poisson.py                            | test_poisson                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_poisson.py                            | test_poisson_varying_rates                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_poisson.py                            | test_poisson_large_lambda                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_poisson.py                            | test_poisson_zero_rate                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_prelu_kernel_backward.py              | test_prelu_kernel_backward                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_prelu_kernel_backward.py              | test_prelu_kernel_backward_per_channel                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_rad2deg.py                            | test_rad2deg                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_rad2deg.py                            | test_rad2deg_                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_reflection_pad3d_backward.py          | test_reflection_pad3d_backward                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_replication_pad1d.py                  | test_replication_pad1d                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_replication_pad1d.py                  | test_replication_pad1d_out                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_resize_output.py                      | test_resize_output                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm                                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm_out                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm_empty                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm_schema                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                          | test_scaled_mm_fast_route                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq.py                                 | test_eq                                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq.py                                 | test_eq_scalar                                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq.py                                 | test_eq_integer_precision                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq.py                                 | test_eq_scalar_integer_precision                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_eq.py                                 | test_eq_scalar_float_scalar                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_scatter_reduce_two.py                 | test_scatter_reduce_two_                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sigmoid.py                            | test_sigmoid                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sigmoid.py                            | test_sigmoid_                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sigmoid.py                            | test_sigmoid_backward                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_skip_layer_norm.py                    | test_skip_layernorm                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sobol_engine_ff_.py                   | test_sobol_engine_ff_                                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py               | test_sparse_sampled_addmm                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py               | test_sparse_sampled_addmm_large                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py               | test_sparse_sampled_addmm_out                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_bessel_j1.py                  | test_special_bessel_j1                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_entr.py                       | test_special_entr                                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_i0.py                         | test_special_i0                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_i0.py                         | test_special_i0_out                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_log_softmax.py                | test_special_log_softmax                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_log_softmax.py                | test_special_log_softmax_large_n                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_round.py                      | test_special_round                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_round.py                      | test_special_round_out                                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_round.py                      | test_special_round_midpoints                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_round.py                      | test_special_round_decimals                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_round.py                      | test_special_round_out_decimals                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                    | test_special_xlog1py                                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                    | test_special_xlog1py_x_zero                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                    | test_special_xlog1py_y_nan                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test_stack                                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test__stack                                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_stack.py                              | test__stack_complex                                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_same_dtype                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_broadcast                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_dtype_fallback                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_inplace_mixed_dtype_triton                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_same_dtype                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_broadcast                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_contiguous_boundaries_exact                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_noncontiguous_source_exact                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_noncontiguous_destination_exact                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_dtype_conversion_matches_cpu                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_broadcast_sources_exact                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_unaligned_storage_offset_exact                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_preserves_special_values_exact                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_across_devices_exact                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_identical_alias_is_a_no_op                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_shapes_exact                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_strided_template_exact                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_functional_cross_dtype_exact                              |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_splits_when_wider_than_pointwise_can_index                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_capped_grid_covers_every_tile                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_flat_kernel_offset_widths_agree                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_ascend_review_lazy_bits                                   |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_ascend_review_overlap                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_ascend_review_lazy_same_storage                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_copy.py                               | test_copy_ascend_review_component_capped_offsets                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes_                                                      |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__negative_dims                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__non_contiguous                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__scalar_and_empty                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__aliasing                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__expanded_overlap                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_swapaxes_.py                          | test_swapaxes__int_dtypes                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_tensor_wrapper.py                     | test_typed_pointer                                                  |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_tensor_wrapper.py                     | test_typed_pointer_reinterpret_with_offset                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_tensor_wrapper.py                     | test_typed_pointer_as_is                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_threshold_.py                         | test_threshold_                                                     |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy_complex                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy_scalar                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy_non_contiguous                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy_same_dim_does_not_alias                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_transpose_copy_invalid_dims                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_transpose_copy.py                     | test_accuracy_transpose_copy_float8                                 |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc                                                          |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_                                                         |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_special_values                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_always_rounds_towards_zero                               |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_integer_input                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_non_contiguous                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_trunc.py                              | test_trunc_empty                                                    |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_uniform.py                            | test_uniform_                                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_uniform.py                            | test_uniform                                                        |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_uniform.py                            | test_uniform_upper_bound_zero                                       |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_unsafe_masked_index.py                | test_unsafe_masked_index                                            |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_upsample_bilinear2d_aa_backward.py    | test_upsample_bilinear2d_aa_backward                                |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_upsample_nearest3d.py                 | test_upsample_nearest3d                                             |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_upsample_trilinear3d.py               | test_upsample_trilinear3d                                           |      0 |      1 |     0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                       | test_lu_with_info_exact_zero_pivot_position                         |      0 |      0 |     0 |       0 |       1 |        1 |
| tests/test_copy.py                               | test_copy_real_large_input_exact                                    |      0 |      0 |     1 |       0 |       0 |        1 |
| TOTAL                                            |                                                                     |    103 |    309 |     1 |      64 |       1 |      478 |
|                              filepath                              |                              function                               | passed | failed | skipped | SUBTOTAL |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_addmm.py                                                | test_addmm                                                          |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_beta_zero_ignores_bias                                   |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_vector_bias_shapes                                       |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_scalar_bias                                              |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_padded_vector_bias                                       |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_out                                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_out_beta_zero_ignores_bias                               |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_out_vector_bias_shapes                                   |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_out_noncontiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_broadcast_bias                                           |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_dtype_fp32_accum                                         |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_dtype_beta_zero_ignores_bias                             |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_dtype_out_fp32_accum                                     |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_dtype_out_beta_zero_ignores_bias                         |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_large_matrix                                             |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_large_matrix_alpha_beta                                  |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_small_square_large                                       |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_out_large_matrix                                         |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_bounded_workspace                                        |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_fallback_autotuning                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_ascend_beta_cache_transitions                            |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_ascend_review_empty                                      |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_ascend_review_out_resize                                 |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_ascend_review_beta_zero                                  |      1 |      0 |       0 |        1 |
| tests/test_addmm.py                                                | test_addmm_ascend_review_out_validation                             |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_self                                                     |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_self_scalar                                              |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_self_scalar_other                                        |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_self_out                                                 |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_self_out_cross_device                                    |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_uniform_condition_strided_storage                        |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_uniform_condition_boundaries                             |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_uniform_condition_scalar                                 |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_uniform_condition_overlapping_output_rejected            |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_uniform_condition_graph_reloads_inputs                   |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                       | test_where_invalid_broadcast_raises_runtime_error                   |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                           | test_zeros_like                                                     |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                           | test_zeros_like_launch_core_dim_limit                               |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                           | test_zeros_like_factory_capped_grid_exact                           |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                           | test_zeros_like_ascend_review_options                               |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos                                                           |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward                                                  |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward_registered_dispatch                              |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward_special_values                                   |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward_noncontiguous                                    |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward_empty                                            |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool2d_backward.py                         | test_adaptive_max_pool2d_backward                                   |      1 |      0 |       0 |        1 |
| tests/test_irshift.py                                              | test_irshift                                                        |      1 |      0 |       0 |        1 |
| tests/test_irshift.py                                              | test_irshift_edge_cases                                             |      1 |      0 |       0 |        1 |
| tests/test_isnan.py                                                | test_isnan                                                          |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss                                                        |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss_int_reduction                                          |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss_broadcast                                              |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss_empty                                                  |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss_functional                                             |      1 |      0 |       0 |        1 |
| tests/test_l1_loss.py                                              | test_l1_loss_invalid_reduction                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_batched                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_vector_rhs                                    |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_broadcast_batches                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_row_pivoting                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_singular                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_non_contiguous                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_dtype_mismatch                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_aux_outputs_usable                            |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn                                                            |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_                                                           |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_special_values                                             |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_noncontiguous                                              |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_out_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_out_resizes_empty_tensor                                   |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_out_rejects_mismatched_dtype                               |      1 |      0 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_empty                                                      |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_t.py               | test_special_shifted_chebyshev_polynomial_t                         |      1 |      0 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_t.py               | test_special_shifted_chebyshev_polynomial_t_scalar_n                |      1 |      0 |       0 |        1 |
| tests/test_special_erfc.py                                         | test_special_erfc                                                   |      1 |      0 |       0 |        1 |
| tests/test_special_erfc.py                                         | test_erfc                                                           |      1 |      0 |       0 |        1 |
| tests/test_special_erfc.py                                         | test_erfc_                                                          |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bin_ct                                      |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bin_ct_with_range                           |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bin_ct_with_weight                          |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bin_ct_with_density                         |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bins_tensor                                 |      1 |      0 |       0 |        1 |
| tests/test_histogram.py                                            | test_accuracy_histogram_bins_tensor_nonuniform                      |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_rejects_matrix_rhs                                     |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_rejects_left_false                                     |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_rejects_device_mismatch                                |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_requires_csr                                           |      1 |      0 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_requires_square                                        |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_update_stats.py                              | test_batch_norm_update_stats                                        |      1 |      0 |       0 |        1 |
| tests/test_dswiglu.py                                              | test_dswiglu                                                        |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_forward_modes                                         |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_comprehensive_forward_shapes                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_large_hidden                                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_dropout_one                                           |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_data                                                  |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                             | test_rnn_tanh_data_dispatch                                         |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                           | test_less_equal                                                     |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                           | test_less_equal_scalar                                              |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                           | test_less_equal_byte                                                |      1 |      0 |       0 |        1 |
| tests/test_less_equal.py                                           | test_less_equal_scalar_byte                                         |      1 |      0 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_                                                   |      1 |      0 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_fast                                               |      1 |      0 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_scalar_tensor_out                                  |      1 |      0 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_out_rejects_non_double_output                      |      1 |      0 |       0 |        1 |
| tests/test_frexp.py                                                | test_frexp                                                          |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_values                                                     |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_tile_config                                                |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_element_order                                              |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_mixed_dtype                                                |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_special_values                                             |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_invalid_rank                                               |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_invalid_out                                                |      1 |      0 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_non_tensor_operand                                         |      1 |      0 |       0 |        1 |
| tests/test_hardshrink_backward.py                                  | test_hardshrink_backward_grad_input                                 |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_rejects_non_tensor_argument                           |      1 |      0 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_out_rejects_mismatched_dtype                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor.py                                     | test_linalg_lu_factor                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor.py                                     | test_linalg_lu_factor_out                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_lu_factor.py                                     | test_linalg_lu_factor_zero_pivot                                    |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_unpack_pivots_false                                  |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_backward_entry                                       |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_backward_batched                                     |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_backward_nondifferentiable_outputs                   |      1 |      0 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_backward_dispatch                                    |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                              | test_max_pool3d_with_indices                                        |      1 |      0 |       0 |        1 |
| tests/test_max_pool3d_with_indices.py                              | test_max_pool3d_with_indices_ties                                   |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis                                                       |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_int_dtype                                             |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_intlist_overload                                      |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_identity                                              |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_result_aliases_input                                  |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_non_contiguous_input                                  |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_empty_tensor                                          |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_autograd                                              |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_out_of_range                                          |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_destination_out_of_range                              |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_length_mismatch                                       |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_repeated_axis                                         |      1 |      0 |       0 |        1 |
| tests/test_moveaxis.py                                             | test_moveaxis_0d_out_of_range                                       |      1 |      0 |       0 |        1 |
| tests/test_new_zeros_with_same_feature_meta.py                     | test_new_zeros_with_same_feature_meta                               |      1 |      0 |       0 |        1 |
| tests/test_new_zeros_with_same_feature_meta.py                     | test_new_zeros_with_same_feature_meta_invalid_batch_dims            |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add                                                      |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_contiguous_suffix                                    |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_                                                     |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_inplace_contiguous_suffix                            |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_invalid_index                                        |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_empty_index                                          |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_repeated_indices                                     |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_graph_replay                                         |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_short_invalid_preserves_input                        |      1 |      0 |       0 |        1 |
| tests/test_index_add.py                                            | test_index_add_multiblock_bounds                                    |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss                                         |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_target_patterns                         |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_default_reduction                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_extra_reduction_values                  |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_out                                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_noncontiguous                           |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_empty_batch                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_class_count                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_class_split_path                        |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_out_of_range_target                     |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_out_dtype                               |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_out_invalid                             |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_invalid_target_dtype                    |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_invalid_target_shape                    |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_invalid_reduction                       |      1 |      0 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_invalid_rank                            |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_rejects_rank                         |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_rejects_dtype                        |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_rejects_geometry                     |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_rejects_pattern                      |      1 |      0 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_rejects_non_tensor                   |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_alpha_types                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_bias_shape                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_special_values                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_empty                               |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_out                                 |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                              | test_unflatten_dense_tensors                                        |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                              | test_unflatten_dense_tensors_metadata                               |      1 |      0 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                              | test_unflatten_dense_tensors_exceeds_flat                           |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_scalar_input                       |      1 |      0 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_dense_input                        |      1 |      0 |       0 |        1 |
| tests/test_accuracy_utils.py                                       | test_gems_log_prefix                                                |      1 |      0 |       0 |        1 |
| tests/test_accuracy_utils.py                                       | test_gems_log_logger                                                |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin_out_integer_rejects                                     |      1 |      0 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin_boundaries_and_out_of_domain                            |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_gather_stats_with_counts.py                  | test_batch_norm_gather_stats_with_counts                            |      1 |      0 |       0 |        1 |
| tests/test_bmm.py                                                  | test_bmm_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_bmm.py                                                  | test_bmm_out_empty_dimensions                                       |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_coalesced_input                                       |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_rejects_dense_input                                   |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_rejects_csr_input                                     |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_rejects_coalesced_input                              |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_rejects_dense_input                                  |      1 |      0 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_rejects_csr_input                                    |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__tensor                                                     |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__scalar                                                     |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__backward_other                                             |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__backward_view                                              |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__scalar_backward                                            |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__backward_rejects_leaf                                      |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__scalar_backward_rejects_leaf                               |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__special_values                                             |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__view_receiver                                              |      1 |      0 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__invalid_operand                                            |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                                  | test_triplet_margin_loss                                            |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                                  | test_triplet_margin_loss_p1_eps                                     |      1 |      0 |       0 |        1 |
| tests/test_triplet_margin_loss.py                                  | test_triplet_margin_loss_empty_feature_dim                          |      1 |      0 |       0 |        1 |
| tests/deprecated/test_mrope.py                                     | test_mrope_vs_torch                                                 |      1 |      0 |       0 |        1 |
| tests/test_is_any_true.py                                          | test_is_any_true_invalid_dtype                                      |      1 |      0 |       0 |        1 |
| tests/test_det.py                                                  | test_det_non_contiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_det.py                                                  | test_det_errors                                                     |      1 |      0 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm_ascend_core_dim_boundary                              |      1 |      0 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm_ascend_core_dim_capped                                |      1 |      0 |       0 |        1 |
| tests/test_acos.py                                                 | test_acos_backward_fused_kernel_count                               |      0 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test_linalg_solve_ex                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test_linalg_solve_ex_batched                                        |      0 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_left_false                                    |      0 |      1 |       0 |        1 |
| tests/test_linalg_solve_ex.py                                      | test__linalg_solve_ex_complex                                       |      0 |      1 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_complex_extreme_values                                     |      0 |      1 |       0 |        1 |
| tests/test_sgn.py                                                  | test_sgn_complex_nonfinite_values                                   |      0 |      1 |       0 |        1 |
| tests/test_special_erfc.py                                         | test_erfc_byte                                                      |      0 |      1 |       0 |        1 |
| tests/test_special_erfc.py                                         | test_erfc_int_rejects_inplace                                       |      0 |      1 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_vector_rhs                                             |      0 |      1 |       0 |        1 |
| tests/test_spsolve.py                                              | test_spsolve_identity                                               |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_backward_                                          |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_inplace_version                                    |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_lower_tail                                         |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_no_repeated_blocks                                 |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_far_left_tail                                      |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_consecutive_calls_advance_generator                |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_tensor                                      |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_tensor_noncontiguous_and_special_values     |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_scalar                                      |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_scalar_tensor                                      |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_tensor_out                                  |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_scalar_out                                  |      0 |      1 |       0 |        1 |
| tests/test_float_power.py                                          | test_float_power_tensor_tensor_out_noncontiguous                    |      0 |      1 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_strided_input                                              |      0 |      1 |       0 |        1 |
| tests/test_ger.py                                                  | test_ger_backward                                                   |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr                                                       |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_result_is_independent_of_input_storage                |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_out                                                   |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_out_resizes_buffer                                    |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_out_writes_through_strided_buffer                     |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_reads_layout                                          |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_extreme_qparams                                       |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_storage_boundaries                                    |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_quantized_special_values                              |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_degenerate_shape                                      |      0 |      1 |       0 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_out_degenerate_shape                                  |      0 |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                           | test_linalg_eig                                                     |      0 |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                           | test_linalg_eig_batched                                             |      0 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack                                                      |      0 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_unpack_data_false                                    |      0 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_batched                                              |      0 |      1 |       0 |        1 |
| tests/test_lu_unpack.py                                            | test_lu_unpack_out                                                  |      0 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_backward                                |      0 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_backward_duplicate_targets              |      0 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_backward_strided_grad_out               |      0 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_backward_large_class_count              |      0 |      1 |       0 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_special_values                          |      0 |      1 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured                                      |      0 |      1 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_strided                              |      0 |      1 |       0 |        1 |
| tests/test_to_sparse_semi_structured.py                            | test_to_sparse_semi_structured_special_values                       |      0 |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_reference                           |      0 |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py                           | test_cudnn_convolution_add_relu_invalid_ndim                        |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                        | test_rowwise_prune                                                  |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                        | test_rowwise_prune_all_kept                                         |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                                        | test_rowwise_prune_none_kept                                        |      0 |      1 |       0 |        1 |
| tests/test_unflatten_dense_tensors.py                              | test_unflatten_dense_tensors_strided_flat                           |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax                                            |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_int_conversion                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_int_default_dtype                          |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_special_values                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_int_special_values                         |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_backward                                   |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_empty_input                                |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_uncoalesced_input                          |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_out                                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_invalid_dim                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_log_softmax.py                                  | test_sparse_log_softmax_dispatch                                    |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin                                                         |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin_                                                        |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                               | test_arcsin_empty_and_scalar                                        |      0 |      1 |       0 |        1 |
| tests/test_bmm.py                                                  | test_bmm                                                            |      0 |      1 |       0 |        1 |
| tests/test_bmm.py                                                  | test_bmm_empty_dimensions                                           |      0 |      1 |       0 |        1 |
| tests/test_bmm.py                                                  | test_bmm_empty_dimensions_reject_mismatch                           |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce                                                       |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_large_key_space                                       |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_value_ranges                                          |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_nan_inf                                               |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test_coalesce_rejects_fp8_input                                     |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce                                                      |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_value_ranges                                         |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_nan_inf                                              |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_out                                                  |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_dense_dims                                           |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_out_dense_dims                                       |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_out_rejects_wrong_dtype_and_shape                    |      0 |      1 |       0 |        1 |
| tests/test_coalesce.py                                             | test__coalesce_rejects_fp8_input                                    |      0 |      1 |       0 |        1 |
| tests/test_ge_.py                                                  | test_ge__aliased_operand                                            |      0 |      1 |       0 |        1 |
| tests/test_triplet_margin_loss.py                                  | test_triplet_margin_loss_extremal_p                                 |      0 |      1 |       0 |        1 |
| tests/deprecated/test_FLA/test_fused_recurrent_gated_delta_rule.py | test_fused_recurrent_gated_delta_rule_fwd_accuracy                  |      0 |      1 |       0 |        1 |
| tests/deprecated/test_flash_mla.py                                 | test_flash_mla                                                      |      0 |      1 |       0 |        1 |
| tests/test_is_any_true.py                                          | test_is_any_true                                                    |      0 |      1 |       0 |        1 |
| tests/test_is_any_true.py                                          | test_is_any_true_arbitrary_rank                                     |      0 |      1 |       0 |        1 |
| tests/test_is_any_true.py                                          | test_is_any_true_scalar_and_expand                                  |      0 |      1 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py            | test_scaled_dot_product_attention_math_for_mps                      |      0 |      1 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py            | test_scaled_dot_product_attention_math_for_mps_all_negative_logits  |      0 |      1 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py            | test_scaled_dot_product_attention_math_for_mps_weights_match_output |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_random                                                     |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_random_batch                                               |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_positive_definite                                          |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_negative_determinant                                       |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_diagonal                                                   |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_singular                                                   |      0 |      1 |       0 |        1 |
| tests/test_det.py                                                  | test_det_empty                                                      |      0 |      1 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm                                                       |      0 |      1 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm_ascend_small_shapes                                   |      0 |      1 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm_ascend_review_empty                                   |      0 |      1 |       0 |        1 |
| tests/test_rms_norm.py                                             | test_rms_norm_ascend_review_weight_gradient                         |      0 |      1 |       0 |        1 |
| tests/test_exponential_.py                                         | test_exponential_race_matches_categorical                           |      0 |      0 |       1 |        1 |
| tests/test_ger.py                                                  | test_ger_unsupported_dtype                                          |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                                  | test_hardshrink_backward                                            |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                                  | test_hardshrink_backward_special_values                             |      0 |      0 |       1 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_per_channel_quantization                              |      0 |      0 |       1 |        1 |
| tests/test_int_repr.py                                             | test_int_repr_rejects_non_quantized_dtype                           |      0 |      0 |       1 |        1 |
| tests/test_linalg_lu_factor.py                                     | test_linalg_lu_factor_zero_pivot_no_pivot_dense                     |      0 |      0 |       1 |        1 |
| tests/test_multilabel_margin_loss.py                               | test_multilabel_margin_loss_invalid_input_dtype                     |      0 |      0 |       1 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_unsupported_dtype                  |      0 |      0 |       1 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_invalid_dtype_param                |      0 |      0 |       1 |        1 |
| tests/test__sparse_log_softmax.py                                  | test__sparse_log_softmax_rejects_half_to_float                      |      0 |      0 |       1 |        1 |
| tests/test_bmm.py                                                  | test_bmm_w8a8_fp8                                                   |      0 |      0 |       1 |        1 |
| tests/test_ge_.py                                                  | test_ge__unsupported_dtype                                          |      0 |      0 |       1 |        1 |
| tests/test_triplet_margin_loss.py                                  | test_triplet_margin_loss_broadcast_and_promotion                    |      0 |      0 |       1 |        1 |
| tests/deprecated/test_FLA/test_fused_recurrent_gated_delta_rule.py | test_fused_recurrent_gated_delta_rule_matches_vllm                  |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_fused_inv_rope_fp8_quant                                       |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_output_layout                                                  |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_per_group_contiguity                                           |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_scales_are_power_of_two                                        |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_large_values                                                   |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_nope_dims_unchanged                                            |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_zero_positions                                                 |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_dequant_numerical_accuracy                                     |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_unfused_path                                                   |      0 |      0 |       1 |        1 |
| tests/deprecated/test_fused_inv_rope_fp8_quant.py                  | test_with_real_deepseek_v4_rope                                     |      0 |      0 |       1 |        1 |
| tests/deprecated/test_topk_softmax.py                              | test_topk_softmax                                                   |      0 |      0 |       1 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py            | test_scaled_dot_product_attention_math_for_mps_negative_nonsquare   |      0 |      0 |       1 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py            | test_kunlunxin_flash_negative_tail                                  |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                   | test_rms_norm_w8a16_fp8_weight_updates                              |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                   | test_rms_norm_w8a16_fp8_encodings                                   |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                   | test_rms_norm_w8a16_fp8                                             |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                   | test_rms_norm_w8a16_fp8_mthreads_shapes                             |      0 |      0 |       1 |        1 |
| TOTAL                                                              |                                                                     |    221 |    108 |      32 |      361 |
