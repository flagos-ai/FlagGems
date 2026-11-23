|                           filepath                           |                                  function                                   | passed | failed | skipped | SUBTOTAL |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/core/test_external_baseline.py                         | test_nested_metadata_matches_even_when_shape_key_is_a_descriptor            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_tensor_identity_mismatch_does_not_use_first_shape_shortcut             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_scalar_type_and_value_mismatches_fall_back                             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_ambiguity_does_not_select_first_record                                 |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_named_inputs_require_binding_and_compare_scalars                       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_legacy_shape_and_microsecond_units_and_private_operator_names          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_invalid_latency_is_a_miss                                              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_invalid_file_warns_and_disables_comparison                             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_missing_file_warns_and_valid_metadata_is_preserved                     |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_scaling_selects_chip_and_resource_and_prefers_measured                 |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_invalid_scaling_falls_back_to_raw_ratio                                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_external_ratio_preserves_native_measurements_and_records_provenance    |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_skip_native_can_use_external_data_and_a_miss_keeps_na                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_real_summary_prefers_external_per_case_and_preserves_fallback          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_malformed_metadata_does_not_fall_through_to_legacy_shape               |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_malformed_resource_and_oversized_latency_do_not_break_benchmarks       |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_factory_device_matches_selected_backend_without_weakening_other_inputs |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_thead_runtime_vendor_uses_ops_baseline_t_head_scaling                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_h200_does_not_use_h20_scaling                                          |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_device_normalization_requires_the_selected_target                      |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_factory_normalization_still_checks_all_other_parameters                |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_named_device_metadata_uses_the_same_target_rule                        |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_overlapping_models_select_the_exact_chip_and_preserve_suffixes         |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_vendor_alias_is_explicit_and_does_not_match_other_vendors              |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_malformed_kwargs_metadata_falls_back_without_crashing                  |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_legacy_shape_cannot_establish_multi_input_or_parameter_abi             |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_legacy_sglang_packed_silu_and_mul_does_not_match_two_tensor_gems_abi   |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_incomplete_named_metadata_cannot_choose_a_semantic_baseline            |      1 |      0 |       0 |        1 |
| tests/core/test_external_baseline.py                         | test_complete_named_metadata_still_matches_and_checks_all_parameters        |      1 |      0 |       0 |        1 |
| tests/flagtune/test_inference_candidate_domain.py            | test_run_proposer_filters_runtime_configs_outside_model_domain              |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_matches_reference_without_final_state           |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_uses_initial_state_and_returns_final_state      |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_supports_two_sequence_varlen_pack               |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_accepts_non_contiguous_seq_first_inputs         |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_supports_qk_l2norm_option                       |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_supports_qk_l2norm_on_chunk_path                |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_fused_tail_vblock_matches_reference             |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_direct_path_shape_variants                      |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_varlen_initial_state_gqa_non_aligned_lengths    |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_zero_values_extreme_gates_return_zero_state     |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_value_edge_cases_match_reference                |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_accepts_non_contiguous_head_first_inputs        |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_repeated_calls_are_deterministic                |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_rejects_invalid_public_arguments                |      1 |      0 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule.py                | test_chunk_gated_delta_rule_does_not_broadly_reject_iluvatar_chunk_path     |      1 |      0 |       0 |        1 |
| tests/test__list_to_tensor.py                                | test_accuracy__list_to_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test__transformer_encoder_layer_fwd.py                 | test__transformer_encoder_layer_fwd                                         |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                            | test_accuracy_adaptive_avg_pool2d_forward                                   |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                            | test_accuracy_adaptive_avg_pool2d_boundary_layout                           |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                            | test_accuracy_adaptive_avg_pool2d_out                                       |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                            | test_accuracy_adaptive_avg_pool2d_out_layout                                |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d.py                            | test_accuracy_adaptive_avg_pool2d_out_resizes                               |      1 |      0 |       0 |        1 |
| tests/test_add_rms_norm.py                                   | test_add_rms_norm                                                           |      1 |      0 |       0 |        1 |
| tests/test_addr_.py                                          | test_addr_                                                                  |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args                                 |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_invalid_indices                 |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_invalid_metadata                |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_empty_nonunit_stride            |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_zero_batch_compressed_stride    |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_nonempty_plain_stride           |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_dispatch                        |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_zero_compressed_axis            |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_zero_axis_metadata              |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_zero_axis_invalid_indices       |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_compressed_tensor_args.py         | test_validate_sparse_compressed_tensor_args_large_batch                     |      1 |      0 |       0 |        1 |
| tests/test_archsinh.py                                       | test_arcsinh_                                                               |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy                                               |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_default_storage_offset                        |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_out_noncontiguous                             |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_out_resizes                                   |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_out_aliases_input                             |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_out_dtype_mismatch_raises                     |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_float8_byte_path                              |      1 |      0 |       0 |        1 |
| tests/test_as_strided_copy.py                                | test_accuracy_as_strided_copy_out_float8_byte_path                          |      1 |      0 |       0 |        1 |
| tests/test_atleast_3d.py                                     | test_atleast_3d                                                             |      1 |      0 |       0 |        1 |
| tests/test_atleast_3d.py                                     | test_atleast_3d_nan_inf                                                     |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_elemt.py                               | test_batch_norm_elemt                                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod                                                        |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_ignores_unstored_zeros                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_stored_zero_wins                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_empty_input                                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_dtype                                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_unsorted_columns                                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_empty_rows_and_columns                                 |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                | test_sparse_csr_prod_rejects_bad_request                                    |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount                                                      |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_with_weights                                         |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_with_minlength                                       |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_empty                                                |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_single                                               |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_all_zeros                                            |      1 |      0 |       0 |        1 |
| tests/test_bincount.py                                       | test_accuracy_bincount_weights_edge_cases                                   |      1 |      0 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                              | test_bitwise_or_tensor                                                      |      1 |      0 |       0 |        1 |
| tests/test_bitwise_or_tensor.py                              | test_bitwise_or_tensor_                                                     |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_dense_layout                                         |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_cross_dtype_cast                                     |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_ignores_input_metadata                               |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_grad_view                                            |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_cast_boundaries                                      |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_special_values                                       |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_sparse_input                                         |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_sparse_uncoalesced                                   |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_sparse_empty_domain                                  |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_sparse_compressed                                    |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_masked_grad_integer                                  |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_masked_grad_dense_input                              |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_rejects_non_tensor_grad                              |      1 |      0 |       0 |        1 |
| tests/test_to_dense_backward.py                              | test_to_dense_backward_rejects_invalid_masked_grad                          |      1 |      0 |       0 |        1 |
| tests/test_channel_shuffle.py                                | test_channel_shuffle                                                        |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                                   | test_clamp_tensor                                                           |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                                   | test_clamp_tensor_                                                          |      1 |      0 |       0 |        1 |
| tests/test_concatenate.py                                    | test_concatenate                                                            |      1 |      0 |       0 |        1 |
| tests/test_concatenate.py                                    | test_concatenate_empty_tensor                                               |      1 |      0 |       0 |        1 |
| tests/test_conv_depthwise2d.py                               | test_conv_depthwise2d                                                       |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution                                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_empty_batch                                                |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_noncontiguous_input                                        |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_noncontiguous_weight                                       |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_spatial_rank                                               |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_depthwise                                                  |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_transposed                                                 |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_bias_is_optional                                           |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_non_positive_groups                                        |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_input_channel_mismatch                                     |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_groups_do_not_divide_channels                              |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_empty_weight                                               |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_bias_shape_mismatch                                        |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_rank_mismatch                                              |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_non_positive_stride                                        |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_non_positive_dilation                                      |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_negative_padding                                           |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_output_padding_too_large                                   |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_kernel_larger_than_padded_input                            |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_dilated_kernel_larger_than_input                           |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_rejects_complex                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution.py                                    | test_convolution_autograd_input_is_detached                                 |      1 |      0 |       0 |        1 |
| tests/test_copysign_.py                                      | test_copysign_                                                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor                                                    |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_views                                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_invalid_size                                       |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_negative_dimension_message                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_pin_memory_ignored                                 |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_non_strided_layout_rejected                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dtype_matrix                                   |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_resizes_mismatched_out                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_non_contiguous_gapped                          |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_transposed_view                                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_complex_gapped                                 |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_zero_size                                      |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_cpu                                            |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_cpu_non_contiguous                             |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_zero_tensor_rejected                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_requires_grad_rejected                         |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_meta                                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_negative_size                                  |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_registered_dispatch_path                           |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_registered_dispatch_path                       |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dispatch_non_contiguous                        |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dispatch_complex                               |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dispatch_complex_non_contiguous                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dispatch_zero_tensor_rejected                  |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_downstream_dispatch                                |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_inplace_view_rejected                              |      1 |      0 |       0 |        1 |
| tests/test_efficientzerotensor.py                            | test_efficientzerotensor_out_dispatch_resize_growth                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_attention_forward.py                        | test_cudnn_attention_forward                                                |      1 |      0 |       0 |        1 |
| tests/test_cummin.py                                         | test_cummin                                                                 |      1 |      0 |       0 |        1 |
| tests/test_cummin.py                                         | test_cummin_with_nan                                                        |      1 |      0 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py                  | test__sparse_semi_structured_apply                                          |      1 |      0 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py                  | test_sparse_semi_structured_apply_dispatch                                  |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_strided                                                            |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_strided_views                                                      |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_sparse_compressed                                                  |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_dtype_and_device_independence                                      |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_requires_grad_and_meta                                             |      1 |      0 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_arg_validation                                                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_rejects_dense_mask                                         |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dense_input_raises                                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_csr_input_raises                                            |      1 |      0 |       0 |        1 |
| tests/test_euclidean_dist.py                                 | test_euclidean_dist                                                         |      1 |      0 |       0 |        1 |
| tests/test_eye.py                                            | test_eye                                                                    |      1 |      0 |       0 |        1 |
| tests/test_eye.py                                            | test_eye_m                                                                  |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py     | test_accuracy_fake_quantize_per_channel_affine_cachemask                    |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py     | test_accuracy_fake_quantize_per_channel_affine_cachemask_half_to_even       |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py     | test_accuracy_fake_quantize_per_channel_affine_cachemask_noncontiguous      |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py     | test_accuracy_fake_quantize_per_channel_affine_cachemask_out                |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_channel_affine_cachemask.py     | test_accuracy_fake_quantize_per_channel_affine_cachemask_empty              |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                | test_feature_dropout                                                        |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                | test_feature_dropout_no_train                                               |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                | test_feature_dropout_p_zero                                                 |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                | test_feature_dropout_p_one                                                  |      1 |      0 |       0 |        1 |
| tests/test_feature_dropout.py                                | test_feature_dropout_inplace                                                |      1 |      0 |       0 |        1 |
| tests/test_fix_.py                                           | test_fix_                                                                   |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                         | test_accuracy_flipud                                                        |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                         | test_accuracy_flipud_non_float                                              |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                         | test_accuracy_flipud_autograd                                               |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                         | test_accuracy_flipud_noncontiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_flipud.py                                         | test_flipud_rejects_zero_dimensional_input                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_fast_path                                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_fallback_cases                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_int_fallback                                       |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_writes_through_view                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_rejects_empty_list                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_fast_path                                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_half_to_even                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_large_magnitude                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_fallback_cases                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_writes_through_view                              |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_rejects_empty_list                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_fast_path                                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_fallback_cases                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_int_fallback                                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_writes_through_view                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_rejects_empty_list                                |      1 |      0 |       0 |        1 |
| tests/test_frac.py                                           | test_frac                                                                   |      1 |      0 |       0 |        1 |
| tests/test_functional_sym_constrain_range.py                 | test_functional_sym_constrain_range                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_rng_bridge_build_failure                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_python_reservation_matches_native                        |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_python_reservation_contract                              |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout                                                          |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_layout                                                   |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_generator_replay_and_independence                        |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_generator_word_increment                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_generator_counter_carry                                  |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_generator_offset_overflow_leaves_state_unchanged         |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_default_generator_replay                                 |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_empty_default_rng                                        |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_rejects_nonfloating_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_rejects_wrong_generator_device                           |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_concurrent_generator_reservations                        |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_checkpoint_replays_default_rng                           |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_rejects_graph_capture_without_consuming_rng              |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_special_values                                           |      1 |      0 |       0 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py | test_combine_topk_swa_indices_accuracy                                      |      1 |      0 |       0 |        1 |
| tests/test_ge.py                                             | test_ge                                                                     |      1 |      0 |       0 |        1 |
| tests/test_ge.py                                             | test_ge_scalar                                                              |      1 |      0 |       0 |        1 |
| tests/test_glu_backward_jvp.py                               | test_glu_backward_jvp                                                       |      1 |      0 |       0 |        1 |
| tests/test_glu_backward_jvp.py                               | test_glu_backward_jvp_broadcast_dgrad                                       |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                | test_grid_sampler_2d                                                        |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                | test_grid_sampler_2d_oh_ne_iw                                               |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                | test_grid_sampler_2d_nearest_half_integer                                   |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler_2d.py                                | test_grid_sampler_2d_dim_one_reflection                                     |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                   | test_column_stack                                                           |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                   | test_column_stack_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                                   | test_exception_column_stack                                                 |      1 |      0 |       0 |        1 |
| tests/test_hardswish.py                                      | test_hardswish_                                                             |      1 |      0 |       0 |        1 |
| tests/test_hardswish.py                                      | test_hardswish                                                              |      1 |      0 |       0 |        1 |
| tests/test_hardswish.py                                      | test_hardswish_out                                                          |      1 |      0 |       0 |        1 |
| tests/test_hardswish.py                                      | test_hardswish_special_values                                               |      1 |      0 |       0 |        1 |
| tests/test_heaviside.py                                      | test_heaviside_tensor                                                       |      1 |      0 |       0 |        1 |
| tests/test_heaviside.py                                      | test_heaviside_out                                                          |      1 |      0 |       0 |        1 |
| tests/test_igamma_.py                                        | test_igamma_                                                                |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac                                                                |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_out                                                            |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_boundary_x_zero                                                |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_boundary_a_one                                                 |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_boundary_large_x                                               |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_extreme_asym                                                   |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_extreme_asym_large                                             |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_inf_nan                                                        |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_extreme_ratios                                                 |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_large_a_small_x                                                |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_series_cf_boundary                                             |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_asym_threshold                                                 |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_large_a_moderate_x                                             |      1 |      0 |       0 |        1 |
| tests/test_igammac.py                                        | test_igammac_log_uniform                                                    |      1 |      0 |       0 |        1 |
| tests/test_index_put_impl_.py                                | test__index_put_impl__acc_false                                             |      1 |      0 |       0 |        1 |
| tests/test_index_put_impl_.py                                | test__index_put_impl__unsafe_param                                          |      1 |      0 |       0 |        1 |
| tests/test_index_put_impl_.py                                | test__index_put_impl__error_all_none                                        |      1 |      0 |       0 |        1 |
| tests/test_is_all_true.py                                    | test_is_all_true                                                            |      1 |      0 |       0 |        1 |
| tests/test_is_all_true.py                                    | test_accuracy_is_all_true_empty                                             |      1 |      0 |       0 |        1 |
| tests/test_isnan.py                                          | test_isnan                                                                  |      1 |      0 |       0 |        1 |
| tests/test_kron.py                                           | test_kron                                                                   |      1 |      0 |       0 |        1 |
| tests/test_le_.py                                            | test_le_                                                                    |      1 |      0 |       0 |        1 |
| tests/test_le_.py                                            | test_le_scalar_                                                             |      1 |      0 |       0 |        1 |
| tests/test_dswiglu.py                                        | test_dswiglu                                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals_out_dtype_mismatch                                      |      1 |      0 |       0 |        1 |
| tests/test_gelu_and_mul.py                                   | test_gelu_and_mul                                                           |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                          | test_geqrf                                                                  |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                          | test_geqrf_non_contiguous                                                   |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                          | test_geqrf_return_structure                                                 |      1 |      0 |       0 |        1 |
| tests/test_geqrf.py                                          | test_geqrf_zero_and_tiny_sizes                                              |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                          | test_expm1                                                                  |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                          | test_expm1_backward                                                         |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                          | test_expm1_backward_expanded_grad                                           |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                          | test_expm1_                                                                 |      1 |      0 |       0 |        1 |
| tests/test_expm1.py                                          | test_expm1_out                                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_ind2                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_ind1                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_non_spd                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_zero_diagonal_pivot                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_blocked_path                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_ind3                                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_ind1_higher_dim                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_singular                                              |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_non_positive_ind                                      |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_invalid_input                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_non_contiguous                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_non_spd_blocked_path                                  |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_blocked_path_singular                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_input_not_mutated                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_tensorinv.py                               | test_linalg_tensorinv_out_overwrites_and_non_contiguous                     |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10                                                                  |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_                                                                 |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_out                                                              |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_special_values                                                   |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_empty                                                            |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_noncontiguous                                                    |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_int_promotes_to_float                                            |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_backward                                                         |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_backward_expanded_grad                                           |      1 |      0 |       0 |        1 |
| tests/test_log10.py                                          | test_log10_backward_special_values                                          |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                                    | test_log_softmax                                                            |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                                    | test_accuracy_log_softmax_out                                               |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                                    | test_log_softmax_backward_data                                              |      1 |      0 |       0 |        1 |
| tests/test_log_softmax.py                                    | test_accuracy_log_softmax_backward_out                                      |      1 |      0 |       0 |        1 |
| tests/test_logical_xor.py                                    | test_logical_xor                                                            |      1 |      0 |       0 |        1 |
| tests/test_divide.py                                         | test_divide                                                                 |      1 |      0 |       0 |        1 |
| tests/test_margin_ranking_loss.py                            | test_margin_ranking_loss                                                    |      1 |      0 |       0 |        1 |
| tests/test_margin_ranking_loss.py                            | test_margin_ranking_loss_backward                                           |      1 |      0 |       0 |        1 |
| tests/test_masked_softmax_backward.py                        | test_masked_softmax_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_max_pool1d.py                                     | test_max_pool1d                                                             |      1 |      0 |       0 |        1 |
| tests/test_max_pool1d.py                                     | test_max_pool1d_default_stride                                              |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test_conj_physical                                                          |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test_conj_physical_complex                                                  |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_real_does_not_alias                                     |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_empty                                                   |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_matches_backward_formula                                |      1 |      0 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_out_rejects_wrong_dtype                                 |      1 |      0 |       0 |        1 |
| tests/test_miopen_batch_norm.py                              | test_miopen_batch_norm                                                      |      1 |      0 |       0 |        1 |
| tests/test_miopen_batch_norm.py                              | test_miopen_batch_norm_eval                                                 |      1 |      0 |       0 |        1 |
| tests/test_moe_align_block_size.py                           | test_accuracy_moe_align_block_size                                          |      1 |      0 |       0 |        1 |
| tests/test_moe_align_block_size.py                           | test_accuracy_moe_align_block_size_triton                                   |      1 |      0 |       0 |        1 |
| tests/test_moe_align_block_size.py                           | test_accuracy_moe_align_block_size_fast_paths                               |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss                                                      |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_forward_backward_consistency                         |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_out_variants                                         |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_empty_batch                                          |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_unbatched_none_is_scalar                             |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_noncontiguous_inputs                                 |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_rejects_invalid_reduction                            |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_rejects_invalid_p                                    |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_validates_shapes_and_dtypes                          |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_requires_vector_target_for_2d_input                  |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_requires_vector_weight                               |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_out_of_bounds_target_in_subprocess                   |      1 |      0 |       0 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_backward_out_of_bounds_target_in_subprocess          |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean                                                                |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_dim                                                            |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_multi_dim                                                      |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_dim_out                                                        |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_edge                                                           |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_empty_and_dtype                                                |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_zero_sized_free_dimensions                                     |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_dtype_downcast                                                 |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_invalid_dims                                                   |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_out_resizes_and_returns_out                                    |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_out_dtype_downcast                                             |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_out_rejects_explicit_dtype_mismatch                            |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_out_rejects_device_mismatch                                    |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_backward_all_nan_slice                                         |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_out_rejects_autograd                                           |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_complex_input_real_dtype                                       |      1 |      0 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_backward_dim                                                   |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention                                            |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_self_attention                             |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_no_weights                                 |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_no_average                                 |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_src_mask                                   |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_key_padding_mask                           |      1 |      0 |       0 |        1 |
| tests/test_native_multi_head_attention.py                    | test_native_multi_head_attention_generic_mask                               |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_basic                        |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_int64_sizes                  |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_empty_and_scalar_paths       |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_zero_components              |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_int32_truncation             |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_non_contiguous_raw_read      |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_dtype_error                  |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_dim_error                    |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_constructs_nested_tensor     |      1 |      0 |       0 |        1 |
| tests/test_nested_compute_contiguous_strides_offsets.py      | test_nested_compute_contiguous_strides_offsets_dispatch_and_repeatability   |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_stored_offsets_with_gap                  |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_empty_nested                             |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_cpu_nested                               |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_non_nested_error                         |      1 |      0 |       0 |        1 |
| tests/test_new_zeros_with_same_feature_meta.py               | test_new_zeros_with_same_feature_meta                                       |      1 |      0 |       0 |        1 |
| tests/test_new_zeros_with_same_feature_meta.py               | test_new_zeros_with_same_feature_meta_invalid_batch_dims                    |      1 |      0 |       0 |        1 |
| tests/test_nll_loss_nd.py                                    | test_nll_loss_nd_forward                                                    |      1 |      0 |       0 |        1 |
| tests/test_nll_loss_nd.py                                    | test_nll_loss_nd_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_normed_cumsum.py                                  | test_normed_cumsum                                                          |      1 |      0 |       0 |        1 |
| tests/test_normed_cumsum.py                                  | test_normed_cumsum_output_is_normalized                                     |      1 |      0 |       0 |        1 |
| tests/test_or_tensor.py                                      | test_or_tensor                                                              |      1 |      0 |       0 |        1 |
| tests/test_pad_enum.py                                       | test__pad_enum_constant                                                     |      1 |      0 |       0 |        1 |
| tests/test_pad_enum.py                                       | test__pad_enum_reflect                                                      |      1 |      0 |       0 |        1 |
| tests/test_pad_enum.py                                       | test__pad_enum_replicate                                                    |      1 |      0 |       0 |        1 |
| tests/test_pad_enum.py                                       | test__pad_enum_circular                                                     |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                                 | test_pdist_backward                                                         |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                                 | test_pdist_backward_p1                                                      |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                                 | test_pdist_backward_pinf                                                    |      1 |      0 |       0 |        1 |
| tests/test_pdist_backward.py                                 | test_pdist_backward_general                                                 |      1 |      0 |       0 |        1 |
| tests/test_pixel_unshuffle.py                                | test_pixel_unshuffle                                                        |      1 |      0 |       0 |        1 |
| tests/test_pixel_unshuffle.py                                | test_pixel_unshuffle_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_rejects_mismatched_metadata                   |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_rejects_mismatched_device                     |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_validates_normalized_shape                    |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_backward_residual_only                        |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel                                                   |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_last_axis                                         |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_extremes                                          |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_tie_cases                                         |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_float_zero_points                                 |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_float_zero_points_tie_cases                       |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_float_zero_points_qscheme_differs                 |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_out                                               |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_out_matches_aten_out                              |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_out_rejects_wrong_dtype                           |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_out_noncontiguous                                 |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_non_float_input                           |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_float_zero_point_non_float_input          |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_bad_axis                                  |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_cpu_qparams                               |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_multi_dim_qparams                         |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_short_qparams                             |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_integer_scales                            |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_unsupported_dtype                         |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_rejects_out_of_range_zero_point                   |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_zero_point_error_matches_native                   |      1 |      0 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_accepts_valid_zero_points                         |      1 |      0 |       0 |        1 |
| tests/test_rad2deg.py                                        | test_rad2deg                                                                |      1 |      0 |       0 |        1 |
| tests/test_rad2deg.py                                        | test_rad2deg_                                                               |      1 |      0 |       0 |        1 |
| tests/test_range.py                                          | test_range                                                                  |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d.py                               | test_reflection_pad3d                                                       |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d.py                               | test_reflection_pad3d_list_padding                                          |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d.py                               | test_reflection_pad3d_empty_padding                                         |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d.py                               | test_reflection_pad3d_4d_input                                              |      1 |      0 |       0 |        1 |
| tests/test_reflection_pad3d.py                               | test_reflection_pad3d_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_renorm_.py                                        | test_renorm_                                                                |      1 |      0 |       0 |        1 |
| tests/test_replication_pad3d_backward.py                     | test_replication_pad3d_backward_rejects_mismatched_leading_shape            |      1 |      0 |       0 |        1 |
| tests/test_resize_output.py                                  | test_resize_output                                                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_forward_modes                                                 |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_comprehensive_forward_shapes                                  |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_large_hidden                                                  |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_dropout_one                                                   |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_data                                                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh.py                                       | test_rnn_tanh_data_dispatch                                                 |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar                                                           |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_public_api                                                |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_noncontiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_empty                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_rank_deficient                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_invalid_input                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_out_noncontiguous_buffers                                 |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                           | test_rsub_tensor                                                            |      1 |      0 |       0 |        1 |
| tests/test_rsub.py                                           | test_rsub_scalar                                                            |      1 |      0 |       0 |        1 |
| tests/test_scaled_dot_product_attention_math_for_mps.py      | test_scaled_dot_product_attention_math_for_mps                              |      1 |      0 |       0 |        1 |
| tests/test_scaled_softmax.py                                 | test_scaled_softmax_forward                                                 |      1 |      0 |       0 |        1 |
| tests/test_scaled_softmax.py                                 | test_scaled_softmax_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_select_scatter.py                                 | test_select_scatter                                                         |      1 |      0 |       0 |        1 |
| tests/test_select_scatter.py                                 | test_select_scatter_with_self_overlapping_input                             |      1 |      0 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_cuda_runtime_owns_visibility_and_tensor_operations                     |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_hip_runtime_uses_rocm_visibility_names_without_changing_torch_api      |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_maca_runtime_uses_metax_visibility_without_changing_torch_api          |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_musa_runtime_uses_mthreads_visibility_and_torch_api                    |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_runtime_rejects_missing_or_empty_device_api                            |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_runtime_metadata_keeps_architecture_separate_from_platform             |      0 |      0 |       1 |        1 |
| tests/flagtune/test_device_runtime.py                        | test_runtime_resolves_benchmark_only_through_triton_boundary                |      0 |      0 |       1 |        1 |
| tests/test_metax_index_offset.py                             | test_metax_index_large_state_stride                                         |      0 |      0 |       1 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_large_64bit_indexing                                     |      0 |      0 |       1 |        1 |
| tests/test_deepseek_v4_attention_combine_topk_swa_indices.py | test_combine_topk_swa_indices_vllm_accuracy                                 |      0 |      0 |       1 |        1 |
| tests/test_igammac.py                                        | test_igammac_composed_baseline_matches_torch                                |      0 |      0 |       1 |        1 |
| tests/test_multi_margin_loss.py                              | test_multi_margin_loss_hygon_reduced_routes                                 |      0 |      0 |       1 |        1 |
| tests/test_linalg_polar.py                                   | test_linalg_polar_dispatcher                                                |      0 |      0 |       1 |        1 |
| tests/test_dim.py                                            | test_dim_sparse_coo                                                         |      0 |      1 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_nested                                                             |      0 |      1 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_matches_reference_table                                            |      0 |      1 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_aliases_agree                                                      |      0 |      1 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_is_dispatch_free                                                   |      0 |      1 |       0 |        1 |
| tests/test_dim.py                                            | test_dim_with_registration_active                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_value_ranges                                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_sparse_self                                                |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_non_contiguous                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_out                                                        |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_nan_inf                                                    |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_backward                                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_shape_mismatch                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_mask.py                                    | test_sparse_mask_dispatch                                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_default                                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_default_empty                                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_default_uncoalesced                                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dtype                                                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim                                                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_all                                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_dtype                                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_out                                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_negative                                                |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_keeps_structural_zeros                                  |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_hybrid_dense_dims                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_uncoalesced_input                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_zero_size_dims                                              |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_integer_promotion                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_duplicate_dim_raises                                        |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_out_of_range_dim_raises                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dim_out_dtype_mismatch_raises                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_dtype_accumulator_precision                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_complex                                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_dispatch_reaches_gems_impl                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_dispatch_covers_every_overload                                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                                     | test_sparse_sum_uncoalesced_matches_aten                                    |      0 |      1 |       0 |        1 |
| tests/test_embedding_sparse_backward.py                      | test_embedding_sparse_backward                                              |      0 |      1 |       0 |        1 |
| tests/test_flipud.py                                         | test_accuracy_flipud_complex                                                |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_abs_inplace_registration_is_live                               |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_round_inplace_registration_is_live                             |      0 |      1 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                         | test_foreach_sign_inplace_registration_is_live                              |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_reservations_interoperate_with_native_rand               |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                                  | test_fused_dropout_aten_autograd_uses_gems                                  |      0 |      1 |       0 |        1 |
| tests/test_index_put_impl_.py                                | test__index_put_impl__acc_true                                              |      0 |      1 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                         | test_copy_sparse_to_sparse_                                                 |      0 |      1 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                         | test_copy_sparse_to_sparse_self_alias                                       |      0 |      1 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                         | test_copy_sparse_to_sparse_dtype_conversion                                 |      0 |      1 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                         | test_copy_sparse_to_sparse_nonempty_shrink_raises                           |      0 |      1 |       0 |        1 |
| tests/test_copy_sparse_to_sparse_.py                         | test_copy_sparse_to_sparse_nonempty_grow                                    |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test__linalg_eigvals                                                        |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test__linalg_eigvals_complex                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test__linalg_eigvals_batched                                                |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals                                                         |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals_out                                                     |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals_out_resize                                              |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals_out_non_contiguous                                      |      0 |      1 |       0 |        1 |
| tests/test_linalg_eigvals.py                                 | test_linalg_eigvals_out_float64                                             |      0 |      1 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_layout_parity                                           |      0 |      1 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_conj_bit_input                                          |      0 |      1 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_out_layout_parity                                       |      0 |      1 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_out_non_contiguous                                      |      0 |      1 |       0 |        1 |
| tests/test_conj_physical.py                                  | test__conj_physical_out_resizes                                             |      0 |      1 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_real_input_complex_output                                      |      0 |      1 |       0 |        1 |
| tests/test_nanmean.py                                        | test_nanmean_complex_and_autograd                                           |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_accuracy_nested_tensor_storage_offsets                                 |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_from_shared_buffer_views                 |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_scalar_nesting                           |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_aliases_native_metadata                  |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                  | test_nested_tensor_storage_offsets_dispatch_stability                       |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_forward                                       |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_no_grad_with_grad_inputs                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_optional_affine                               |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_large_normalized_shape                        |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_noncontiguous_fallback                        |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_backward                                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_backward_optional_affine                      |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                       | test_post_layer_norm_residual_backward_large_normalized_shape               |      0 |      1 |       0 |        1 |
| tests/test_quantize_per_channel.py                           | test_quantize_per_channel_round_trip                                        |      0 |      1 |       0 |        1 |
| tests/test_replication_pad3d_backward.py                     | test_replication_pad3d_backward                                             |      0 |      1 |       0 |        1 |
| TOTAL                                                        |                                                                             |    463 |     79 |      13 |      555 |
|                        filepath                        |                                function                                | passed | skipped | SUBTOTAL |
| ------------------------------------------------------ | ---------------------------------------------------------------------- | -----: | ------: | -------: |
| tests/test_silu.py                                     | test_silu                                                              |      1 |       0 |        1 |
| tests/test_silu.py                                     | test_silu_                                                             |      1 |       0 |        1 |
| tests/test_silu.py                                     | test_silu_backward                                                     |      1 |       0 |        1 |
| tests/test_pad_circular.py                             | test_pad_circular                                                      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss                                                    |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_backward                                           |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_backward_scalar_grad_output                        |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_backward_beta_zero_equal_inputs_cuda_behavior      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_broadcast                                          |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_special_values                                     |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_out                                                |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_out_reduced                                        |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_functional                                         |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_negative_beta                                      |      1 |       0 |        1 |
| tests/test_smooth_l1_loss.py                           | test_smooth_l1_loss_backward_negative_beta                             |      1 |       0 |        1 |
| tests/test_softshrink_backward.py                      | test_softshrink_backward                                               |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                     | test_sparse_sampled_addmm                                              |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                     | test_sparse_sampled_addmm_large                                        |      1 |       0 |        1 |
| tests/test_sparse_sampled_addmm.py                     | test_sparse_sampled_addmm_out                                          |      1 |       0 |        1 |
| tests/test_special_bessel_j1.py                        | test_special_bessel_j1                                                 |      1 |       0 |        1 |
| tests/test_special_entr.py                             | test_special_entr                                                      |      1 |       0 |        1 |
| tests/test_batch_norm.py                               | test_native_batch_norm                                                 |      1 |       0 |        1 |
| tests/test_batch_norm.py                               | test_batch_norm                                                        |      1 |       0 |        1 |
| tests/test_batch_norm.py                               | test_batch_norm_backward                                               |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p                                                     |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p_non_tensor                                          |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p_out                                                 |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p_negative                                            |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p_nan_inf                                             |      1 |       0 |        1 |
| tests/test_special_log1p.py                            | test_special_log1p_small_values                                        |      1 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py               | test_special_modified_bessel_k1                                        |      1 |       0 |        1 |
| tests/test_special_modified_bessel_k1.py               | test_special_modified_bessel_k1_out                                    |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py   | test_special_shifted_chebyshev_polynomial_u                            |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py   | test_special_shifted_chebyshev_polynomial_u_scalar_n                   |      1 |       0 |        1 |
| tests/test_special_shifted_chebyshev_polynomial_u.py   | test_special_shifted_chebyshev_polynomial_u_                           |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_broadcast                                            |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_out                                                  |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_tensor_scalar                                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_tensor_scalar_out                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_scalar_tensor                                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_scalar_tensor_out                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_special_values                                       |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_numerical_regimes                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_negative_noninteger_q                                |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_integer_promotion                                    |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_half_bfloat16_promotion                              |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_rejects_native_low_precision                         |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_out_exact_alias                                      |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_out_partial_overlap_rejected                         |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_out_casts_after_float_compute                        |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_q_gradient                                           |      1 |       0 |        1 |
| tests/test_special_zeta.py                             | test_special_zeta_x_gradient_is_not_implemented                        |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                 | test_rnn_relu                                                          |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                 | test_rnn_relu_direct_wrapper                                           |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                 | test_rnn_relu_direct_backward                                          |      1 |       0 |        1 |
| tests/test_rnn_relu.py                                 | test_rnn_relu_large_hidden                                             |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_tensor_tensor                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_tensor_scalar                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_tensor_tensor_                                                |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_tensor_scalar_                                                |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_scalar_tensor                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_scalar_scalar                                                 |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_complex_complex                                               |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_complex_float_tensor                                          |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_complex_int_tensor                                            |      1 |       0 |        1 |
| tests/test_sub.py                                      | test_sub_complex_int_scalar                                            |      1 |       0 |        1 |
| tests/test_sym_size.py                                 | test_sym_size                                                          |      1 |       0 |        1 |
| tests/test_tanh.py                                     | test_tanh                                                              |      1 |       0 |        1 |
| tests/test_tanh.py                                     | test_tanh_                                                             |      1 |       0 |        1 |
| tests/test_tanh.py                                     | test_tanh_backward                                                     |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_rejects_unsupported_dtype           |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward                                     |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_optional_grads                      |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_noncontiguous                       |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_end_to_end                          |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_reshaped_workspace                  |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_empty                               |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_special_values                      |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward.py            | test_thnn_fused_lstm_cell_backward_requires_no_grad                    |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy_complex                                   |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy                                           |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy_scalar                                    |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy_non_contiguous                            |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy_same_dim_does_not_alias                   |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_transpose_copy_invalid_dims                                       |      1 |       0 |        1 |
| tests/test_transpose_copy.py                           | test_accuracy_transpose_copy_float8                                    |      1 |       0 |        1 |
| tests/test_triu.py                                     | test_triu                                                              |      1 |       0 |        1 |
| tests/test_triu.py                                     | test_triu_noncontiguous                                                |      1 |       0 |        1 |
| tests/test_triu.py                                     | test_triu_                                                             |      1 |       0 |        1 |
| tests/test_triu.py                                     | test_triu_inplace_noncontiguous                                        |      1 |       0 |        1 |
| tests/test_unfold.py                                   | test_unfold                                                            |      1 |       0 |        1 |
| tests/test_unique_dim.py                               | test_unique_dim_int                                                    |      1 |       0 |        1 |
| tests/test_unique_dim.py                               | test_unique_dim_float                                                  |      1 |       0 |        1 |
| tests/test_unique_dim.py                               | test_unique_dim_returns_three_tensors                                  |      1 |       0 |        1 |
| tests/test_conv1d.py                                   | test_conv1d                                                            |      1 |       0 |        1 |
| tests/test_conv1d.py                                   | test_conv1d_padding                                                    |      1 |       0 |        1 |
| tests/test_conv1d.py                                   | test_conv1d_dilation                                                   |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward                                      |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_grad_input                           |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_edges                                |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_grad_input_resize                    |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_grad_input_errors                    |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_invalid                              |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_scale_contract                       |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_deterministic                        |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_clamped_scale_cpu_reference          |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_grad_input_channels_last             |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_channels_last_cache_reuse            |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_grad_input_channels_last_cache_reuse |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_fp64_opmath                          |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_nonfinite_cpu_reference              |      1 |       0 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_extreme_positive_scales              |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod                                                    |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_int_bool                                           |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_sequence_form                                      |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_non_contiguous                                     |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_row_order                                          |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_backward                                           |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_nan_inf                                            |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_rejects_empty_list                                 |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_rejects_multidim_input                             |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_rejects_mixed_device                               |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_rejects_mixed_dtype                                |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_rejects_non_tensor                                 |      1 |       0 |        1 |
| tests/test_cartesian_prod.py                           | test_cartesian_prod_backward_mixed_requires_grad                       |      1 |       0 |        1 |
| tests/test_upsample_trilinear3d_backward.py            | test_upsample_trilinear3d_backward                                     |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_fresh_tensor                                              |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_bumps_on_inplace_ops                                      |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_inplace_under_no_grad                                     |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_views_share_base_counter                                  |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_noncontiguous_and_strided_views                           |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_dtype_sweep                                               |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_sparse_layouts                                            |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_inference_mode_raises                                     |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_meta_tensor                                               |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_matches_reference_table                                   |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_dispatch_sentinel_composite_key                           |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_disjoint_from_native_neighbors                            |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_non_tensor_input                                          |      1 |       0 |        1 |
| tests/test_version.py                                  | test_version_bump_sequence_matches_native                              |      1 |       0 |        1 |
| tests/test_weight_int4pack_mm_with_scales_and_zeros.py | test_weight_int4pack_mm_with_scales_and_zeros                          |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy                                                             |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_out                                                         |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_special_values                                              |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_tensor_scalar                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_tensor_scalar_out                                           |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_scalar_tensor                                               |      1 |       0 |        1 |
| tests/test_xlogy.py                                    | test_xlogy_scalar_tensor_out                                           |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                           | test_frobenius_norm                                                    |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                           | test_frobenius_norm_zero_size                                          |      1 |       0 |        1 |
| tests/test_frobenius_norm.py                           | test_frobenius_norm_duplicate_dim                                      |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_uses_flag_gems_view_path                                     |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_complex_view_semantics                                       |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_noncontiguous_complex_view                                   |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_non_complex_returns_input                                    |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_float8_returns_input                                         |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_autograd_view_relation                                       |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_conjugate_view_version_tracking                              |      1 |       0 |        1 |
| tests/test_real.py                                     | test_real_conjugate_nonleaf_inplace_view_replay                        |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_                                                           |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_float_float_                                               |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_float_tensor                                               |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_tensor_float                                               |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_honors_generator                                           |      1 |       0 |        1 |
| tests/test_normal.py                                   | test_normal_tensor_tensor                                              |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_                                                      |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_fast                                                  |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_backward_                                             |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_inplace_version                                       |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_lower_tail                                            |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_no_repeated_blocks                                    |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_far_left_tail                                         |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_consecutive_calls_advance_generator                   |      1 |       0 |        1 |
| tests/test_exponential_.py                             | test_exponential_race_matches_categorical                              |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__tensor                                                        |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__scalar                                                        |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__backward                                                      |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__special_values                                                |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__aliased_operand                                               |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__view_receiver                                                 |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__invalid_operand                                               |      1 |       0 |        1 |
| tests/test_ge_.py                                      | test_ge__unsupported_dtype                                             |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim                                                           |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_int_dtype                                                 |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_intlist_overload                                          |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_identity                                                  |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_result_aliases_input                                      |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_non_contiguous_input                                      |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_empty_tensor                                              |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_autograd                                                  |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_out_of_range                                              |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_destination_out_of_range                                  |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_length_mismatch                                           |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_repeated_axis                                             |      1 |       0 |        1 |
| tests/test_movedim.py                                  | test_movedim_0d_out_of_range                                           |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward                                          |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward_dim_sweep                                |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward_cpu_oracle                               |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward_native_dropout                           |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward_empty_gems_cpu_oracle                    |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_scaled_dot_product_cudnn_attention_backward                       |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_efficient_attention_backward                                      |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_scaled_dot_product_efficient_attention_backward                   |      1 |       0 |        1 |
| tests/test_flash_attention_backward.py                 | test_scaled_dot_product_flash_attention_backward                       |      1 |       0 |        1 |
| tests/test_lift_fresh_copy.py                          | test_accuracy_lift_fresh_copy                                          |      1 |       0 |        1 |
| tests/test_lift_fresh_copy.py                          | test_accuracy_lift_fresh_copy_empty                                    |      1 |       0 |        1 |
| tests/test_bitwise_left_shift.py                       | test_bitwise_left_shift                                                |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_1d                                              |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_1d_dilation                                     |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_2d                                              |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_large_kernel                                    |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_2d_dilation                                     |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_2d_asymmetric                                   |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_3d                                              |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_3d_dilation                                     |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_1d_asymmetric                                   |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_3d_asymmetric                                   |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_channel_counts_2d                               |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_channel_counts_1d_3d                            |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_pointwise_escape                                |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_scalar_params                                   |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_output_shape                                    |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_output_shape_grid                               |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_non_contiguous_input                            |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_non_contiguous_weight                           |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_interface                                       |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_edge_cases                                      |      1 |       0 |        1 |
| tests/test_cudnn_convolution.py                        | test_cudnn_convolution_invalid_ndim                                    |      1 |       0 |        1 |
| tests/test_w8a8_block_fp8_matmul.py                    | test_w8a8_block_fp8_matmul                                             |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_cross_attention_matches_reference                                 |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_cross_attention_mask_contract                                     |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_fully_masked_rows_are_exact_zero                                  |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_noncontiguous_qkv_and_mask                                        |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_cross_attention_validation_contract                               |      1 |       0 |        1 |
| tests/test_cross_attention.py                          | test_head_dimension_upper_bound                                        |      1 |       0 |        1 |
| tests/test_values.py                                   | test_accuracy_values_coo                                               |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_mutation_through_view                                      |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_strided_raises                                             |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_sparse_compressed                                          |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_unsupported_layouts_reject_like_native                     |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_non_float_dtypes_alias                                     |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_int_hybrid_alias                                           |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_requires_grad_nondifferentiable_contract                   |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_sparse_requires_grad_view_contract                         |      1 |       0 |        1 |
| tests/test_values.py                                   | test_values_dispatch_stability                                         |      1 |       0 |        1 |
| tests/test_mm.py                                       | test_mm                                                                |      1 |       0 |        1 |
| tests/test_mm.py                                       | test_mm_broadcast_stride_zero                                          |      1 |       0 |        1 |
| tests/test_mm.py                                       | test_mm_out_vllm_tma_column_major_weight                               |      1 |       0 |        1 |
| tests/test_mm.py                                       | test_mm_self_transpose                                                 |      1 |       0 |        1 |
| tests/test_mm.py                                       | test_mm_out_self_transpose                                             |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_uses_platform_appropriate_lock                           |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_flagtune_environment_controls_operator_selection                  |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_tuning_mode_environment_matrix                                    |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_tuning_mode_defers_missing_package_handling_to_policy             |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_legacy_fallback_honors_operator_include                           |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_expanded_request_does_not_probe_platform_package                  |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_adapted_operator_rejects_invalid_cost_model_setting               |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_disabled_flagtune_does_not_probe_platform_package                 |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_adapted_tuning_mode_prefers_cost_model_when_both_enabled          |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_tuning_mode_rejects_non_binary_environment_values                 |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_adapted_libtuner_switches_default_expanded_and_cost_model         |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libtuner_apply_defers_model_availability_to_policy                |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_official_triton_libtuner_uses_unadapted_routes                    |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_decorator_cascade                                                 |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_pass_kernel_arg_via_kw                                            |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_kernel_arg_apply_default                                          |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_threadsafety                                                      |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_descriptor_cache_key_uses_resolved_type                           |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_descriptor_cache_key_without_descriptor_module                    |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_model_bundle_missing_error_supports_older_flagtree                |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_key_without_vendor_hook                                  |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_key_with_hygon_backend                                   |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_key_normalizes_descriptor_arguments                      |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_run_normalizes_keyword_descriptors                       |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_key_orders_specialization_then_dns_then_constexpr        |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_key_buckets_integers_by_triton_width                     |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_launch_options_key_selects_compile_options               |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libentry_run_separates_compile_options                            |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_hash_generation                                                   |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_hash_changes_when_dependency_modified                             |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_flagtree_policy_is_bypassed_when_expanded_flagtune_is_enabled     |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_official_triton_treats_model_annotation_as_unadapted              |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_operator_fuse_invalidates_existing_dispatch_caches                |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_legacy_fallback_prunes_and_restores_tuner_state                   |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_flagtree_policy_is_bypassed_when_cost_model_is_disabled           |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_benchmark_success_count_tracks_finite_uncached_benchmarks         |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_benchmark_key_preserves_raw_shape_and_scopes_timing_protocol      |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_ordinary_libtuner_keeps_legacy_benchmark_arguments                |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_cost_model_protocol_stays_explicit                                |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_replay_fallback_reuses_event_config_cache_namespace               |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_config_cache_namespace_separates_replay_protocols                 |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_adapted_config_cache_namespace_separates_tuning_modes             |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_benchmark_config_reuses_kernel_context_and_bypasses_caches        |      1 |       0 |        1 |
| tests/test_libentry.py                                 | test_libcache_vllm_signal_scenario                                     |      1 |       0 |        1 |
| tests/test_var_mean.py                                 | test_var_mean                                                          |      1 |       0 |        1 |
| tests/test_var.py                                      | test_accuracy_var                                                      |      1 |       0 |        1 |
| tests/test_var.py                                      | test_accuracy_var_dim                                                  |      1 |       0 |        1 |
| tests/test_var.py                                      | test_accuracy_var_correction                                           |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone                                                             |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone_preserve_format_layout                                      |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone_non_contiguous_values                                       |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone_autograd                                                    |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone_autograd_non_contiguous                                     |      1 |       0 |        1 |
| tests/test_clone.py                                    | test_clone_under_torch_compile                                         |      1 |       0 |        1 |
| tests/test_diff.py                                     | test_diff                                                              |      1 |       0 |        1 |
| tests/test_diff.py                                     | test_diff_3d                                                           |      1 |       0 |        1 |
| tests/test_diff.py                                     | test_diff_prepend_append                                               |      1 |       0 |        1 |
| tests/test_diff.py                                     | test_diff_second_order_rounding                                        |      1 |       0 |        1 |
| tests/test_index_copy_.py                              | test_index_copy                                                        |      1 |       0 |        1 |
| tests/test_index_copy_.py                              | test_index_copy_                                                       |      1 |       0 |        1 |
| tests/test_pad.py                                      | test_pad                                                               |      1 |       0 |        1 |
| tests/test_zero.py                                     | test_zero                                                              |      1 |       0 |        1 |
| tests/test_zero.py                                     | test_zero_                                                             |      1 |       0 |        1 |
| tests/test_zero.py                                     | test_zero_out                                                          |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward                                 |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward_with_scale                      |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward_grad_input                      |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward_empty                           |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward_uint8                           |      1 |       0 |        1 |
| tests/test_upsample_nearest_exact1d_backward.py        | test_upsample_nearest_exact1d_backward_invalid_size                    |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean                                                              |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_empty                                                        |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_empty_explicit_dtype                                         |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_empty_rejects_non_floating_input                             |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_empty_rejects_non_floating_dtype                             |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_dim                                                          |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_dim_empty                                                    |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_dim_empty_explicit_dtype                                     |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_dim_large_k                                                  |      1 |       0 |        1 |
| tests/test_mean.py                                     | test_mean_dim_large_innerdim                                           |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_operator_markers_are_declared                                     |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_ops                                              |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_ops_registration_key_exists                               |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_ops_registration_is_live                                  |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_paired_noncontiguous                             |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_ops_length_mismatch_rejected                              |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_reduction_orders                                 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_zero_writes_through_views                        |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_ops_launch_count_is_independent_of_list_length            |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_int_promotion                                    |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_inplace_rejects_promotion                                 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_ops_dtype_sets_match_aten                                 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_bool                                             |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_add_alpha                                        |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_alpha_keeps_one_launch                                    |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_copy_mixed_dtype                                 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_max_large_values                                 |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_norm_orders                                      |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_norm_all_zero                                    |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_accuracy_foreach_pow_bool_base                                    |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_pow_list_rejects_bool                                     |      1 |       0 |        1 |
| tests/test_foreach_ops.py                              | test_foreach_clamp_scalar_rejects_bool                                 |      1 |       0 |        1 |
| tests/test_silu.py                                     | test_silu_edge_cases                                                   |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_silu_forward_backward_edge                                        |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_forward_contract                                       |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_grid_balance_shapes                                    |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_empty                                                  |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_special_values                                         |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_noncontiguous_uses_common_fallback                     |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_public_path_record                                     |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_inplace_keeps_alias                                    |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_direct_backward_stays_common                           |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_registrar_scope                                        |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_supported_contiguous_uses_single_route                 |      0 |       1 |        1 |
| tests/test_silu.py                                     | test_metax_silu_large_noncontiguous_uses_generic_fallback              |      0 |       1 |        1 |
| tests/test_topk_softmax.py                             | test_topk_softmax                                                      |      0 |       1 |        1 |
| tests/test_upsample_bilinear2d_backward.py             | test_upsample_bilinear2d_backward_unsupported_fp64                     |      0 |       1 |        1 |
| tests/test_flash_attention_backward.py                 | test_flash_attention_backward_saved_state_arithmetic_range             |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_large_shapes                                            |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_transposed_inputs_and_k_tails                           |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_large_shape_entrypoints                                 |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_empty                                            |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_out_resize                                       |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_capped_routes                                    |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_narrow_real_core_dim                             |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_panel_loops                                      |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_broadcast_materialize                            |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_specialized_launch_cap                           |      0 |       1 |        1 |
| tests/test_mm.py                                       | test_mm_ascend_review_out_validation                                   |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_flagtree_policy_uses_cost_model_by_default_for_adapted_operator   |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_enabled_flagtree_policy_propagates_contract_failures              |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_cost_model_failures_fuse_only_auto                                |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_auto_source_failure_warns_once_retries_and_recovers               |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_cost_model_singleton_does_not_load_or_swallow_legacy_failure      |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_flagtree_proposer_cache_tracks_resolved_model_version             |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_hopper_mm_config_compiles_without_runtime_registration            |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_mul_config_compiles_and_binds_runtime_kernels                     |      0 |       1 |        1 |
| tests/test_libentry.py                                 | test_libcache_concurrent_write_on_signal                               |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_slow_path_bounds                                             |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_fast_path_tail                                               |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_strided_input                                                |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_empty_boundaries                                             |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_zero_order_copy                                              |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_large_second_order                                           |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_graph_replay                                                 |      0 |       1 |        1 |
| tests/test_diff.py                                     | test_diff_launch_boundaries                                            |      0 |       1 |        1 |
| TOTAL                                                  |                                                                        |    352 |      45 |      397 |
