|                             filepath                             |                                  function                                   | passed | failed | skipped | xfailed | SUBTOTAL |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- | -----: | -----: | ------: | ------: | -------: |
| tests/core/test_external_baseline.py                             | test_nested_metadata_matches_even_when_shape_key_is_a_descriptor            |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_tensor_identity_mismatch_does_not_use_first_shape_shortcut             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_scalar_type_and_value_mismatches_fall_back                             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_ambiguity_does_not_select_first_record                                 |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_named_inputs_require_binding_and_compare_scalars                       |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_legacy_shape_and_microsecond_units_and_private_operator_names          |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_invalid_latency_is_a_miss                                              |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_invalid_file_warns_and_disables_comparison                             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_missing_file_warns_and_valid_metadata_is_preserved                     |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_scaling_selects_chip_and_resource_and_prefers_measured                 |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_invalid_scaling_falls_back_to_raw_ratio                                |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_external_ratio_preserves_native_measurements_and_records_provenance    |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_skip_native_can_use_external_data_and_a_miss_keeps_na                  |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_real_summary_prefers_external_per_case_and_preserves_fallback          |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_malformed_metadata_does_not_fall_through_to_legacy_shape               |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_malformed_resource_and_oversized_latency_do_not_break_benchmarks       |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_factory_device_matches_selected_backend_without_weakening_other_inputs |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_thead_runtime_vendor_uses_ops_baseline_t_head_scaling                  |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_h200_does_not_use_h20_scaling                                          |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_device_normalization_requires_the_selected_target                      |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_factory_normalization_still_checks_all_other_parameters                |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_named_device_metadata_uses_the_same_target_rule                        |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_overlapping_models_select_the_exact_chip_and_preserve_suffixes         |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_vendor_alias_is_explicit_and_does_not_match_other_vendors              |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_malformed_kwargs_metadata_falls_back_without_crashing                  |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_legacy_shape_cannot_establish_multi_input_or_parameter_abi             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_legacy_sglang_packed_silu_and_mul_does_not_match_two_tensor_gems_abi   |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_incomplete_named_metadata_cannot_choose_a_semantic_baseline            |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_external_baseline.py                             | test_complete_named_metadata_still_matches_and_checks_all_parameters        |      1 |      0 |       0 |       0 |        1 |
| tests/test__pin_memory.py                                        | test__pin_memory                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test__pin_memory.py                                        | test__pin_memory_registered_on_cpu_key                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_moe_load_balance_loss_matches_reference                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_tle_path_with_irregular_large_shape                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_noncontiguous_logits_and_mask                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_all_masked_tokens_return_zero                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_default_top_k_is_two                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_moe_load_balance_loss.py                              | test_moe_load_balance_loss_validation                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_addmv.py                                              | test_addmv                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_addmv.py                                              | test_addmv_out                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                    | test_embedding_spmdm_matches_embedding_bag                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                    | test_embedding_spmdm_index_dtypes                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                    | test_embedding_spmdm_noncontiguous_inputs                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding_spmdm.py                                    | test_embedding_spmdm_validation                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_fmax_backward.py                                      | test_fmax_backward                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_fmax_backward.py                                      | test_fmax_backward_one_input                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_fmax_backward.py                                      | test_fmax_backward_grad_fn                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_bartlett_window.py                                    | test_bartlett_window                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_.py                                            | test_cumsum_                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_conj_physical_.py                                     | test_conj_physical_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cummin.py                                             | test_cummin                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_cummin.py                                             | test_cummin_with_nan                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_non_float_input                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_float_zero_point_non_float_input          |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_bad_axis                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_cpu_qparams                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_multi_dim_qparams                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_short_qparams                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_integer_scales                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_unsupported_dtype                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_rejects_out_of_range_zero_point                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                                           | test_corrcoef                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                                           | test_corrcoef_1d                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_corrcoef.py                                           | test_corrcoef_non_contiguous                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_dtype                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_low_precision_multitile                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_long_row_int16                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_low_precision_multitile                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_non_contiguous                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_non_contiguous_tile_boundary                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_dtype_mismatch                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_inplace_bool_unsupported                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumprod.py                                            | test_cumprod_backward                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rng_bridge_build_failure                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_python_reservation_matches_native                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_python_reservation_contract                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_layout                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_replay_and_independence                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_word_increment                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_counter_carry                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_generator_offset_overflow_leaves_state_unchanged         |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_default_generator_replay                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_empty_default_rng                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_nonfloating_dtype                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_wrong_generator_device                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_concurrent_generator_reservations                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_aten_autograd_uses_gems                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_checkpoint_replays_default_rng                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_special_values                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter_offset                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter_negative_offset                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter_non_last_dims                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter_large_offset                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_diagonal_scatter.py                                   | test_diagonal_scatter_non_square                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_empty_input                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_rejects_bad_request                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                           | test_copysign                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                           | test_copysign_out                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                           | test_copysign_byte                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                            | test_deg2rad                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                            | test_deg2rad_                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                            | test_deg2rad_out                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                            | test_deg2rad_out_byte                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine_backward.py | test_fake_quantize_learnable_per_tensor_affine_backward                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_learnable_per_tensor_affine_backward.py | test_fake_quantize_learnable_per_tensor_affine_backward_clamped             |      1 |      0 |       0 |       0 |        1 |
| tests/test_kl_div.py                                             | test_kl_div                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_kl_div.py                                             | test_kl_div_zero_target                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_kl_div.py                                             | test_kl_div_broadcast                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_kl_div.py                                             | test_kl_div_functional                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe_dtype                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_bsr_tensor_unsafe.py                           | test_sparse_bsr_tensor_unsafe_skips_index_validation                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fliplr.py                                             | test_accuracy_fliplr                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fliplr.py                                             | test_accuracy_fliplr_non_float                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fliplr.py                                             | test_accuracy_fliplr_noncontiguous                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_fliplr.py                                             | test_fliplr_rejects_inputs_with_fewer_than_two_dims                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_full.py                                               | test_full                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_adam.py                                         | test_fused_adam                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_adam.py                                         | test_fused_adam_                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                            | test_inplace_saved_version_is_checked                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_gather_block_quantized.py                             | test_gather_block_quantized                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_coo_all_sparse                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_coo_hybrid                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_matches_indices_formula                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_empty                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_single_entry                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_uncoalesced                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_nan_inf_values_ignored                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_dense_raises                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_compressed_layouts_raise                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_dimI.py                                               | test_dimI_dispatch_stability                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_gt_tensor_.py                                         | test_gt_tensor_                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_gt_tensor_.py                                         | test_gt_tensor__byte                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardswish_backward.py                                 | test_hardswish_backward                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_hinge_embedding_loss.py                               | test_hinge_embedding_loss                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_huber_loss.py                                         | test_huber_loss                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_huber_loss.py                                         | test_huber_loss_out                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_ilshift.py                                            | test_ilshift                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_ilshift.py                                            | test_ilshift_int8                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_noncontiguous                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_duplicate_index_short_source                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_unique_index                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_empty_index                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_noncontiguous_out_of_place                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_out                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_out_aliases_input                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_empty_index_out_of_place                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_index_reduce.py                                       | test_index_reduce_invalid_reduce                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_2x2_kernel                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_jacobi                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_batch_2x2_kernel                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_batch_jacobi                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_trivial                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_batch_trivial                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_2x2_low_precision                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_uplo_upper                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_jacobi_global                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_ascending_order                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_nonsquare_raises                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_non_contiguous                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_underlying_linalg_eigh                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_underlying_linalg_eigh_no_vectors                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_underlying_linalg_eigh_no_vectors_2x2                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_underlying_linalg_eigh_no_vectors_global                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_gradients                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_matches_closed_form_exactly                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_tiled_path_matches_fused                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_closed_form                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_eigenvalue_only                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_batched                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_uplo                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_uses_autograd_function                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_requires_grad_mode                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_compute_v_false                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_trivial                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_non_contiguous_grad                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_expanded_grad                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_repeated_eigenvalues_non_finite                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_isfinite.py                                           | test_isfinite                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_jagged_to_padded_dense_forward.py                     | test_jagged_to_padded_dense_forward                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_jagged_to_padded_dense_forward.py                     | test_jagged_to_padded_dense_forward_padding                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_cpu_scalar_operand                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_dtype_promotion                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_broadcast_noncontiguous_and_empty                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_special_values                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_alias_resize_and_stride                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_dtype_casting                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_aliasing                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_rejects_partial_overlap                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_rejects_device_mismatch                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_out_rejects_invalid_dtype                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                        | test_less_equal_                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                        | test_less_equal_scalar_                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                        | test_less_equal__byte                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                        | test_less_equal_scalar__byte                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                        | test_linalg_cond                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                        | test_linalg_cond_with_p                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                        | test_linalg_cond_svd                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_ldl_factor_ex.py                               | test_linalg_ldl_factor_ex                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_edge                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_1d_rejected                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_same_dim_rejected                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_unsupported_ord_rejected                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_out                                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_slogdet.py                                     | test_linalg_slogdet                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vecdot.py                                      | test_linalg_vecdot                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_vecdot.py                                      | test_linalg_vecdot_out                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_large_inplace                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_special_values                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_noncontiguous                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_empty                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_.py                                               | test_log_unsupported_dtype_raises                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_logaddexp2.py                                         | test_logaddexp2                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_logaddexp2.py                                         | test_logaddexp2_out                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_logaddexp2.py                                         | test_logaddexp2_non_finite                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit.py                                              | test_logit                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit.py                                              | test_logit_                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit.py                                              | test_logit_out                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_logit.py                                              | test_logit_out_non_contiguous_out                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_singular                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_singular_positions                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_empty                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_info_shape                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_empty_batch_dimensions                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_extended_dtypes                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_broadcast                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_broadcast_metadata                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_incompatible_broadcast                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_errors                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_large_integer_values                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_forward                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_forward_no_indices                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_int_output_size                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_nan                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_ties                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_all_negative                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_mixed_sign                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_all_zero                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_no_dim                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_complex_no_dim                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_complex_nonempty_errors                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_reduced_dim_raises                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_scalar_dim                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_boundaries                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_all_equal                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_error_paths                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_mse_loss.py                                           | test_mse_loss                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply_.py                                          | test_multiply_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply_.py                                          | test_multiply_tensor_scalar_                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_special_values_and_scalar_tensor_q                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_errors                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_sum_backward.py                                | test_nested_sum_backward                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_sum_backward.py                                | test_nested_sum_backward_edge                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_sum_backward.py                                | test_nested_sum_backward_non_contiguous                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_new_full.py                                           | test_new_full                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_nll_loss_forward.py                                   | test_nll_loss_forward                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_normal_functional.py                                  | test_normal_functional                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_single                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_multi                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_infers_terminal_total_L                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_empty_output                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_noncontiguous_dense                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_float16_preserves_bits                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_validation                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_revalidates_mutated_offsets             |      1 |      0 |       0 |       0 |        1 |
| tests/test_permute_copy.py                                       | test_permute_copy                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_default                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_no_opmath                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_int_to_float                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_always_bool                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_complex_to_long                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pointwise_type_promotion.py                           | test_type_promotion_bool_to_long                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_prelu_kernel.py                                       | test_prelu_kernel                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_non_contiguous                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_zero_sized                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_autograd_graph                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_keyword_arguments                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_reflection_pad1d_backward.py                          | test_reflection_pad1d_backward                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_relu6_.py                                             | test_relu6_                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_output.py                                      | test_resize_output                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_rot90.py                                              | test_rot90                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_rrelu_with_noise_backward.py                          | test_rrelu_with_noise_backward                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_uniform_alpha                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_skewed_alpha                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_1d                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_k1                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_large_k                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_sample_dirichlet.py                                   | test_sample_dirichlet_tiny_alpha                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                  | test_scaled_grouped_mm                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                  | test_scaled_grouped_mm_fp8_core                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                  | test_scaled_grouped_mm_fp8_encodings                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                  | test_scaled_grouped_mm_scale_before_output_cast                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                                  | test_scaled_grouped_mm_reduction_addressing                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_lengths                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_offsets                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_backward_lengths                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_lengths_out                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_offsets_out                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_backward_lengths_out                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_backward_offsets_out                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_backward_offsets                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_segment_reduce.py                                     | test_segment_reduce_indices_unsupported                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                               | test_tanh                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                               | test_tanh_                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                               | test_tanh_backward                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_skip_layer_norm.py                                    | test_skip_layernorm                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sort.py                                               | test_sort                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_sort.py                                               | test_sort_stable                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_entr.py                                       | test_special_entr                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_sparse_layout                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_bool_extent                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_rejects_non_tensor_self                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_rejects_non_sequence_size                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_log_ndtr.py                                   | test_special_log_ndtr                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_ndtri.py                                      | test_special_ndtri                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_ndtri.py                                      | test_special_ndtri_edge_values                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_ndtri.py                                      | test_special_ndtri_non_contiguous                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_ndtri.py                                      | test_special_ndtri_unsupported_dtype                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_sinc.py                                       | test_special_sinc                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sqrt.py                                               | test_sqrt                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_sqrt.py                                               | test_sqrt_                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad_near_mode                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad_away_from_mode                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad_small_output                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad_mismatched_dtype_raises                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_standard_gamma_grad.py                                | test_standard_gamma_grad_integer_dtype_raises                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mm_w8a8_int8_strides                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_scaled_empty_bias                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_tensor_scales_without_bias                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_prequantized                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_long_k_overflow                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mixed_scales                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_graph_updates                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_invalid                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_tan.py                                                | test_tan                                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_tan.py                                                | test_tan_                                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell.py                               | test_thnn_fused_lstm_cell                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell.py                               | test_thnn_fused_lstm_cell_with_bias                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_range.py                                              | test_range                                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_random                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_random_batch                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_positive_definite                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_negative_determinant                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_diagonal                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_singular                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_empty                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_non_contiguous                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_det.py                                                | test_det_errors                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_training_cli_requires_config_and_variant                               |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_training_cli_rejects_unsafe_variant                                    |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_training_cli_rejects_negative_progress_interval                        |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_status_and_disabled_progress_have_explicit_output_contract             |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_training_rejects_same_platform_with_different_architectures            |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_training_exports_platform_identity_and_retained_architecture           |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_routed_training_separates_model_dtype_groups                           |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_routed_training_excludes_unsupported_rows                              |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_routed_training_does_not_export_empty_or_failed_collection             |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_collection_rows_are_flattened_to_streaming_training_jsonl              |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_mul_collection_rows_reuse_derived_variant_inputs                       |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_collection_failure_uses_workload_not_model_input_dimensions            |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_collection_failure_reports_missing_workload_dimensions                 |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_generic_config_timing_serialization_uses_triton_quantile_order         |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_generic_config_progress_interval_handles_environment_values            |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_worker_log_forwarding_buffers_partial_lines                            |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_flagtune_train.py                            | test_worker_environment_contains_device_and_database_overrides              |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_published_regression.py                      | test_published_regression                                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_cslt_sparse_mm.py                                     | test_accuracy_cslt_sparse_mm                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_cslt_sparse_mm.py                                     | test_accuracy_cslt_sparse_mm_with_alpha                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_cslt_sparse_mm.py                                     | test_accuracy_cslt_sparse_mm_transpose                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_reservations_interoperate_with_native_rand               |      0 |      0 |       1 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_large_64bit_indexing                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_fused_dropout.py                                      | test_fused_dropout_rejects_graph_capture_without_consuming_rng              |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dropout_pos_rejected                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dropout_neg_degrades_to_zero                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_scale                                         |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_attn_bias_shapes                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_dual_dim_bias                                 |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_onesided_cumseq_rejected                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_varlen_rejected                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_cudnn_attention_backward.py                           | test_cudnn_attention_backward_value_head_dim                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_get_paged_mqa_logits_metadata.py                      | test_get_paged_mqa_logits_metadata                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_zero_dim_promotion                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_complex                                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_2d                                                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_2d_keepdim                                                             |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_batch                                                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_batch_nondefault_dim                                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_dtype_param                                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_norm.py                                 | test_large                                                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_extra_no_dim_dtypes                                             |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim                                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_full_registration                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_large                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_no_dim_beyond_old_flat_limit                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_bool_dim_count_selects_first_index                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_empty_output_unsupported_dtype                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_float64_key_select_nan_first_index                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out_wrong_device                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_padded_dense_to_jagged_forward.py                     | test_padded_dense_to_jagged_forward_float64_preserves_bits                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mm_w8a8_int8                                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mm_w8a8_int8_updates                                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mm_w8a8_int8_empty                                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_mm_w8a8_int8_reject_float                                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_scaled_bias                                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_scaled_graph_bias_updates                                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_scaled_validation                                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                       | test_scaled_unaligned_weight                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_convolution_overrideable_backward.py                  | test_convolution_overrideable_backward                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_conj_copy.py                                          | test_conj_copy                                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_conj_copy.py                                          | test_conj_copy_out                                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_conj_copy.py                                          | test_conj_copy_with_conj_bit                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_last_axis                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_extremes                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_tie_cases                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_round_trip                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_float_zero_points                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_float_zero_points_tie_cases                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_float_zero_points_qscheme_differs                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_out                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_out_matches_aten_out                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_out_rejects_wrong_dtype                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_out_noncontiguous                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_zero_point_error_matches_native                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_channel.py                               | test_quantize_per_channel_accepts_valid_zero_points                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_dropout.py                                            | test_dropout                                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_dropout.py                                            | test_native_dropout_backward                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod                                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_ignores_unstored_zeros                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_stored_zero_wins                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_dtype                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_unsorted_columns                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_empty_rows_and_columns                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_prod.py                                    | test_sparse_csr_prod_out                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                            | test_backward_real_dispatch                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_atanh_backward_dispatch.py                            | test_backward_uses_gems_compute                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_complex                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_trivial_complex                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_batch_complex                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_uplo_upper_complex                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_complex_global                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_underlying_linalg_eigh_no_vectors_complex_global                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigh.py                                        | test_linalg_eigh_backward_complex_not_implemented                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_integral_exponent_boundaries                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_real_operands_complex_out                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                              | test_ldexp_lazy_conjugation                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_info_ignores_nonfinite_pivots                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_info_reports_nan_only_as_nonzero_when_zero_absent                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_lu_with_info.py                                       | test_lu_with_info_exact_zero_pivot_position                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_kernel_boundaries                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_mask_extremes                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_preserves_layout                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_broadcast_nonzero_storage_offset                |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_accuracy_masked_select_backward_noncontiguous_inputs                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_masked_select_backward.py                             | test_masked_select_backward_conjugated_grad                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim                                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_nan                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_lastdim_sort                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_lastdim_sort_nan                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_direct_flat_nan                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_tie_lower_median                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_duplicate_indices_select_value                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_reduction_boundary_dim0                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_public_shapes_dim0                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_nan_first_index_dim0                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_non_contiguous_dim0                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_direct_named_dim0_preserves_names                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_full_registration_nan_semantics                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_non_contiguous                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_high_dim_semantics                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width                                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_float64_key_select                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_extended_lastdim_width                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_lastdim_sort_unique_exact_index                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_lastdim_sort_1536_unique_exact_index                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_boundaries                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_unique_exact_index                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_nan_first_index                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_duplicates_infinities_and_zeros                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp32_key_select_signed_zero_bits                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_boundaries                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_fp16_key_select_unique_exact_index                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_nan_first_index                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_width640_keepdim                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_duplicates_infinities_and_zeros                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_signed_zero_index_bits                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_f16_key_select_width640_nonlast_and_no_dim                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width_nan_first_index                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_no_dim_fallback_nan                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_non_lastdim_fallback_nan_first_index                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_large_reduction_semantics                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_large_reduction_out                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_strided_nonlast_named_dim_preserves_names                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_large_width_duplicate_indices_select_value                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_int_lastdim_select_unique_exact_index                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_out                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_out_error_paths                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_dim_values_out_python_api                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_ties_and_special_values                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_nan                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_non_contiguous                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_is_a_graph_node                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_ties_follow_saved_index                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_nan_follows_saved_index                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_requires_grad_inputs                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_large                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_no_dim_large_duplicate_median                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_median.py                                             | test_median_backward_dim_non_contiguous                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_miopen_batch_norm_backward.py                         | test_miopen_batch_norm_backward                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_tensor                                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_scalar                                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_out                                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_scalar_out                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_dim_none_keepdim_shape                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_maximum_reduction_one_valid                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_large_q_is_tiled                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_gather_interpolations                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_noncontiguous_out                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_nanquantile.py                                        | test_nanquantile_all_nan_and_path_boundaries                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_native_norm.py                                        | test_native_norm                                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_ormqr.py                                              | test_ormqr                                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_padding_neutral                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_all_padding_window                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_non_contiguous                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_storage_offset                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_resizes_and_adopts_quant_params               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_is_input                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_rejects_mismatched_out                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_out_rejects_wrong_device                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_invalid_params                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_invalid_shapes                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_empty_batch                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_non_contiguous_input                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_quant_params                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantized_max_pool1d.py                               | test_quantized_max_pool1d_unit_length                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_invalid_input                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_complex_rejected                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_replication_pad2d.py                                  | test_replication_pad2d                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_replication_pad2d.py                                  | test_replication_pad2d_out                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_replication_pad2d.py                                  | test_replication_pad2d_3d_input                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_sobol_engine_ff_.py                                   | test_sobol_engine_ff_                                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_bessel_j1.py                                  | test_special_bessel_j1                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros                                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_inherits_metadata                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_source_is_not_read_or_mutated                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_dtype_override                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_explicit_device                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_size_argument_forms                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_out                                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_out_resizes_buffer                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_out_writes_through_strided_buffer                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_new_zeros.py                                          | test_new_zeros_empty_source_still_supplies_defaults                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_adaptive_max_pool3d.py                                | test_accuracy_adaptive_max_pool3d_empty_output                              |      0 |      0 |       0 |       1 |        1 |
| TOTAL                                                            |                                                                             |    360 |    151 |      60 |       1 |      572 |
|                      filepath                      |                              function                               | passed | failed | skipped | SUBTOTAL |
| -------------------------------------------------- | ------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_true_divide.py                          | test_true_divide                                                    |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_                                                   |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_out                                                |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_tensor_scalar                                      |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_tensor_scalar_                                     |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_scalar_tensor                                      |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_tensor_dispatch                                    |      1 |      0 |       0 |        1 |
| tests/test_true_divide.py                          | test_true_divide_tensor_inplace_dispatch                            |      1 |      0 |       0 |        1 |
| tests/test_unfold_copy.py                          | test_unfold_copy                                                    |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack                                                         |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_int_bool                                                |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_zero_sized                                              |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_element_order                                           |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_nan_inf                                                 |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_mixed_dtype_promotion                                   |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_mixed_real_complex                                      |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_rejects_mismatched_shapes                               |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_rejects_empty_list                                      |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_rejects_non_tensor                                      |      1 |      0 |       0 |        1 |
| tests/test_dstack.py                               | test_dstack_backward                                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d.py                   | test_upsample_bicubic2d                                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py       | test_upsample_lanczos2d_aa_backward                                 |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py       | test_upsample_lanczos2d_aa_backward_dtypes                          |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py       | test_upsample_lanczos2d_aa_backward_precomputed_path                |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py       | test_upsample_lanczos2d_aa_backward_out                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py       | test_upsample_lanczos2d_aa_backward_noncontiguous_grad              |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d.py             | test_accuracy__upsample_nearest_exact1d                             |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py   | test_value_selecting_reduction_backward                             |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py   | test_value_selecting_reduction_backward_max_dim                     |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py   | test_value_selecting_reduction_backward_min_dim                     |      1 |      0 |       0 |        1 |
| tests/test_vstack.py                               | test_accuracy_vstack                                                |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py               | test_wrapped_linear_prepack                                         |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py               | test_wrapped_linear_prepack_rejects_invalid_shapes                  |      1 |      0 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py | test_native_batch_norm_legit_no_training                            |      1 |      0 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py | test_native_batch_norm_legit_no_training_backward                   |      1 |      0 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py | test_native_batch_norm_legit_no_training_backward_input_only        |      1 |      0 |       0 |        1 |
| tests/test_adaptive_avg_pool2d_backward.py         | test_adaptive_avg_pool2d_backward                                   |      1 |      0 |       0 |        1 |
| tests/test_addcdiv.py                              | test_addcdiv                                                        |      1 |      0 |       0 |        1 |
| tests/test_allclose.py                             | test_allclose                                                       |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                              | test_arccosh                                                        |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                              | test_arccosh_special_values                                         |      1 |      0 |       0 |        1 |
| tests/test_arccosh.py                              | test_arccosh_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_cpu                                                |      1 |      0 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_sparse                                             |      1 |      0 |       0 |        1 |
| tests/test_atan2_.py                               | test_atan2_                                                         |      1 |      0 |       0 |        1 |
| tests/test_batch_norm.py                           | test_batch_norm                                                     |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py        | test_binary_cross_entropy_backward                                  |      1 |      0 |       0 |        1 |
| tests/test_binary_cross_entropy_backward.py        | test_binary_cross_entropy_backward_weighted                         |      1 |      0 |       0 |        1 |
| tests/test_bitwise_not.py                          | test_bitwise_not                                                    |      1 |      0 |       0 |        1 |
| tests/test_bitwise_not.py                          | test_bitwise_not_                                                   |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_tensor.py                   | test_bitwise_xor_tensor                                             |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_tensor.py                   | test_bitwise_xor_tensor_extremes                                    |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_tensor.py                   | test_bitwise_xor_tensor_                                            |      1 |      0 |       0 |        1 |
| tests/test_bitwise_xor_tensor.py                   | test_bitwise_xor_tensor__extremes                                   |      1 |      0 |       0 |        1 |
| tests/test_cdist.py                                | test_cdist                                                          |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve                                                 |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_larger_shapes                                   |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_rhs_boundaries                                  |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_upper                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_transpose_contiguous_factor                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_blocked_single_rhs                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_ascend_batched_small_lower_single_rhs           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_ascend_blocked_single_rhs_conditioned           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_fp32_blocked_upper                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_fp32_blocked_lower                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_batch                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_broadcast_batch                                 |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_noncontiguous_inputs                            |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_scaled_inputs                                   |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_conditioned_matrix                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_accuracy                                        |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_direct                                          |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out                                             |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out_resizes_for_broadcast_result                |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out_noncontiguous_and_alias                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out_rejects_incompatible_dtype                  |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out_rejects_different_device                    |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_empty_input                                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_invalid_inputs                                  |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                         | test_clamp_tensor                                                   |      1 |      0 |       0 |        1 |
| tests/test_clamp_tensor.py                         | test_clamp_tensor_                                                  |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_coo_plain                                                  |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_coo_hybrid                                                 |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_matches_values_formula                                     |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_empty                                                      |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_single_entry                                               |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_uncoalesced                                                |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_zero_values_still_count                                    |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_compressed_layouts                                         |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_compressed_batched                                         |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_compressed_zero_nnz                                        |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_dense_raises                                               |      1 |      0 |       0 |        1 |
| tests/test_nnz.py                                  | test_nnz_dispatch_stability                                         |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_accuracy_nested_tensor_size                                    |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_dispatch_stability                          |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_from_strided_buffer                         |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_empty_nested                                |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_scalar_nesting                              |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_cpu_nested                                  |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_non_nested_error                            |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_size.py                   | test_nested_tensor_size_registered_on_autograd_key                  |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose2d.py                | test_slow_conv_transpose2d                                          |      1 |      0 |       0 |        1 |
| tests/test_slow_conv_transpose2d.py                | test_slow_conv_transpose2d_out_variant                              |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy                                            |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_noncontiguous_input                        |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_strided_input                              |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_high_rank_noncontiguous                    |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_broadcast_input                            |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_storage_offset                             |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_independent_storage                        |      1 |      0 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_tail_chunk                                 |      1 |      0 |       0 |        1 |
| tests/test_take_along_dim.py                       | test_take_along_dim                                                 |      1 |      0 |       0 |        1 |
| tests/test_take_along_dim.py                       | test_take_along_dim_dtypes                                          |      1 |      0 |       0 |        1 |
| tests/test_take_along_dim.py                       | test_take_along_dim_none                                            |      1 |      0 |       0 |        1 |
| tests/test_take_along_dim.py                       | test_take_along_dim_noncontiguous                                   |      1 |      0 |       0 |        1 |
| tests/test_take_along_dim.py                       | test_take_along_dim_out                                             |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py               | test_unique_dim_consecutive_int                                     |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py               | test_unique_dim_consecutive_float                                   |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py               | test_unique_dim_consecutive_edge                                    |      1 |      0 |       0 |        1 |
| tests/test_unique_dim_consecutive.py               | test_unique_dim_consecutive_wide_rows                               |      1 |      0 |       0 |        1 |
| tests/test_and_tensor.py                           | test_and_tensor                                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_transpose.py          | test_cudnn_convolution_transpose                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_operator_markers_are_declared                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_float                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_int                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_complex                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_rejects_unsupported_dtype               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_list_lengths                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_mixed_shapes                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_empty_and_zero_dim                      |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_mixed_dtypes                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_inplace                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_inplace_writes_through_views            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_inplace_rejects_overlap                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_inplace_rejects_promotion               |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_error_parity                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_out_overload                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_foreach_unary_registration_is_live                             |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_foreach_unary_launch_count_is_independent_of_list_length       |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_foreach_unary_launch_count_scales_with_dtype_groups            |      1 |      0 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_foreach_unary_launch_count_with_heterogeneous_numel            |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu                                     |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_alpha_types                         |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_bias_shape                          |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_special_values                      |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_empty                               |      1 |      0 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_out                                 |      1 |      0 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_dispatch                              |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__basic                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__options                                         |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__grad_scale                                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__empty                                           |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__mismatched_tensor_properties                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__later_entry_bad_dtype_or_device                 |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__later_entry_bad_state_step                      |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__multi_tensor                                    |      1 |      0 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__internal_overlap_rejected                       |      1 |      0 |       0 |        1 |
| tests/test_randn.py                                | test_randn                                                          |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner                                                          |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_1d_dot                                                   |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_matvec                                                   |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_empty_contraction                                        |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_2d                                                       |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_noncontiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_dtype_mismatch                                           |      1 |      0 |       0 |        1 |
| tests/test_inner.py                                | test_inner_contraction_mismatch                                     |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                           | test_zeros_like_launch_core_dim_limit                               |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                           | test_zeros_like_factory_capped_grid_exact                           |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                           | test_zeros_like_ascend_review_options                               |      1 |      0 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte                                                      |      1 |      0 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_fractional                                           |      1 |      0 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_integer_limits                                       |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm                                                     |      1 |      0 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_fp32_affine                                         |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_scalar_scalar                                              |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_scalar_int                                          |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_scalar_int_float_scalar                             |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_scalar_bool                                         |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_scalar_scalar_float                                        |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_tensor_                                             |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_scalar_                                             |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_broadcast_shape                                            |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_large_tensor_scalar                                        |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_large_tensor_scalar_dtypes                                 |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_large_tensor_scalar_inplace                                |      1 |      0 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_complex_grid_meta_review                                   |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_zero_sized_trailing_dims                             |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_non_castable_dtype_raises                        |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_device_mismatch_raises                           |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_overlapping_input_raises                         |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_shape_mismatch_raises                                |      1 |      0 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_empty_list_raises                                    |      1 |      0 |       0 |        1 |
| tests/test_wrapped_linear_prepack.py               | test_wrapped_linear_prepack_rounds_ties_to_even_and_clamps          |      0 |      1 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py | test_native_batch_norm_legit_no_training_backward_noncontiguous     |      0 |      1 |       0 |        1 |
| tests/test__native_batch_norm_legit_no_training.py | test_native_batch_norm_legit_no_training_backward_saved_stats_dtype |      0 |      1 |       0 |        1 |
| tests/test_addcdiv.py                              | test_addcdiv_out                                                    |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double                                                    |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_strided                                            |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_special_values                                     |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_integer_limits                                     |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_autograd                                           |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_complex                                            |      0 |      1 |       0 |        1 |
| tests/test_cast_Double.py                          | test_cast_Double_negative_view                                      |      0 |      1 |       0 |        1 |
| tests/test_batch_norm.py                           | test_native_batch_norm                                              |      0 |      1 |       0 |        1 |
| tests/test_batch_norm.py                           | test_batch_norm_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_sparse_semi_structured_tile.py          | test_sparse_semi_structured_tile                                    |      0 |      1 |       0 |        1 |
| tests/test_split_copy.py                           | test_accuracy_split_copy_error_paths                                |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy                                     |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_dtype_mapping                       |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_values                              |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_noncontiguous                       |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_transposed                          |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_high_rank_noncontiguous             |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_expanded                            |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_storage_offset                      |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_1d_strided                          |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_independent_storage                 |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_conj                                |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_empty                               |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_zero_dim                            |      0 |      1 |       0 |        1 |
| tests/test_view_as_real_copy.py                    | test_accuracy_view_as_real_copy_error_paths                         |      0 |      1 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_bool                                    |      0 |      1 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_layouts                                 |      0 |      1 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_unary_complex_large                           |      0 |      1 |       0 |        1 |
| tests/test_foreach_unary.py                        | test_accuracy_foreach_abs_complex_large                             |      0 |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_reference                           |      0 |      1 |       0 |        1 |
| tests/test_cudnn_convolution_add_relu.py           | test_cudnn_convolution_add_relu_invalid_ndim                        |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                        | test_rowwise_prune                                                  |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                        | test_rowwise_prune_all_kept                                         |      0 |      1 |       0 |        1 |
| tests/test_rowwise_prune.py                        | test_rowwise_prune_none_kept                                        |      0 |      1 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight                                       |      0 |      1 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_flatten_parameters                    |      0 |      1 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_out                                   |      0 |      1 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_other_dtypes                          |      0 |      1 |       0 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_errors                                |      0 |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__found_inf                                       |      0 |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__mixed_empty_and_nonempty                        |      0 |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__mismatched_list_lengths                         |      0 |      1 |       0 |        1 |
| tests/test_fused_adagrad_.py                       | test_fused_adagrad__transposed_parity                               |      0 |      1 |       0 |        1 |
| tests/test_inner.py                                | test_inner_scalar                                                   |      0 |      1 |       0 |        1 |
| tests/test_zeros_like.py                           | test_zeros_like                                                     |      0 |      1 |       0 |        1 |
| tests/test__ctc_loss.py                            | test__ctc_loss_accuracy                                             |      0 |      1 |       0 |        1 |
| tests/test__ctc_loss.py                            | test__ctc_loss_blank_and_zero_infinity                              |      0 |      1 |       0 |        1 |
| tests/test__ctc_loss.py                            | test__ctc_loss_tensor_overload                                      |      0 |      1 |       0 |        1 |
| tests/test__ctc_loss.py                            | test__ctc_loss_out                                                  |      0 |      1 |       0 |        1 |
| tests/test__ctc_loss.py                            | test__ctc_loss_out_resize                                           |      0 |      1 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_strided                                              |      0 |      1 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_requires_grad                                        |      0 |      1 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_complex                                              |      0 |      1 |       0 |        1 |
| tests/test_cast_Byte.py                            | test_cast_Byte_negative_view                                        |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_native_layer_norm                                              |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_native_layer_norm_statistics                                   |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_backward_large_m                                    |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_backward_output_mask                                |      0 |      1 |       0 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_backward_from_forward                               |      0 |      1 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_tensor                                              |      0 |      1 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_tensor_scalar                                              |      0 |      1 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_scalar_tensor                                              |      0 |      1 |       0 |        1 |
| tests/test_mul.py                                  | test_mul_complex_grid_stride_review                                 |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack                                                      |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_memory_format                                        |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_mixed_dtype_promotion                                |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out                                                  |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_memory_format                                    |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_non_contiguous                                   |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_resize                                           |      0 |      1 |       0 |        1 |
| tests/test_row_stack.py                            | test_row_stack_out_dtype_cast                                       |      0 |      1 |       0 |        1 |
| tests/test_underscore_sobol_engine_draw.py         | test_accuracy_underscore_sobol_engine_draw                          |      0 |      1 |       0 |        1 |
| tests/test_underscore_sobol_engine_draw.py         | test_accuracy_underscore_sobol_engine_draw_with_offset              |      0 |      1 |       0 |        1 |
| tests/test_underscore_sobol_engine_draw.py         | test_accuracy_underscore_sobol_engine_draw_edge_cases               |      0 |      1 |       0 |        1 |
| tests/test_underscore_sobol_engine_draw.py         | test_accuracy_underscore_sobol_engine_draw_large                    |      0 |      1 |       0 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_small_gather_single_rhs                         |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_fp64_blocked                                    |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex                                         |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_noncontiguous                           |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_broadcast                               |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_small_gather_conditioned                |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_blocked                                 |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_blocked_conditioned                     |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_lazy_conjugate_upper                    |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_complex_lazy_conjugate_with_all_gems            |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                       | test_cholesky_solve_out_safe_dtype_cast                             |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                   | test_rms_norm_w8a16_fp8_weight_updates                              |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                   | test_rms_norm_w8a16_fp8_encodings                                   |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                   | test_rms_norm_w8a16_fp8                                             |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                   | test_rms_norm_w8a16_fp8_mthreads_shapes                             |      0 |      0 |       1 |        1 |
| tests/test_cudnn_rnn_flatten_weight.py             | test_cudnn_rnn_flatten_weight_non_current_device                    |      0 |      0 |       1 |        1 |
| tests/test_randn.py                                | test_randn_large_output_offsets                                     |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_backward_mthreads_vendor_dispatch                   |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_contract                                             |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_strides                                              |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_special_values                                       |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_real_shape                                           |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_errors                                               |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_exact_segments                                       |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_core_shapes                                          |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layernorm_large_normalized_extents                             |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_saved_statistics_backward                           |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_native_layer_norm_fp32_statistics                              |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_nvidia_paths                                        |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_fp32_affine_noncontiguous                           |      0 |      0 |       1 |        1 |
| tests/test_layer_norm.py                           | test_layer_norm_empty_rows                                          |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_complex_complex                                            |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_complex_float_tensor                                       |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_complex_int_tensor                                         |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_complex_int_scalar                                         |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_hopper_broadcast_survives_monkeypatched_next_power_of_2    |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_basic                                             |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_broadcast                                         |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_noncontiguous_and_selective                       |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_mixed_dtype                                       |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_complex                                           |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_registered_dispatch                               |      0 |      0 |       1 |        1 |
| tests/test_mul.py                                  | test_mul_backward_fused_kernel_count                                |      0 |      0 |       1 |        1 |
| TOTAL                                              |                                                                     |    193 |     80 |      43 |      316 |
