|                                filepath                                |                                function                                 | passed | failed | skipped | xfailed | SUBTOTAL |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------- | -----: | -----: | ------: | ------: | -------: |
| tests/core/test_reference_only.py                                      | test_benchmark_runs_original_baseline_once                              |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_backward_reference_uses_original_grad_semantics                    |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_skip_native_is_not_a_pass_or_failure                               |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_failure_is_not_passed                                    |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_capability_failures_are_recorded_individually_and_later_cases_run  |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_failure_categories_do_not_guess_api_or_dtype_support               |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_capability_failure_does_not_continue_after_sync_failure            |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_interrupt_propagates_and_leaves_unexecuted_cases         |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_capability_failure_before_invocation_keeps_zero_call_count         |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_source_skip_preserves_node_semantics_and_unexecuted_records        |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_unsupported_custom_runner_cannot_execute_candidate                 |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_conflicting_modes_are_rejected                                     |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_report_replaces_stale_data_and_preserves_case_skips      |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_unknown_case_selection_fails                             |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_partial_calls_before_pytest_skip_do_not_claim_complete_readiness   |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_mode_cannot_run_correctness_pytest                       |      1 |      0 |       0 |       0 |        1 |
| tests/core/test_reference_only.py                                      | test_reference_only_defaults_to_core_without_changing_normal_defaults   |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_inference_candidate_domain.py                      | test_run_proposer_filters_runtime_configs_outside_model_domain          |      1 |      0 |       0 |       0 |        1 |
| tests/test__pin_memory.py                                              | test__pin_memory                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test__pin_memory.py                                              | test__pin_memory_registered_on_cpu_key                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_infinitely_differentiable_gelu_backward.py                  | test_infinitely_differentiable_gelu_backward                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_affine_grid_generator_backward.py                           | test_affine_grid_generator_backward_2d                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_non_contiguous                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_zero_sized                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_invalid_input                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_autograd_graph                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_keyword_arguments                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh_backward.py                                            | test_cosh_backward                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh_backward.py                                            | test_cosh_backward_expanded_grad                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh_backward.py                                            | test_cosh_backward_noncontiguous_input                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh_backward.py                                            | test_cosh_backward_empty                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_non_contiguous                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_zero_sized                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_invalid_input                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_autograd_graph                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_keyword_arguments                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_backend_heuristics.py                                       | test_non_nvidia_heuristic_import_failure_does_not_fallback              |      1 |      0 |       0 |       0 |        1 |
| tests/test_backend_heuristics.py                                       | test_mthreads_softmax_non_inner_uses_mthreads_heuristic                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_empty                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_invalid_shape                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_rejects_low_precision                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_rejects_complex                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_column_stack.py                                             | test_column_stack                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_column_stack.py                                             | test_column_stack_out                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_column_stack.py                                             | test_exception_column_stack                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_autograd_chain                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_non_contiguous                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_middle_and_negative_dim                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_empty                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_scalar_input                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cumsum_backward.py                                          | test_cumsum_backward_use_gems_entry                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_empty                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_invalid_shape                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_invalid_uplo                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_rejects_complex                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_out                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_special_values                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_empty                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_even_property                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_conv_tbc_backward.py                                        | test_conv_tbc_backward                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_convolution_overrideable.py                                 | test_convolution_overrideable                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_convolution_overrideable.py                                 | test_convolution_overrideable_out                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_1d                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_correction                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_fweights                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_aweights                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_fweights_aweights                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov_non_contiguous                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable_backward.py | test_scaled_dot_product_fused_attention_overrideable_backward           |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_dot_product_fused_attention_overrideable_backward.py | test_scaled_dot_product_fused_attention_overrideable_backward_mask      |      1 |      0 |       0 |       0 |        1 |
| tests/test_digmma.py                                                   | test_digamma_                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_argwhere.py                                                 | test_argwhere                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_argwhere.py                                                 | test_argwhere_edge_shapes                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_argwhere.py                                                 | test_argwhere_special_inputs                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding.py                                                | test_embedding                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_embedding.py                                                | test_embedding_backward                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                                 | test_copysign                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                                 | test_copysign_out                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_copysign.py                                                 | test_copysign_byte                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                                  | test_deg2rad                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                                  | test_deg2rad_                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                                  | test_deg2rad_out                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_deg2rad.py                                                  | test_deg2rad_out_byte                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                                  | test_digamma                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                                  | test_digamma_byte                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_digamma.py                                                  | test_digamma_int_rejects_inplace                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_tensor_qparams_rounding                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_tensor_qparams_extreme_values                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams           |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.py | test_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_disabled  |      1 |      0 |       0 |       0 |        1 |
| tests/test_fix.py                                                      | test_fix                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_flatten.py                                                  | test_flatten                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_fmin.py                                                     | test_fmin                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_fmin.py                                                     | test_fmin_out                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_frac.py                                                     | test_frac                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_functional_sym_constrain_range.py                           | test_functional_sym_constrain_range                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                                    | test_cholesky_solve_helper                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                                    | test_cholesky_solve_helper_layouts                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                                    | test_cholesky_solve_helper_batch_strides                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_cholesky_solve_helper.py                                    | test_cholesky_solve_helper_empty                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                           | test_fused_rms_norm                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                           | test_fused_rms_norm_no_weight                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                           | test_fused_rms_norm_aten_dispatch                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                           | test_fused_rms_norm_backward                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_fused_rms_norm.py                                           | test_fused_rms_norm_backward_no_weight                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal_scalar                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal_scalar_byte                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal_                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal__byte                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal_scalar_                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal_scalar__byte                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardsigmoid.py                                              | test_hardsigmoid_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardsigmoid.py                                              | test_hardsigmoid__non_contiguous                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardsigmoid.py                                              | test_hardsigmoid__special_values                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardsigmoid.py                                              | test_hardsigmoid                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_hardsigmoid.py                                              | test_hardsigmoid_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_hypot_.py                                                   | test_hypot_                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_singleton_uncoalesced_values       |      1 |      0 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_index_dtype_message                |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_nonzero.py                                               | test_is_nonzero                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_is_nonzero.py                                               | test_is_nonzero_exception                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_isposinf.py                                                 | test_isposinf                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_cpu_scalar_operand                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_dtype_promotion                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_broadcast_noncontiguous_and_empty                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_special_values                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_alias_resize_and_stride                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_dtype_casting                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_aliasing                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_rejects_partial_overlap                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_rejects_device_mismatch                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_out_rejects_invalid_dtype                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                              | test_less_equal_                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                              | test_less_equal_scalar_                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                              | test_less_equal__byte                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_less_equal_.py                                              | test_less_equal_scalar__byte                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                              | test_linalg_cond                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                              | test_linalg_cond_with_p                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cond.py                                              | test_linalg_cond_svd                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_default_identity                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_well_separated_spectrum                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_rank_deficient                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_nonempty_zero_matrix                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_shapes                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_matches_adjoint                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_aah_svd_matches_hermitian              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_tolerance_combinations                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_scalar_tolerance_types                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_legacy_float_tolerance                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_per_batch_tolerance                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_broadcast_tolerance                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_false                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_true                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_uses_lower_triangle          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_order_33                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_shapes                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_general_large_shapes                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_general_dense_low_rank                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_dense_low_rank               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_empty                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_out                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_tensor_tolerance_out                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_legacy_float_tolerance_out             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_legacy_tensor_tolerance_out            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_out_wrong_dtype                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_out_wrong_device                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_out_wrong_shape_warns_and_resizes      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_fp64_rejected                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_no_native_fp64_k32                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_rejects_complex_tolerance              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_ascend_slow_decay_spectrum             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_nonsquare_tail_energy                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_nonsquare_lowrank                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_strict_threshold             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_negative_tolerances                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_ascend_long_rectangular                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_deflated_spectrum            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_ascend_shape_boundaries                |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_hermitian_ignores_strict_upper_large   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_extreme_scales                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_mixed_magnitude_batch                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_empty_validates_tolerances             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_lower_left                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_upper_left                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_right                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_unitriangular                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_batched                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_batched_multiblock                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_grid_above_core_count                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_linalg_solve_triangular_out                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_empty                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_large_n_f64                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linspace.py                                                 | test_linspace                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_sigmoid.py                                              | test_log_sigmoid                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_logical_and.py                                              | test_logical_and                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_logical_and.py                                              | test_logical_and_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_accuracy_logsumexp_backward                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_logsumexp_backward_matches_aten_composite                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_accuracy_logsumexp_backward_large_and_neg_inf                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_accuracy_logsumexp_backward_non_contiguous                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_accuracy_logsumexp_backward_empty_reduction                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_logsumexp_backward_no_grad_input                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_accuracy_logsumexp                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_tensor                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_scalar                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_tensor_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_scalar_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_scalar_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_backward_                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_scalar_sliced_view                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_sliced_view_tensor                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                                    | test_slow_conv_transpose3d                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                                    | test_slow_conv_transpose3d_bias                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                                    | test_slow_conv_transpose3d_strided_bias                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                                    | test_slow_conv_transpose3d_output_padding_dilation                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_slow_conv_transpose3d.py                                    | test_slow_conv_transpose3d_scalar_stride                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_axis                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_device_default_cpu              |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_requires_quantized_dtype        |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_rejects_bad_qparams             |      1 |      0 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_rejects_string_memory_format    |      1 |      0 |       0 |       0 |        1 |
| tests/test_mse_loss_backward.py                                        | test_mse_loss_backward                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_i0.py                                                       | test_i0                                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_i0.py                                                       | test_i0_                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_i0.py                                                       | test_i0_out                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_i0.py                                                       | test_i0_backward                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_dim                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_dim_out                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_out                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nansum.py                                                   | test_nansum_edge                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_ne_.py                                                      | test_ne_                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_ne_.py                                                      | test_ne_scalar_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_tensor_from_mask_left_aligned.py                     | test_nested_tensor_from_mask_left_aligned                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_tensor_from_mask_left_aligned.py                     | test_nested_tensor_from_mask_left_aligned_edge                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_nan                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_inf                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_finfo_extremes                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_scalar_y                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_scalar_x                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_scalar_nan_y                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_scalar_inf_y                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nonzero_numpy.py                                            | test_nonzero_numpy                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_not_equal_.py                                               | test_accuracy_not_equal_                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_not_equal_.py                                               | test_accuracy_not_equal_scalar_                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_accuracy_2d                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_accuracy_3d                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_shape_consistency                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_custom_padding                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_default_inf_padding                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_single_batch                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_short_sequences                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_3d_single_batch                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_3d_short_sequences                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_block_sizes                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_int8_custom_padding                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_int8_valid_region_with_default_padding                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_int8_out_of_range_padding_is_unvalidated                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_patch_util.py                                               | test_vllm_extension_legacy_and_stable_abi_fallback                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_pixel_shuffle.py                                            | test_pixel_shuffle                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_polygamma.py                                                | test_polygamma                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_polygamma.py                                                | test_polygamma_wide_domain                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_polygamma.py                                                | test_polygamma_out                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_polygamma.py                                                | test_polygamma_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_keeps_layout_not_values                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_explicit_zeros                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_empty_input                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_dtype                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_unsorted_columns                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_rejects_bad_request                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_reciprocal.py                                               | test_reciprocal                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_reciprocal.py                                               | test_reciprocal_                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_relu.py                                                     | test_relu                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_relu.py                                                     | test_relu_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                               | test_replication_pad1d_backward                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                               | test_replication_pad1d_backward_grad_output_dtype_mismatch              |      1 |      0 |       0 |       0 |        1 |
| tests/test_replication_pad1d_backward.py                               | test_replication_pad1d_backward_grad_input_takes_input_options          |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_mismatched_numel                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_non_broadcastable                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as__non_broadcastable                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_empty                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_non_contiguous                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_resize_as.py                                                | test_resize_as_non_contiguous_template                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_roll.py                                                     | test_roll_single_dim                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_roll.py                                                     | test_roll_multi_dims                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_roll.py                                                     | test_roll_flatten                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_roll.py                                                     | test_roll_with_non_dense_input                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_rrelu_with_noise_functional.py                              | test_rrelu_with_noise_functional                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_scalar_tensor.py                                            | test_scalar_tensor                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_import_enumerates_all_cases_without_device_calls                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_ascend_known_fnuz_conversion_is_unsupported                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_unrelated_compiler_errors_are_not_skipped                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_metax_known_native_rowwise_rejection                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_other_native_errors_are_not_skipped                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_resource_and_unknown_errors_propagate                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_unselected_tests_do_not_probe                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_select_scatter.py                                           | test_select_scatter                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_select_scatter.py                                           | test_select_scatter_with_self_overlapping_input                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_silu.py                                                     | test_silu                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_silu.py                                                     | test_silu_edge_cases                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_silu.py                                                     | test_silu_forward_backward_edge                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_silu.py                                                     | test_silu_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_silu.py                                                     | test_silu_backward                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward_oob_end                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_soft_margin_loss.py                                         | test_soft_margin_loss                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_celu.py                                                     | test_celu                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_celu.py                                                     | test_celu_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_i0e.py                                              | test_special_i0e                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_i0e.py                                              | test_special_i0e_out                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_eps_none                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_extreme_values                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_nan_input                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_inf_input                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_of_range                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_eps_none                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_out_of_range                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_extreme_values                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_nan_input                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_logit.py                                            | test_special_logit_out_inf_input                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                        | test_special_scaled_modified_bessel_k1                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                        | test_special_scaled_modified_bessel_k1_out                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k1.py                        | test_special_scaled_modified_bessel_k1_edge_cases                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                                          | test_special_xlog1py                                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                                          | test_special_xlog1py_x_zero                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_xlog1py.py                                          | test_special_xlog1py_y_nan                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_tensor                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_scalar                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_tensor_                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_tensor_scalar_                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_scalar_tensor                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_scalar_scalar                                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_sym_size.py                                                 | test_sym_size                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh                                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh_                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_tanh.py                                                     | test_tanh_backward                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_cuda_runtime_owns_visibility_and_tensor_operations                 |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_hip_runtime_uses_rocm_visibility_names_without_changing_torch_api  |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_maca_runtime_uses_metax_visibility_without_changing_torch_api      |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_musa_runtime_uses_mthreads_visibility_and_torch_api                |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_runtime_rejects_missing_or_empty_device_api                        |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_runtime_metadata_keeps_architecture_separate_from_platform         |      0 |      0 |       1 |       0 |        1 |
| tests/flagtune/test_device_runtime.py                                  | test_runtime_resolves_benchmark_only_through_triton_boundary            |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_fp64_preserved                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_fp64_preserved                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_embedding.py                                                | test_embedding_dense_backward                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                             | test_grouped_topk                                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                             | test_grouped_topk_large_scale                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                             | test_grouped_topk_scaling_factor                                        |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                             | test_grouped_topk_single_token                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_grouped_topk.py                                             | test_grouped_topk_sigmoid                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_indexer_k_quant_and_cache.py                                | test_indexer_k_quant_and_cache_matches_reference                        |      0 |      0 |       1 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_zero_dim_promotion                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_complex                                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_float64_preserves_small_singular_value |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_float64_tolerance_precision            |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_official_dtype_contract                |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_tolerance_precision                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_fp64_critical_spectrum                 |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_ignores_strict_upper   |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_deflated               |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_extreme_scales         |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_critical_spectrum      |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_batched                |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_matrix_rank.py                                       | test_accuracy_linalg_matrix_rank_large_hermitian_repeatable             |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_residual_f64                                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_linalg_solve_triangular.py                                  | test_no_tle_fallback                                                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_dtype_and_endpoints                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_cpu_allocation_options                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_default_and_meta_device                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_invalid_zero_endpoints                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_existing_step_rejection                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_no_fill_launch_and_fresh_output           |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_meta_endpoints                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_meta_endpoint_rank_validation             |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_meta_allocation_options                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_linspace.py                                                 | TestGenericLinspaceEmpty.test_meta_sparse_layout_rejected               |      0 |      0 |       1 |       0 |        1 |
| tests/test_median_out.py                                               | test_median_out_rejects_cpu_out                                         |      0 |      0 |       1 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_fp8_basic                                                 |      0 |      0 |       1 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_fp8_custom_padding                                        |      0 |      0 |       1 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_fp8_default_inf_padding                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_pack_seq.py                                                 | test_pack_seq_fp8_block_sizes                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm                                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm_out                                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm_empty                                                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm_schema                                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm_fast_route                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_scaled_mm.py                                                | test_scaled_mm_descriptor_dtype_switch                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_forward_contract                                        |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_grid_balance_shapes                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_empty                                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_special_values                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_noncontiguous_uses_common_fallback                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_public_path_record                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_inplace_keeps_alias                                     |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_direct_backward_stays_common                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_registrar_scope                                         |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_supported_contiguous_uses_single_route                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_silu.py                                                     | test_metax_silu_large_noncontiguous_uses_generic_fallback               |      0 |      0 |       1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_complex                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_float_tensor                                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_int_tensor                                             |      0 |      0 |       1 |       0 |        1 |
| tests/test_sub.py                                                      | test_sub_complex_int_scalar                                             |      0 |      0 |       1 |       0 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                      | test_chunk_gated_delta_rule_fwd_accuracy                                |      0 |      0 |       0 |       1 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                      | test_chunk_gated_delta_rule_fwd_no_initial_state                        |      0 |      0 |       0 |       1 |        1 |
| tests/test_FLA/test_chunk_gated_delta_rule_fwd.py                      | test_chunk_gated_delta_rule_fwd_with_cu_seqlens                         |      0 |      0 |       0 |       1 |        1 |
| tests/test_affine_grid_generator_backward.py                           | test_affine_grid_generator_backward_3d                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_rnn_relu_cell.py                                            | test_accuracy_rnn_relu_cell_complex_rejected                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                                            | test_accuracy_rnn_tanh_cell_complex_rejected                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits_backward.py                | test_binary_cross_entropy_with_logits_backward                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits_backward.py                | test_backward_real_dispatch                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits_backward.py                | test_backward_uses_gems_compute                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_binary_cross_entropy_with_logits_backward.py                | test_bce_backward_low_precision_boundaries                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_noncontiguous                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_round_trip                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_row_swap_pivoting                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_batch_independence                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_binds_single_argument                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_singular_does_not_hang_or_crash                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_inv.py                                               | test_linalg_inv_autograd_input_is_detached                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_noncontiguous                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_identity                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_degenerate_eigenvalues                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_ascending_order                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_trace_is_preserved                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_reads_requested_triangle                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_default_uplo_is_lower                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_batch_independence                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_autograd_input_is_detached                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_out                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_out_resize                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eigvalsh.py                                          | test_linalg_eigvalsh_out_rejects_mismatched_dtype                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_cosh.py                                                     | test_cosh_noncontiguous                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_default_params                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_inplace_semantics                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_value_range                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_batched_key                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_stream_continuity                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test__philox_uniform_.py                                         | test_philox_uniform_large_offset                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_convolution_overrideable.py                                 | test_convolution_overrideable_transposed                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_cov.py                                                      | test_cov                                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_conv_depthwise3d.py                                         | test_conv_depthwise3d                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_dequantize.py                                               | test_dequantize                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_geglu.py                                                    | test_geglu                                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_greater_equal.py                                            | test_greater_equal__byte_scalar                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py                    | test_nested_from_padded_and_nested_example_3d                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py                    | test_nested_from_padded_and_nested_example_2d                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py                    | test_nested_from_padded_and_nested_example_ragged_idx                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_from_padded_and_nested_example.py                    | test_nested_from_padded_and_nested_example_zero_length                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_invalid                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_dense_and_scalar                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_invalid_high_rank                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_values_nnz                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_strided_offset                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_error_across_tiles                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_validate_sparse_coo_tensor_args.py                          | test_validate_sparse_coo_tensor_args_dispatch                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_integral_exponent_boundaries                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_real_operands_complex_out                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_ldexp.py                                                    | test_ldexp_lazy_conjugation                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                                         | test_linalg_ldl_solve                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_ldl_solve.py                                         | test_linalg_ldl_solve_batched                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_logsumexp.py                                                | test_logsumexp_backward_high_rank_and_duplicate_dims                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_backward                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_tensor_value_gradient_mixed_dtype                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_fill.py                                                     | test_fill_tensor_double_backward                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_float_zero_points               |      0 |      1 |       0 |       0 |        1 |
| tests/test_empty_per_channel_affine_quantized.py                       | test_empty_per_channel_affine_quantized_memory_format                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_median_out.py                                               | test_median_out                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_median_out.py                                               | test_median_out_even_count_picks_lower_middle                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_median_out.py                                               | test_median_out_rejects_bad_out_dtype                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                                     | test_slow_conv2d_backward                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_slow_conv2d_backward.py                                     | test_slow_conv2d_backward_dispatches_to_gems                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_nextafter_.py                                               | test_nextafter_zero_boundary                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csr_sum.py                                           | test_sparse_csr_sum_out                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_randint.py                                                  | test_randint                                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_slice_backward.py                                           | test_slice_backward_oob_start                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_semi_structured_addmm.py                             | test_sparse_semi_structured_addmm                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_semi_structured_addmm.py                             | test_sparse_semi_structured_addmm_with_alpha_beta                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_erfcx.py                                            | test_special_erfcx                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_squeeze_copy.py                                             | test_squeeze_copy                                                       |      0 |      1 |       0 |       0 |        1 |
| TOTAL                                                                  |                                                                         |    353 |     78 |      67 |       3 |      501 |
|                             filepath                             |                                function                                | passed | failed | skipped | SUBTOTAL |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/test_thnn_fused_lstm_cell.py                               | test_thnn_fused_lstm_cell                                              |      1 |      0 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell.py                               | test_thnn_fused_lstm_cell_with_bias                                    |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                          | test_transpose_same_dim                                                |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                          | test_transpose_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_transpose.py                                          | test_transpose_invalid_dims                                            |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                                | test_cat                                                               |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                                | test_cat_empty_tensor                                                  |      1 |      0 |       0 |        1 |
| tests/test_cat.py                                                | test_cat_out_matches_reference                                         |      1 |      0 |       0 |        1 |
| tests/test_unfold_backward.py                                    | test_unfold_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                             | test_accuracy_addbmm                                                   |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                             | test_addbmm_backward                                                   |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                             | test_addbmm_backward_registered_dispatch                               |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                             | test_addbmm_backward_single_input                                      |      1 |      0 |       0 |        1 |
| tests/test_addbmm.py                                             | test_addbmm_backward_noncontiguous                                     |      1 |      0 |       0 |        1 |
| tests/test_upsample_bicubic2d.py                                 | test_upsample_bicubic2d                                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                     | test_upsample_lanczos2d_aa_backward                                    |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                     | test_upsample_lanczos2d_aa_backward_dtypes                             |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                     | test_upsample_lanczos2d_aa_backward_precomputed_path                   |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                     | test_upsample_lanczos2d_aa_backward_out                                |      1 |      0 |       0 |        1 |
| tests/test_upsample_lanczos2d_aa_backward.py                     | test_upsample_lanczos2d_aa_backward_noncontiguous_grad                 |      1 |      0 |       0 |        1 |
| tests/test_upsample_nearest_exact1d.py                           | test_accuracy__upsample_nearest_exact1d                                |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py                 | test_value_selecting_reduction_backward                                |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py                 | test_value_selecting_reduction_backward_max_dim                        |      1 |      0 |       0 |        1 |
| tests/test_value_selecting_reduction_backward.py                 | test_value_selecting_reduction_backward_min_dim                        |      1 |      0 |       0 |        1 |
| tests/test_addbmm_.py                                            | test_accuracy_addbmm_                                                  |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_self                                                        |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_self_scalar                                                 |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_self_scalar_other                                           |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_self_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_self_out_cross_device                                       |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_uniform_condition_strided_storage                           |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_uniform_condition_boundaries                                |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_uniform_condition_scalar                                    |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_uniform_condition_overlapping_output_rejected               |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_uniform_condition_graph_reloads_inputs                      |      1 |      0 |       0 |        1 |
| tests/test_where_self_out.py                                     | test_where_invalid_broadcast_raises_runtime_error                      |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                         | test_zeros_like                                                        |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                         | test_zeros_like_launch_core_dim_limit                                  |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                         | test_zeros_like_factory_capped_grid_exact                              |      1 |      0 |       0 |        1 |
| tests/test_zeros_like.py                                         | test_zeros_like_ascend_review_options                                  |      1 |      0 |       0 |        1 |
| tests/test_cdist.py                                              | test_cdist                                                             |      1 |      0 |       0 |        1 |
| tests/test_hardshrink_backward.py                                | test_hardshrink_backward_grad_input                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                           | test_sparse_compressed_tensor_explicit_size                            |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                           | test_sparse_compressed_tensor_infer_size                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                           | test_sparse_compressed_tensor_rejects_bad_layout                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                           | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap       |      1 |      0 |       0 |        1 |
| tests/test_special_i1e.py                                        | test_special_i1e                                                       |      1 |      0 |       0 |        1 |
| tests/test_special_i1e.py                                        | test_special_i1e_out                                                   |      1 |      0 |       0 |        1 |
| tests/test_Concat.py                                             | test_Concat                                                            |      1 |      0 |       0 |        1 |
| tests/test_add_relu_.py                                          | test_add_relu__tensor                                                  |      1 |      0 |       0 |        1 |
| tests/test_affine_grid_generator.py                              | test_affine_grid_generator                                             |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_invalid_args                                           |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_dtype_mismatch                                     |      1 |      0 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_device_mismatch                                    |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh                                                             |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_various_sizes                                               |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_edge_cases                                                  |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_empty_tensor                                                |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_accuracy_asinh_out                                                |      1 |      0 |       0 |        1 |
| tests/test_asinh.py                                              | test_asinh_                                                            |      1 |      0 |       0 |        1 |
| tests/test_avg_pool3d.py                                         | test_avg_pool3d_backward                                               |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli_                                                        |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli_various_p                                               |      1 |      0 |       0 |        1 |
| tests/test_bernoulli.py                                          | test_bernoulli                                                         |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_tensor.py                                 | test_bitwise_and_tensor                                                |      1 |      0 |       0 |        1 |
| tests/test_bitwise_and_tensor.py                                 | test_bitwise_and_tensor_                                               |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell                                                |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_non_contiguous                                 |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_zero_sized                                     |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_autograd_graph                                 |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_keyword_arguments                              |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul                                                      |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_non_contiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_out                                                  |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_nan_inf                                              |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_backward                                             |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_empty_list                                   |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_non_tensor                                   |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_1d_matrix                                    |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_3d_tensor                                    |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_mismatched_dims                              |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_rejects_int_dtype                                    |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                                       | test_chain_matmul_out_rejects_wrong_dtype                              |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                | test_special_erf                                                       |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                | test_erf                                                               |      1 |      0 |       0 |        1 |
| tests/test_erf.py                                                | test_erf_                                                              |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_fast_path                                     |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_fallback_cases                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_int_fallback                                  |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_writes_through_view                           |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_registration_is_live                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_abs_inplace_rejects_empty_list                            |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_fast_path                                   |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_half_to_even                                |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_large_magnitude                             |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_fallback_cases                              |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_writes_through_view                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_registration_is_live                        |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_round_inplace_rejects_empty_list                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_fast_path                                    |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_fallback_cases                               |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_int_fallback                                 |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_writes_through_view                          |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_registration_is_live                         |      1 |      0 |       0 |        1 |
| tests/test_foreach_abs_round_sign.py                             | test_foreach_sign_inplace_rejects_empty_list                           |      1 |      0 |       0 |        1 |
| tests/test_group_norm.py                                         | test_native_group_norm                                                 |      1 |      0 |       0 |        1 |
| tests/test_group_norm.py                                         | test_group_norm                                                        |      1 |      0 |       0 |        1 |
| tests/test_group_norm.py                                         | test_group_norm_backward                                               |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr                                                         |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_out                                                     |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_out_invalid                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_non_contiguous                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_invalid_mode                                            |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_input_not_mutated                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_zero_column                                             |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_rank_deficient                                          |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_empty                                                   |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_transposed_input                                        |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_unsupported_dtype                                       |      1 |      0 |       0 |        1 |
| tests/test_linalg_qr.py                                          | test_linalg_qr_dim_lt_2                                                |      1 |      0 |       0 |        1 |
| tests/test_matmuladd.py                                          | test_matmuladd                                                         |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py                     | test_nested_tensor_from_tensor_list                                    |      1 |      0 |       0 |        1 |
| tests/test_nested_tensor_from_tensor_list.py                     | test_dispatch_reaches_gems_impl                                        |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T                                                           |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_shares_storage                                            |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_low_rank_distinct_object                                  |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_noncontiguous                                             |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_strided_and_expanded                                      |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_autograd                                                  |      1 |      0 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_rejects_non_tensor                                        |      1 |      0 |       0 |        1 |
| tests/test_softmax.py                                            | test_softmax_wide_inner_row                                            |      1 |      0 |       0 |        1 |
| tests/test_softmax.py                                            | test_softmax                                                           |      1 |      0 |       0 |        1 |
| tests/test_softmax.py                                            | test_softmax_out                                                       |      1 |      0 |       0 |        1 |
| tests/test_softmax.py                                            | test_softmax_backward_out                                              |      1 |      0 |       0 |        1 |
| tests/test_softmax.py                                            | test_softmax_backward                                                  |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_all_size_one_dims                                         |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_dim                                                       |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_dims                                                      |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_noncontiguous_layout                                      |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_result_writes_through_to_input                            |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_backward                                                  |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_out_of_range_dim_raises                                   |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_out_of_range_dims_raises                                  |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_duplicate_dims_raises                                     |      1 |      0 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_rejects_non_tensor                                        |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py | test_fake_quantize_per_tensor_affine_cachemask_backward                |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py | test_fake_quantize_per_tensor_affine_cachemask_backward_mask_semantics |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py | test_fake_quantize_per_tensor_affine_cachemask_backward_noncontiguous  |      1 |      0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py | test_fake_quantize_per_tensor_affine_cachemask_backward_empty          |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_explicit_size                                   |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_infer_size                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_infer_size_trailing_empty_rows                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_aliases_values                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_rejects_dtype_mismatch                          |      1 |      0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                  | test_sparse_csc_tensor_roundtrip_to_csr                                |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes                                                          |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_negative_dims                                            |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_non_contiguous                                           |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_scalar_and_empty                                         |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_storage_offset_preserved                                 |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_shares_storage                                           |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_autograd                                                 |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_invalid_axes                                             |      1 |      0 |       0 |        1 |
| tests/test_swapaxes.py                                           | test_swapaxes_dispatch_reaches_gems_impl                               |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler                                                      |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_oh_ne_iw                                             |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_nearest_half_integer                                 |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_dim_one_reflection                                   |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_5d                                                   |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_5d_varied_output                                     |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_non_contiguous                                       |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_out_of_range_grid                                    |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_empty_output                                         |      1 |      0 |       0 |        1 |
| tests/test_grid_sampler.py                                       | test_grid_sampler_error_paths                                          |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py                    | test_validate_sparse_csr_tensor_args                                   |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py                    | test_validate_sparse_csr_tensor_args_invalid_metadata                  |      1 |      0 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py                    | test_validate_sparse_csr_tensor_args_dispatch                          |      1 |      0 |       0 |        1 |
| tests/test_add.py                                                | test_add_scalar_scalar                                                 |      1 |      0 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_rejects_non_int_offset                                   |      1 |      0 |       0 |        1 |
| tests/test_index.py                                              | test_index                                                             |      1 |      0 |       0 |        1 |
| tests/test_index.py                                              | test_index_error_empty_indices                                         |      1 |      0 |       0 |        1 |
| tests/test_index.py                                              | test_index_error_too_many_indices                                      |      1 |      0 |       0 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_radix_final_config_prefers_large_vocab      |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_full_vocab                                  |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_large_vocab_partial_nonzero_range           |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_variable_lengths                            |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_nonzero_starts                              |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_top_k_per_row_prefill_vs_vllm                                     |      0 |      0 |       1 |        1 |
| tests/test_top_k_per_row_prefill.py                              | test_topk_greater_than_row_len                                         |      0 |      0 |       1 |        1 |
| tests/test_transpose.py                                          | test_transpose                                                         |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                                | test_hardshrink_backward                                               |      0 |      0 |       1 |        1 |
| tests/test_hardshrink_backward.py                                | test_hardshrink_backward_special_values                                |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_public_linear_rejects_mismatched_k_before_launch                  |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_public_linear_valid_dimensions_match_reference                    |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_public_linear_matches_fp64_and_keeps_default_configs              |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_public_linear_registered_aten_path                                |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_tuned_linear_capture_and_two_replays                              |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_parameter_metadata_and_values_preserved                           |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_public_selection_exclusion_branches                               |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_selection_metadata_rejections                                     |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_tuned_numerical_boundaries                                        |      0 |      0 |       1 |        1 |
| tests/test_metax_linear.py                                       | test_tuned_nonfinite_inputs                                            |      0 |      0 |       1 |        1 |
| tests/test_apply_repetition_penalties.py                         | test_apply_repetition_penalty                                          |      0 |      0 |       1 |        1 |
| tests/test_cudnn_batch_norm_backward.py                          | test_cudnn_batch_norm_backward                                         |      0 |      0 |       1 |        1 |
| tests/test_cudnn_batch_norm_backward.py                          | test_cudnn_batch_norm_backward_fp64_split                              |      0 |      0 |       1 |        1 |
| tests/test_fused_indexer_q_rope_quant.py                         | test_fused_indexer_q_rope_quant                                        |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                 | test_rms_norm_w8a16_fp8_weight_updates                                 |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                 | test_rms_norm_w8a16_fp8_encodings                                      |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                 | test_rms_norm_w8a16_fp8                                                |      0 |      0 |       1 |        1 |
| tests/test_rms_norm_w8a16_fp8.py                                 | test_rms_norm_w8a16_fp8_mthreads_shapes                                |      0 |      0 |       1 |        1 |
| tests/test_add.py                                                | test_add_complex                                                       |      0 |      0 |       1 |        1 |
| tests/test_index.py                                              | test_index_large_strided_int32_offset                                  |      0 |      0 |       1 |        1 |
| tests/test_addbmm.py                                             | test_addbmm_backward_fused_kernel_count                                |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_basic                                                  |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_default_args                                           |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_negative_start_end                                     |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_step                                                   |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_edge_cases                                             |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_negative_dim                                           |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_int_dtype                                              |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_all_dtypes                                             |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_float8_bitwise                                         |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_complex                                                |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_lazy_conj_neg                                          |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_non_contiguous_input                                   |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out                                                    |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_vs_aten_all_dtypes                                 |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_mismatched_size                                    |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_non_contiguous                                     |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_overlapping                                        |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_internal_overlap                                   |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_out_non_contiguous_input                               |      0 |      1 |       0 |        1 |
| tests/test_slice_copy.py                                         | test_slice_copy_int64_index                                            |      0 |      1 |       0 |        1 |
| tests/test_avg_pool3d.py                                         | test_avg_pool3d                                                        |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_invalid_input                                  |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                                          | test_accuracy_lstm_cell_complex_rejected                               |      0 |      1 |       0 |        1 |
| tests/test_inplace_fused_experts.py                              | test_inplace_fused_experts_accuracy                                    |      0 |      1 |       0 |        1 |
| tests/test_inplace_fused_experts.py                              | test_inplace_fused_experts_matches_outplace                            |      0 |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                         | test_linalg_eig                                                        |      0 |      1 |       0 |        1 |
| tests/test_linalg_eig.py                                         | test_linalg_eig_batched                                                |      0 |      1 |       0 |        1 |
| tests/test_numpy_T.py                                            | test_numpy_T_lazy_conj_bit_forwarded                                   |      0 |      1 |       0 |        1 |
| tests/test_squeeze.py                                            | test_squeeze_keeps_conjugate_bit                                       |      0 |      1 |       0 |        1 |
| tests/test_validate_sparse_csr_tensor_args.py                    | test_validate_sparse_csr_tensor_args_invalid_indices                   |      0 |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py                      | test__sparse_semi_structured_apply                                     |      0 |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py                      | test_sparse_semi_structured_apply_dispatch                             |      0 |      1 |       0 |        1 |
| tests/test_add.py                                                | test_add                                                               |      0 |      1 |       0 |        1 |
| tests/test_add.py                                                | test_add_                                                              |      0 |      1 |       0 |        1 |
| tests/test_add.py                                                | test_add_tensor_scalar                                                 |      0 |      1 |       0 |        1 |
| tests/test_add.py                                                | test_add_tensor_scalar_                                                |      0 |      1 |       0 |        1 |
| tests/test_add.py                                                | test_add_scalar_tensor                                                 |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                              | test_atan2                                                             |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                              | test_atan2_out                                                         |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                              | test_atan2_out_integer_rejects                                         |      0 |      1 |       0 |        1 |
| tests/test_bitwise_right_shift.py                                | test_bitwise_right_shift                                               |      0 |      1 |       0 |        1 |
| tests/test_bitwise_right_shift.py                                | test_bitwise_right_shift_                                              |      0 |      1 |       0 |        1 |
| tests/test_cast_Long.py                                          | test_cast_Long                                                         |      0 |      1 |       0 |        1 |
| tests/test_cast_Long.py                                          | test_cast_Long_strided                                                 |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat                                                          |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_int_bool                                                 |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_element_order                                            |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_non_contiguous                                           |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_transposed                                               |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_nan_inf                                                  |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_default_offset                                           |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_backward                                                 |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_backward_sum                                             |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_backward_non_contiguous_grad                             |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_backward_complex_sum                                     |      0 |      1 |       0 |        1 |
| tests/test_diagflat.py                                           | test_diagflat_large_output_is_deterministic                            |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_nonleading_adjacent_tensor_indices                          |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_with_none_basic_indexing                                    |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_with_none_and_tensor                                        |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_boolean_mask                                                |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_empty_tensor                                                |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_1d_special_case                                             |      0 |      1 |       0 |        1 |
| tests/test_index.py                                              | test_index_int32_offset_overflow                                       |      0 |      1 |       0 |        1 |
| TOTAL                                                            |                                                                        |    182 |     64 |      30 |      276 |
