|                       filepath                        |                              function                               | passed | failed | skipped | SUBTOTAL |
| ----------------------------------------------------- | ------------------------------------------------------------------- | -----: | -----: | ------: | -------: |
| tests/core/test_benchmark_case_contract.py            | test_case_list_contract_excludes_private_materialization_state      |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py            | test_shapes_and_parameters_roundtrip                                |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py            | test_metadata_does_not_silently_stringify_non_json_objects          |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_case_contract.py            | test_plan_defaults_are_not_shared                                   |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_runs_candidate_once_per_selected_case                |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_without_override_preserves_gems_dispatch             |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_failure_cannot_be_recorded_as_passed                 |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_rejects_legacy_and_empty_case_plans                  |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_report_replaces_stale_data                           |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_unknown_or_unexecuted_case_and_all_skip_cannot_pass            |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_preflight_rejects_conflicting_cli_modes                        |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_profile_still_uses_shared_candidate_resolution                 |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_pytest_profile_plugin_dispatch                                 |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_benchmark_times_reference_and_live_override_separately         |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_profile_invokes_kernel_returns_false_for_cpu_only              |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_profile_invokes_kernel_returns_true_for_device_event           |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_base_without_device_kernel_is_rejected                         |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_base_with_device_kernel_is_accepted                            |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_verify_base_invokes_kernel_caches_result                       |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_verify_base_invokes_kernel_caches_failure                      |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_benchmark_cannot_claim_an_unused_override                      |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_case_invokes_kernel_passes                        |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_case_no_kernel_records_false                      |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_all_cases_run_by_default                          |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_failure_records_error                             |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_legacy_fallback_invokes_kernel                    |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_legacy_fallback_no_kernel                         |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_rejects_empty_case_plan                           |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_session_passes_if_all_operators_have_kernel       |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_session_fails_if_any_operator_has_no_kernel       |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_session_fails_if_no_kernel                        |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_session_fails_if_no_records                       |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_does_not_downgrade_interrupt                      |      1 |      0 |       0 |        1 |
| tests/core/test_benchmark_preflight.py                | test_kernel_check_rejects_conflicting_cli_modes                     |      1 |      0 |       0 |        1 |
| tests/test__linalg_slogdet.py                         | test__linalg_slogdet                                                |      1 |      0 |       0 |        1 |
| tests/test__weight_norm.py                            | test_underscore_weight_norm                                         |      1 |      0 |       0 |        1 |
| tests/test_adaptive_max_pool2d_backward.py            | test_adaptive_max_pool2d_backward                                   |      1 |      0 |       0 |        1 |
| tests/test_addcmul.py                                 | test_addcmul                                                        |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                           | test_fmax_backward                                                  |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                           | test_fmax_backward_one_input                                        |      1 |      0 |       0 |        1 |
| tests/test_fmax_backward.py                           | test_fmax_backward_grad_fn                                          |      1 |      0 |       0 |        1 |
| tests/test_any.py                                     | test_any                                                            |      1 |      0 |       0 |        1 |
| tests/test_any.py                                     | test_any_dims                                                       |      1 |      0 |       0 |        1 |
| tests/test_any.py                                     | test_any_dim                                                        |      1 |      0 |       0 |        1 |
| tests/test_argmin.py                                  | test_argmin                                                         |      1 |      0 |       0 |        1 |
| tests/test_avg_pool1d.py                              | test_avg_pool1d                                                     |      1 |      0 |       0 |        1 |
| tests/test_avg_pool1d.py                              | test_avg_pool1d_default_stride                                      |      1 |      0 |       0 |        1 |
| tests/test_batch_norm_no_update.py                    | test_batch_norm_no_update                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve                                                 |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_larger_shapes                                   |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_rhs_boundaries                                  |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_upper                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_transpose_contiguous_factor                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_blocked_single_rhs                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_ascend_batched_small_lower_single_rhs           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_ascend_blocked_single_rhs_conditioned           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_fp32_blocked_upper                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_fp32_blocked_lower                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_batch                                           |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_broadcast_batch                                 |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_noncontiguous_inputs                            |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_scaled_inputs                                   |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_conditioned_matrix                              |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_accuracy                                        |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_direct                                          |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out                                             |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out_resizes_for_broadcast_result                |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out_noncontiguous_and_alias                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out_rejects_incompatible_dtype                  |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out_rejects_different_device                    |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_empty_input                                     |      1 |      0 |       0 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_invalid_inputs                                  |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell                                         |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_non_contiguous                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_zero_sized                              |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_invalid_input                           |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_autograd_graph                          |      1 |      0 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_keyword_arguments                       |      1 |      0 |       0 |        1 |
| tests/test_cdist_forward.py                           | test_cdist_forward                                                  |      1 |      0 |       0 |        1 |
| tests/test_arctanh.py                                 | test_arctanh                                                        |      1 |      0 |       0 |        1 |
| tests/test_arctanh.py                                 | test_arctanh_                                                       |      1 |      0 |       0 |        1 |
| tests/test_arctanh.py                                 | test_arctanh_out                                                    |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                            | test_column_stack                                                   |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                            | test_column_stack_out                                               |      1 |      0 |       0 |        1 |
| tests/test_column_stack.py                            | test_exception_column_stack                                         |      1 |      0 |       0 |        1 |
| tests/test_contiguous.py                              | test_accuracy_contiguous                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_2d                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_2d_dilation                                   |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_1d                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_3d                                            |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_invalid_padding                               |      1 |      0 |       0 |        1 |
| tests/test_convolution_mode.py                        | test_convolution_mode_same_strided_raises                           |      1 |      0 |       0 |        1 |
| tests/test_count_nonzero.py                           | test_count_nonzero                                                  |      1 |      0 |       0 |        1 |
| tests/test_cummin_helper.py                           | test_cummin_helper                                                  |      1 |      0 |       0 |        1 |
| tests/test_cummin_helper.py                           | test_cummin_helper_with_nan                                         |      1 |      0 |       0 |        1 |
| tests/test_cummin_helper.py                           | test_cummin_helper_dim                                              |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_zero_size_dims                                      |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_duplicate_dim_raises                                |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_out_of_range_dim_raises                             |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dense_input_raises                                  |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_csr_input_raises                                    |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_out_dtype_mismatch_raises                       |      1 |      0 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_dispatch_reaches_gems_impl                                     |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_same_tensors                           |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_different_tensors_same_shape           |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_view_of_same_storage                   |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_as_strided_views                       |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_different_dtypes                       |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_different_shapes_different_storages    |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_empty_tensors                          |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_non_contiguous                         |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_chunk_views                            |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_dtype_sweep                            |      1 |      0 |       0 |        1 |
| tests/test_has_same_storage_numel.py                  | test__has_same_storage_numel_dispatch_stability                     |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_matmul                                                  |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_bmm                                                     |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_dot                                                     |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_outer                                                   |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_trace                                                   |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_diagonal                                                |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_transpose                                               |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_sum_all                                                 |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_sum_dim                                                 |      1 |      0 |       0 |        1 |
| tests/test_einsum.py                                  | test_einsum_ellipsis                                                |      1 |      0 |       0 |        1 |
| tests/test_dreglu.py                                  | test_dreglu                                                         |      1 |      0 |       0 |        1 |
| tests/test_diag_embed.py                              | test_accuracy_diag_embed                                            |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_rng_bridge_build_failure                         |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_generator_word_increment                         |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_generator_offset_overflow_leaves_state_unchanged |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_empty_default_rng                                |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_rejects_nonfloating_dtype                        |      1 |      0 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_rejects_wrong_generator_device                   |      1 |      0 |       0 |        1 |
| tests/test_embedding_spmdm.py                         | test_embedding_spmdm_validation                                     |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window                                                  |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_periodic                                         |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_only_strided_layout                              |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_negative_length                                  |      1 |      0 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_requires_float_dtype                             |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_rejects_internally_overlapping_self_                     |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_rejects_integer_self_                                    |      1 |      0 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_rejects_expanding_self_                                  |      1 |      0 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_tile_plan_rejects_inexpressible                     |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__empty_reinterpret                               |      1 |      0 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__rejects_non_coo                                 |      1 |      0 |       0 |        1 |
| tests/test_angle.py                                   | test_angle                                                          |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_matches_native_view_table                              |      1 |      0 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_diagnostics_parity                                     |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_empty                                         |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_empty                                     |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_invalid_shapes                                |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_invalid_shapes                            |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_rejected_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_rejected_dtype                            |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_rejected_float64                              |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_rejects_expanded_output                   |      1 |      0 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_rejects_autograd                          |      1 |      0 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_rejects_complex                                          |      1 |      0 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_invalid_level_for_dual                   |      1 |      0 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_return_structure                         |      1 |      0 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_vf_binding                               |      1 |      0 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_registry_wiring                          |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_rejects_out                                          |      1 |      0 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_nan                                                  |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_zero_sized                                  |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_autograd_graph                              |      1 |      0 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_keyword_arguments                           |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_rejects_mismatched_metadata           |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_rejects_mismatched_device             |      1 |      0 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_validates_normalized_shape            |      1 |      0 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_mismatched_numel                                     |      1 |      0 |       0 |        1 |
| tests/test_roll.py                                    | test_roll_single_dim                                                |      1 |      0 |       0 |        1 |
| tests/test_roll.py                                    | test_roll_multi_dims                                                |      1 |      0 |       0 |        1 |
| tests/test_roll.py                                    | test_roll_with_non_dense_input                                      |      1 |      0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_dense_self                      |      1 |      0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_dense_mask                      |      1 |      0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_size_mismatch                   |      1 |      0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_invalid_accumulate              |      1 |      0 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_dense_out                       |      1 |      0 |       0 |        1 |
| tests/test_special_ndtri.py                           | test_special_ndtri_unsupported_dtype                                |      1 |      0 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_empty_output_warning                                  |      1 |      0 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_dim_errors                                            |      1 |      0 |       0 |        1 |
| tests/test_sym_size.py                                | test_sym_size                                                       |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_empty_list                                |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_non_tensor                                |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_1d_matrix                                 |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_3d_tensor                                 |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_mismatched_dims                           |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_rejects_int_dtype                                 |      1 |      0 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_out_rejects_wrong_dtype                           |      1 |      0 |       0 |        1 |
| tests/flagtune/test_auto_failure_identity.py          | test_failure_isolates_platform_dtype_and_recovery                   |      0 |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py          | test_identity_discovery_failure_recovers_without_stale_state        |      0 |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py          | test_direct_tuner_retry_bypasses_best_config_cache                  |      0 |      0 |       1 |        1 |
| tests/flagtune/test_auto_failure_identity.py          | test_dispatch_switches_only_for_current_model                       |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_small_gather_single_rhs                         |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_fp64_blocked                                    |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex                                         |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_noncontiguous                           |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_broadcast                               |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_small_gather_conditioned                |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_blocked                                 |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_blocked_conditioned                     |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_lazy_conjugate_upper                    |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_complex_lazy_conjugate_with_all_gems            |      0 |      0 |       1 |        1 |
| tests/test_cholesky_solve.py                          | test_cholesky_solve_out_safe_dtype_cast                             |      0 |      0 |       1 |        1 |
| tests/test_cudnn_attention_forward.py                 | test_cudnn_attention_forward                                        |      0 |      0 |       1 |        1 |
| tests/test_embedding_sparse_backward.py               | test_embedding_sparse_backward                                      |      0 |      0 |       1 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_reservations_interoperate_with_native_rand       |      0 |      0 |       1 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_large_64bit_indexing                             |      0 |      0 |       1 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_rejects_graph_capture_without_consuming_rng      |      0 |      0 |       1 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_fp64_exponent_precision_                                 |      0 |      0 |       1 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_complex_                                                 |      0 |      0 |       1 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_complex128_exponent_precision_                           |      0 |      0 |       1 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_mixed_precision_extreme_exponent_                        |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_float64_precision                             |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_chunked_previous_rows                         |      0 |      0 |       1 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_float64_offdiagonal_precision                 |      0 |      0 |       1 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values_out_wrong_device                             |      0 |      0 |       1 |        1 |
| tests/test_addcmul.py                                 | test_addcmul_out_broadcast                                          |      0 |      1 |       0 |        1 |
| tests/test_binary_cross_entropy.py                    | test_binary_cross_entropy                                           |      0 |      1 |       0 |        1 |
| tests/test_binary_cross_entropy.py                    | test_binary_cross_entropy_weight                                    |      0 |      1 |       0 |        1 |
| tests/test_binary_cross_entropy.py                    | test_binary_cross_entropy_out                                       |      0 |      1 |       0 |        1 |
| tests/test_binary_cross_entropy.py                    | test_binary_cross_entropy_boundary                                  |      0 |      1 |       0 |        1 |
| tests/test_rnn_tanh_cell.py                           | test_accuracy_rnn_tanh_cell_complex_rejected                        |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_default                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_default_empty                                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_default_uncoalesced                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dtype                                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim                                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_all                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_dtype                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_out                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_negative                                        |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dim_keeps_structural_zeros                          |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_hybrid_dense_dims                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_uncoalesced_input                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_integer_promotion                                   |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_dtype_accumulator_precision                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_complex                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_dispatch_covers_every_overload                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_sum.py                              | test_sparse_sum_uncoalesced_matches_aten                            |      0 |      1 |       0 |        1 |
| tests/test_diag_embed.py                              | test_diag_embed_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_diag_embed.py                              | test_diag_embed_backward_default_dims                               |      0 |      1 |       0 |        1 |
| tests/test_diag_embed.py                              | test_diag_embed_backward_non_contiguous                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul_structure_and_zeros                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul_out                                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul_rejects_shape_mismatch                    |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul_rejects_dtype_mismatch                    |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_sparse_sparse_matmul_rejects_dense_operand                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_sparse_matmul.py                    | test_dispatch_reaches_gems_impl                                     |      0 |      1 |       0 |        1 |
| tests/test_fix_.py                                    | test_fix_                                                           |      0 |      1 |       0 |        1 |
| tests/test_float_power_.py                            | test_float_power_                                                   |      0 |      1 |       0 |        1 |
| tests/test_float_power_.py                            | test_float_power_scalar                                             |      0 |      1 |       0 |        1 |
| tests/test_frac.py                                    | test_frac                                                           |      0 |      1 |       0 |        1 |
| tests/test_functional_sym_constrain_range_for_size.py | test_functional_sym_constrain_range_for_size                        |      0 |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py           | test__sparse_semi_structured_apply                                  |      0 |      1 |       0 |        1 |
| tests/test__sparse_semi_structured_apply.py           | test_sparse_semi_structured_apply_dispatch                          |      0 |      1 |       0 |        1 |
| tests/test_ge.py                                      | test_ge                                                             |      0 |      1 |       0 |        1 |
| tests/test_ge.py                                      | test_ge_scalar                                                      |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_python_reservation_matches_native                |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_python_reservation_contract                      |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout                                                  |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_layout                                           |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_generator_replay_and_independence                |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_generator_counter_carry                          |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_default_generator_replay                         |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_concurrent_generator_reservations                |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_aten_autograd_uses_gems                          |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_checkpoint_replays_default_rng                   |      0 |      1 |       0 |        1 |
| tests/test_fused_dropout.py                           | test_fused_dropout_special_values                                   |      0 |      1 |       0 |        1 |
| tests/test_gt.py                                      | test_gt                                                             |      0 |      1 |       0 |        1 |
| tests/test_gt.py                                      | test_gt_scalar                                                      |      0 |      1 |       0 |        1 |
| tests/test_hardsigmoid_backward.py                    | test_hardsigmoid_backward                                           |      0 |      1 |       0 |        1 |
| tests/test_gelu_and_mul.py                            | test_gelu_and_mul                                                   |      0 |      1 |       0 |        1 |
| tests/test_bucketize.py                               | test_bucketize                                                      |      0 |      1 |       0 |        1 |
| tests/test_bucketize.py                               | test_bucketize_right                                                |      0 |      1 |       0 |        1 |
| tests/test_bucketize.py                               | test_bucketize_int32                                                |      0 |      1 |       0 |        1 |
| tests/test_bucketize.py                               | test_bucketize_boundary_cases                                       |      0 |      1 |       0 |        1 |
| tests/test_bucketize.py                               | test_bucketize_noncontiguous_input                                  |      0 |      1 |       0 |        1 |
| tests/test_ilshift.py                                 | test_ilshift                                                        |      0 |      1 |       0 |        1 |
| tests/test_ilshift.py                                 | test_ilshift_int8                                                   |      0 |      1 |       0 |        1 |
| tests/test_index_select.py                            | test_index_select                                                   |      0 |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                         | test_embedding_spmdm_matches_embedding_bag                          |      0 |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                         | test_embedding_spmdm_index_dtypes                                   |      0 |      1 |       0 |        1 |
| tests/test_embedding_spmdm.py                         | test_embedding_spmdm_noncontiguous_inputs                           |      0 |      1 |       0 |        1 |
| tests/test_isin.py                                    | test_accuracy_isin                                                  |      0 |      1 |       0 |        1 |
| tests/test_isin.py                                    | test_accuracy_isin_scalar_tensor                                    |      0 |      1 |       0 |        1 |
| tests/test_isin.py                                    | test_accuracy_isin_tensor_scalar                                    |      0 |      1 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_beta                                             |      0 |      1 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_degenerate_lengths                               |      0 |      1 |       0 |        1 |
| tests/test_kaiser_window.py                           | test_kaiser_window_default_device                                   |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_                                                         |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_mixed_dtype_                                             |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_broadcast_noncontiguous_and_empty_                       |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_special_values_                                          |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_extreme_integral_exponents_                              |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_overlapping_operands_                                    |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_rejects_complex_to_real_                                 |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_int64_exponent_outside_int32_range_                      |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_complex_integral_extremes_                               |      0 |      1 |       0 |        1 |
| tests/test_ldexp_.py                                  | test_ldexp_conjugated_complex_exponent_                             |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as                                                     |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_non_contiguous                                      |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_view_strided                                        |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_materialized_copy                                   |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_special                                             |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_backward                                            |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_other_has_no_gradient                               |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_complex_copy                                        |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_dispatch                                            |      0 |      1 |       0 |        1 |
| tests/test_reshape_as.py                              | test_reshape_as_tile_kernel                                         |      0 |      1 |       0 |        1 |
| tests/test_batch_norm_elemt.py                        | test_batch_norm_elemt                                               |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__identity                                        |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__sparse_dims_grow                                |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__hybrid_sparse_dims_grow                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__dense_dims_grow                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__dense_dims_grow_complex                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_resize_.py                          | test_sparse_resize__non_empty_rejects                               |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_real_dtypes                                            |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_complex_dtypes                                         |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_0d                                                     |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_1d_matches_native_error                                |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_non_contiguous                                         |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_mutation_through_view                                  |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_conjugate_bit_input                                    |      0 |      1 |       0 |        1 |
| tests/test_adjoint.py                                 | test_adjoint_dispatch_stability                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum                                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_keeps_layout_not_values                         |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_explicit_zeros                                  |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_empty_input                                     |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_dtype                                           |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_unsorted_columns                                |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_out                                             |      0 |      1 |       0 |        1 |
| tests/test_sparse_csr_sum.py                          | test_sparse_csr_sum_rejects_bad_request                             |      0 |      1 |       0 |        1 |
| tests/test_dimV.py                                    | test_accuracy_dimV_coo                                              |      0 |      1 |       0 |        1 |
| tests/test_dimV.py                                    | test_accuracy_dimV_matches_indices_formula                          |      0 |      1 |       0 |        1 |
| tests/test_dimV.py                                    | test_accuracy_dimV_unsupported_layouts_raise                        |      0 |      1 |       0 |        1 |
| tests/test_dimV.py                                    | test_accuracy_dimV_dispatch_stability                               |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve                                               |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_noncontiguous                                 |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_ignored_entries                               |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_singular_and_near_singular                    |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out                                           |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_nonempty_resize                           |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_aliases_inputs                            |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_mismatched_inputs                             |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_invalid_outputs                           |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_autograd_broadcast                            |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_scipy_reference                               |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_wide_rhs                                      |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_large_ignored_entries                         |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_out_large_ignored_entries                     |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_finite_inverse_overflow                       |      0 |      1 |       0 |        1 |
| tests/test_triangular_solve.py                        | test_triangular_solve_panel_wide_rhs                                |      0 |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor_unsafe.py                | test_sparse_bsc_tensor_unsafe                                       |      0 |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor_unsafe.py                | test_sparse_bsc_tensor_unsafe_dtype                                 |      0 |      1 |       0 |        1 |
| tests/test_sparse_bsc_tensor_unsafe.py                | test_sparse_bsc_tensor_unsafe_skips_index_validation                |      0 |      1 |       0 |        1 |
| tests/test_reshape_alias_copy.py                      | test_accuracy_reshape_alias_copy                                    |      0 |      1 |       0 |        1 |
| tests/test_reshape_alias_copy.py                      | test_accuracy_reshape_alias_copy_noncontiguous_input                |      0 |      1 |       0 |        1 |
| tests/test_reshape_alias_copy.py                      | test_accuracy_reshape_alias_copy_storage_offset                     |      0 |      1 |       0 |        1 |
| tests/test_reshape_alias_copy.py                      | test_accuracy_reshape_alias_copy_independent_storage                |      0 |      1 |       0 |        1 |
| tests/test_reshape_alias_copy.py                      | test_accuracy_reshape_alias_copy_zero_dim                           |      0 |      1 |       0 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values                                              |      0 |      1 |       0 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values_keepdim_shapes                               |      0 |      1 |       0 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values_ties                                         |      0 |      1 |       0 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values_strided_nonlast_reduction                    |      0 |      1 |       0 |        1 |
| tests/test_median_dim_values.py                       | test_median_dim_values_out_rejects_bad_index_dtype                  |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_                                                |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_default_params                                  |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_inplace_semantics                               |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_value_range                                     |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_batched_key                                     |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_stream_continuity                               |      0 |      1 |       0 |        1 |
| tests/test__philox_uniform_.py                        | test_philox_uniform_large_offset                                    |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_preserves_nan_payloads                                   |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_out_alias_across_sort_paths                              |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort                                                          |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_noncontiguous_empty_scalar_and_special_values            |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_out                                                      |      0 |      1 |       0 |        1 |
| tests/test_msort.py                                   | test_msort_out_resize_noncontiguous_alias_and_dtype_error           |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_non_dual                                 |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_level_ignored_for_non_dual               |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_non_contiguous                           |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_offset_and_expanded                      |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_transposed                               |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_dual_level0                              |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_dual_level_is_scoped                     |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_dual_strided                             |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_requires_grad                            |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_no_grad_mode                             |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_inference_tensor                         |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_mutation_through_view                    |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_mutation_through_tangent                 |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_dispatch_stability                       |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_dual_dtype_behavior                      |      0 |      1 |       0 |        1 |
| tests/test__unpack_dual.py                            | test_accuracy__unpack_dual_empty_and_size_one                       |      0 |      1 |       0 |        1 |
| tests/test_native_batch_norm_legit_functional.py      | test_native_batch_norm_legit_functional                             |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter                                                      |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_                                                     |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_out                                                  |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_out_none                                             |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_inplace_is_self                                      |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_edge_values                                          |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_infinity                                             |      0 |      1 |       0 |        1 |
| tests/test_nextafter.py                               | test_nextafter_small_values                                         |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_softmax_with_shape.py        | test_nested_tensor_softmax_with_shape                               |      0 |      1 |       0 |        1 |
| tests/test_nested_tensor_softmax_with_shape.py        | test_nested_tensor_softmax_with_shape_full_length                   |      0 |      1 |       0 |        1 |
| tests/test_nll_loss_backward.py                       | test_nll_loss_backward                                              |      0 |      1 |       0 |        1 |
| tests/test_mT.py                                      | test_mT                                                             |      0 |      1 |       0 |        1 |
| tests/test_mT.py                                      | test_mT_noncontiguous                                               |      0 |      1 |       0 |        1 |
| tests/test_mT.py                                      | test_mT_shares_storage                                              |      0 |      1 |       0 |        1 |
| tests/test_mT.py                                      | test_mT_autograd                                                    |      0 |      1 |       0 |        1 |
| tests/test_mT.py                                      | test_mT_rank1_raises                                                |      0 |      1 |       0 |        1 |
| tests/test_one_hot.py                                 | test_one_hot                                                        |      0 |      1 |       0 |        1 |
| tests/test_one_hot.py                                 | test_one_hot_inference_mode                                         |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell                                             |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_non_contiguous                              |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_invalid_input                               |      0 |      1 |       0 |        1 |
| tests/test_lstm_cell.py                               | test_accuracy_lstm_cell_complex_rejected                            |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist                                                          |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist_p1                                                       |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist_pinf                                                     |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist_p_general                                                |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist_p0                                                       |      0 |      1 |       0 |        1 |
| tests/test_pdist.py                                   | test_pdist_p_large                                                  |      0 |      1 |       0 |        1 |
| tests/test_pixel_unshuffle.py                         | test_pixel_unshuffle                                                |      0 |      1 |       0 |        1 |
| tests/test_pixel_unshuffle.py                         | test_pixel_unshuffle_out                                            |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_forward                               |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_no_grad_with_grad_inputs              |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_optional_affine                       |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_large_normalized_shape                |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_noncontiguous_fallback                |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_backward                              |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_backward_optional_affine              |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_backward_large_normalized_shape       |      0 |      1 |       0 |        1 |
| tests/test_post_layer_norm_residual.py                | test_post_layer_norm_residual_backward_residual_only                |      0 |      1 |       0 |        1 |
| tests/test_negative.py                                | test_negative                                                       |      0 |      1 |       0 |        1 |
| tests/test_rand_like.py                               | test_rand_like                                                      |      0 |      1 |       0 |        1 |
| tests/test_reciprocal.py                              | test_reciprocal                                                     |      0 |      1 |       0 |        1 |
| tests/test_reciprocal.py                              | test_reciprocal_                                                    |      0 |      1 |       0 |        1 |
| tests/test_relu.py                                    | test_relu                                                           |      0 |      1 |       0 |        1 |
| tests/test_relu.py                                    | test_relu_                                                          |      0 |      1 |       0 |        1 |
| tests/test_replication_pad1d.py                       | test_replication_pad1d                                              |      0 |      1 |       0 |        1 |
| tests/test_replication_pad1d.py                       | test_replication_pad1d_out                                          |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as                                                      |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_                                                     |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_non_broadcastable                                    |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as__non_broadcastable                                   |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_empty                                                |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_non_contiguous                                       |      0 |      1 |       0 |        1 |
| tests/test_resize_as.py                               | test_resize_as_non_contiguous_template                              |      0 |      1 |       0 |        1 |
| tests/test_roll.py                                    | test_roll_flatten                                                   |      0 |      1 |       0 |        1 |
| tests/test_rrelu_with_noise_backward.py               | test_rrelu_with_noise_backward                                      |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet                                               |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_uniform_alpha                                 |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_skewed_alpha                                  |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_1d                                            |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_k1                                            |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_large_k                                       |      0 |      1 |       0 |        1 |
| tests/test_sample_dirichlet.py                        | test_sample_dirichlet_tiny_alpha                                    |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                       | test_scaled_grouped_mm                                              |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                       | test_scaled_grouped_mm_fp8_core                                     |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                       | test_scaled_grouped_mm_fp8_encodings                                |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                       | test_scaled_grouped_mm_scale_before_output_cast                     |      0 |      1 |       0 |        1 |
| tests/test_scaled_grouped_mm.py                       | test_scaled_grouped_mm_reduction_addressing                         |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_lengths                                         |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_offsets                                         |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_backward_lengths                                |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_lengths_out                                     |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_offsets_out                                     |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_backward_lengths_out                            |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_backward_offsets_out                            |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_backward_offsets                                |      0 |      1 |       0 |        1 |
| tests/test_segment_reduce.py                          | test_segment_reduce_indices_unsupported                             |      0 |      1 |       0 |        1 |
| tests/test_tanh.py                                    | test_tanh                                                           |      0 |      1 |       0 |        1 |
| tests/test_tanh.py                                    | test_tanh_                                                          |      0 |      1 |       0 |        1 |
| tests/test_tanh.py                                    | test_tanh_backward                                                  |      0 |      1 |       0 |        1 |
| tests/test_skip_layer_norm.py                         | test_skip_layernorm                                                 |      0 |      1 |       0 |        1 |
| tests/test_sobol_engine_ff_.py                        | test_sobol_engine_ff_                                               |      0 |      1 |       0 |        1 |
| tests/test_sort.py                                    | test_sort                                                           |      0 |      1 |       0 |        1 |
| tests/test_sort.py                                    | test_sort_stable                                                    |      0 |      1 |       0 |        1 |
| tests/test_special_bessel_j1.py                       | test_special_bessel_j1                                              |      0 |      1 |       0 |        1 |
| tests/test_special_entr.py                            | test_special_entr                                                   |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_value_range                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_wide_values                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_accumulate_matches                      |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_stored_layout                           |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_hybrid_coo                              |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_strided_storage                         |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_special_values                          |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_out                                     |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_out_stored_layout                       |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_sparse_dim_mismatch             |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_rejects_fp8                             |      0 |      1 |       0 |        1 |
| tests/test__sparse_mask_projection.py                 | test_sparse_mask_projection_dispatch                                |      0 |      1 |       0 |        1 |
| tests/test_special_log_ndtr.py                        | test_special_log_ndtr                                               |      0 |      1 |       0 |        1 |
| tests/test_special_ndtri.py                           | test_special_ndtri                                                  |      0 |      1 |       0 |        1 |
| tests/test_special_ndtri.py                           | test_special_ndtri_edge_values                                      |      0 |      1 |       0 |        1 |
| tests/test_special_ndtri.py                           | test_special_ndtri_non_contiguous                                   |      0 |      1 |       0 |        1 |
| tests/test_special_softmax.py                         | test_special_softmax                                                |      0 |      1 |       0 |        1 |
| tests/test_special_softmax.py                         | test_special_softmax_with_dtype                                     |      0 |      1 |       0 |        1 |
| tests/test_square.py                                  | test_square                                                         |      0 |      1 |       0 |        1 |
| tests/test_square.py                                  | test_square_                                                        |      0 |      1 |       0 |        1 |
| tests/test_square.py                                  | test_square_out                                                     |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean                                                       |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_dim                                                   |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_correction                                            |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_names_dim                                             |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_correction_names                                      |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_correction_out                                        |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_empty_reduction                                       |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_empty_negative_correction_does_not_warn               |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_warns_for_nonpositive_dof                             |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_complex_nonpositive_dof_is_componentwise              |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_numerical_stability                                   |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_numerical_stability_with_first_outlier                |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_large_global_reduction                                |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_low_precision_accumulates_in_float32                  |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_rows_core_and_tail                                    |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_row_dtypes                                            |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_rows_stability                                        |      0 |      1 |       0 |        1 |
| tests/test_std_mean.py                                | test_std_mean_rows_nonfinite                                        |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin                                                         |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin_                                                        |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin_out                                                     |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin_out_integer_rejects                                     |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin_boundaries_and_out_of_domain                            |      0 |      1 |       0 |        1 |
| tests/test_arcsin.py                                  | test_arcsin_empty_and_scalar                                        |      0 |      1 |       0 |        1 |
| tests/test_thnn_fused_lstm_cell_backward_impl.py      | test_thnn_fused_lstm_cell_backward_impl                             |      0 |      1 |       0 |        1 |
| tests/test_arctan2.py                                 | test_arctan2                                                        |      0 |      1 |       0 |        1 |
| tests/test_arctan2.py                                 | test_arctan2_                                                       |      0 |      1 |       0 |        1 |
| tests/test_arctan2.py                                 | test_arctan2__integer_rejects                                       |      0 |      1 |       0 |        1 |
| tests/test_arctan2.py                                 | test_arctan2_special_values                                         |      0 |      1 |       0 |        1 |
| tests/test_arctan2.py                                 | test_arctan2_broadcast                                              |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                   | test_atan2                                                          |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                   | test_atan2_out                                                      |      0 |      1 |       0 |        1 |
| tests/test_atan2.py                                   | test_atan2_out_integer_rejects                                      |      0 |      1 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul                                                   |      0 |      1 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_non_contiguous                                    |      0 |      1 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_out                                               |      0 |      1 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_nan_inf                                           |      0 |      1 |       0 |        1 |
| tests/test_chain_matmul.py                            | test_chain_matmul_backward                                          |      0 |      1 |       0 |        1 |
| TOTAL                                                 |                                                                     |    189 |    325 |      28 |      542 |
|                                   filepath                                   |                                function                                | passed | failed | skipped | xfailed | SUBTOTAL |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -----: | -----: | ------: | ------: | -------: |
| tests/test_uniform.py                                                        | test_uniform_                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_uniform.py                                                        | test_uniform                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_uniform.py                                                        | test_uniform_rejects_inverted_interval                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_uniform.py                                                        | test_uniform_upper_bound_zero                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index                                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_with_none_and_tensor                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_error_bad_index_dtype                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_error_non_broadcastable_indices                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_error_empty_indices                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_error_too_many_indices                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_non_contiguous_input                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_non_contiguous_input_with_none                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_unsafe_index.py                                                   | test_unsafe_index_zero_dim_index_negative                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_grad_input                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_layout                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_impulse                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_resize                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_invalid                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_out_error                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_deterministic                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_nonfinite_cpu_reference               |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_extreme_positive_scale                |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_bfloat16_scalar_rounding              |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_nearest1d.py                                             | test_upsample_nearest1d                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_dtype_mapping                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_values                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_error_paths                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_out                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_special_values                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_tensor_scalar                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_tensor_scalar_out                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_scalar_tensor                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_xlogy.py                                                          | test_xlogy_scalar_tensor_out                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_and_tensor.py                                                     | test_and_tensor                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_cast_Long.py                                                      | test_cast_Long                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cholesky.py                                                | test_linalg_cholesky                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cholesky.py                                                | test_linalg_cholesky_upper                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_cholesky.py                                                | test_linalg_cholesky_batch                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_enable_api.py                                                     | test_enable                                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_enable_api.py                                                     | test_enable_with_exclude                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linear.py                                                         | test_linear_2d_with_bias                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linear.py                                                         | test_linear_2d_without_bias                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linear.py                                                         | test_linear_3d_with_bias                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_linear.py                                                         | test_linear_1d_with_bias                                               |      1 |      0 |       0 |       0 |        1 |
| tests/test_log_normal_.py                                                    | test_log_normal_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_logical_or.py                                                     | test_logical_or                                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_logical_or.py                                                     | test_logical_or_                                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_lt.py                                                             | test_lt                                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_lt.py                                                             | test_lt_scalar                                                         |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_scatter_backward.py                                        | test_accuracy_masked_scatter_backward                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_masked_scatter_backward.py                                        | test_accuracy_masked_scatter_backward_persistent                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward_dispatch                                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_rejects_indices_shape_mismatch                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_mish.py                                                           | test_mish                                                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_mish.py                                                           | test_mish_                                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_mish.py                                                           | test_mish_backward                                                     |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply.py                                                       | test_multiply                                                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_multiply.py                                                       | test_multiply_scalar                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_non_vector                |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_negative_size             |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_invalid_size              |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_invalid_out_int32         |      1 |      0 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_out_dtype_mismatch        |      1 |      0 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_scalar_scalar                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_out_invalid                                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_invalid_mode                                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_empty                                                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_unsupported_dtype                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_dim_lt_2                                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_stored_offsets_with_gap             |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_cpu_nested                          |      1 |      0 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_non_nested_error                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_zeros_like.py                                                     | test_zeros_like_launch_core_dim_limit                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_rejects_non_float_out_buffer    |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_rejects_positional_out          |      1 |      0 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_rejects_non_tensor_input        |      1 |      0 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_invalid                                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_invalid_inputs                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_registration_scope                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_mismatched_devices                              |      1 |      0 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_pin_memory_rejection                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py             | test_fake_quantize_per_tensor_affine_cachemask_backward_empty          |      1 |      0 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_ascend_review_out_validation                                |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_invalid_size                                  |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_negative_dimension_message                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_pin_memory_ignored                            |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_non_strided_layout_rejected                   |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_zero_size                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_cpu                                       |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_cpu_non_contiguous                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_requires_grad_rejected                    |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_meta                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_negative_size                             |      1 |      0 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dispatch_zero_tensor_rejected             |      1 |      0 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_contraction_mismatch                                        |      1 |      0 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_scalar_scalar                                                 |      1 |      0 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_scalar_scalar_float                                           |      1 |      0 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_grid_meta_review                                      |      1 |      0 |       0 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_fp64_opmath                           |      0 |      0 |       1 |       0 |        1 |
| tests/test_upsample_bicubic2d_backward.py                                    | test_upsample_bicubic2d_backward_iluvatar_fp64_rejected                |      0 |      0 |       1 |       0 |        1 |
| tests/test_div.py                                                            | test_div_complex_complex                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_div.py                                                            | test_div_complex_float_tensor                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_int                                                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_div.py                                                            | test_div_complex_int_scalar                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_rejects_unsupported_dtype       |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mm_w8a8_int8                                                      |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mm_w8a8_int8_updates                                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mm_w8a8_int8_empty                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mm_w8a8_int8_reject_float                                         |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_scaled_bias                                                       |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_scaled_graph_bias_updates                                         |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_scaled_validation                                                 |      0 |      0 |       1 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_scaled_unaligned_weight                                           |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_deepseek_v4_attention_dequantize_and_gather_k_cache.py | test_dequantize_and_gather_k_cache_accuracy                            |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_deepseek_v4_attention_dequantize_and_gather_k_cache.py | test_dequantize_and_gather_k_cache_vllm_accuracy                       |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py   | test_q_path_matches_reference                                          |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py   | test_kv_path_matches_reference                                         |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py   | test_kv_path_with_dp_padding                                           |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py   | test_combined_q_and_kv                                                 |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_top_k_per_row_decode.py                                | test_top_k_per_row_decode                                              |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_top_k_per_row_decode.py                                | test_top_k_per_row_decode_partial_seqlen                               |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_top_k_per_row_decode.py                                | test_topk_greater_than_row_len                                         |      0 |      0 |       1 |       0 |        1 |
| tests/deprecated/test_top_k_per_row_decode.py                                | test_logits_diff_in_8LSBits                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_device_index                                    |      0 |      0 |       1 |       0 |        1 |
| tests/test_max_pool1d_with_indices.py                                        | test_max_pool1d_with_indices_special_values                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_max_pool1d_with_indices.py                                        | test_max_pool1d_with_indices_bit_patterns                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_complex                                               |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_float_tensor                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_int_tensor                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_int_scalar                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_hopper_broadcast_survives_monkeypatched_next_power_of_2       |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_basic                                                |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_broadcast                                            |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_noncontiguous_and_selective                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_mixed_dtype                                          |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_complex                                              |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_registered_dispatch                                  |      0 |      0 |       1 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_backward_fused_kernel_count                                   |      0 |      0 |       1 |       0 |        1 |
| tests/test_upsample_nearest_exact2d.py                                       | test_upsample_nearest_exact2d                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_upsample_nearest_exact2d.py                                       | test_upsample_nearest_exact2d_out                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_noncontiguous_batch                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_transposed                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_high_rank_noncontiguous             |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_expanded                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_storage_offset                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_independent_storage                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_view_as_complex_copy.py                                           | test_accuracy_view_as_complex_copy_empty                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_weight_int4pack_mm_with_scales_and_zeros.py                       | test_weight_int4pack_mm_with_scales_and_zeros                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_dgeglu.py                                                         | test_dgeglu                                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_pack_padded_sequence.py                                           | test_pack_padded_sequence                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_pack_padded_sequence.py                                           | test_pack_padded_sequence_ragged_lengths                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Long.py                                                      | test_cast_Long_strided                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_enable_api.py                                                     | test_only_enable                                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_enable_api.py                                                     | test_only_enable_with_yaml                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_enable_api.py                                                     | test_only_enable_default                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_non_contiguous                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward_public_entry                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward_unbatched                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward_non_contiguous_grad_output                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_unpool2d.py                                                   | test_max_unpool2d_backward_conflicting_indices                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_out_int32_grid                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_scalar_input                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_offset_view                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_out                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_out_offset_view                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_convert_indices_from_coo_to_csr.py                                | test_convert_indices_from_coo_to_csr_rejects_unsupported_dtype         |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_tensor                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_tensor_                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_mode_float                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_mode_int                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_mode_float_                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_mode_int_                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_mode_float                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_mode_int                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_mode_float_                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_mode_int_                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_scalar                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_tensor_scalar_                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_                                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_tensor                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_scalar_scalar                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_out_tensor_tensor                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_out_tensor_scalar                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_out_scalar_tensor                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_mode_tensor                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_mode_scalar                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_mode_tensor_                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_div.py                                                            | test_div_mode_scalar_                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_mixed                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_scalar_tensor                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_float                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_float_                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_int                                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_int_                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_scalar_inplace_float                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_floor_divide.py                                                   | test_floor_divide_scalar_inplace_int                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy_int                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy_noncontiguous                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy_out                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy_high_rank                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_fw_primal_copy.py                                                 | test_fw_primal_copy_out_high_rank                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_igamma_.py                                                        | test_igamma_                                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eig.py                                                     | test_linalg_eig                                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_eig.py                                                     | test_linalg_eig_batched                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_out                                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_non_contiguous                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_input_not_mutated                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_zero_column                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_rank_deficient                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_linalg_qr.py                                                      | test_linalg_qr_transposed_input                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_pool3d.py                                                     | test_max_pool3d                                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_pool3d.py                                                     | test_max_pool3d_autograd                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_accuracy_nested_tensor_storage_offsets                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_from_shared_buffer_views            |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_empty_nested                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_scalar_nesting                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_aliases_native_metadata             |      0 |      1 |       0 |       0 |        1 |
| tests/test_nested_tensor_storage_offsets.py                                  | test_nested_tensor_storage_offsets_dispatch_stability                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_half_way_values                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_non_contiguous                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_rejects_non_float32                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_empty                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_dtype_mismatch                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_shape_mismatch                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_memory_format                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_public_entry_keeps_layout                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_zero_point_range                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_contiguous_unchanged                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_matches_native_layout                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_preserves_offset_and_strides              |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_mismatched_qparams                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_rejects_per_channel_out                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_aliasing_input                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_uses_triton_not_native_copy               |      0 |      1 |       0 |       0 |        1 |
| tests/test_quantize_per_tensor.py                                            | test_quantize_per_tensor_out_rejects_internally_overlapping_out        |      0 |      1 |       0 |       0 |        1 |
| tests/test_round.py                                                          | test_round                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_round.py                                                          | test_round_                                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_round.py                                                          | test_round_out                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_round.py                                                          | test_round_decimals                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_round.py                                                          | test_round_half_to_even                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_zeros_like.py                                                     | test_zeros_like                                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_zeros_like.py                                                     | test_zeros_like_factory_capped_grid_exact                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_zeros_like.py                                                     | test_zeros_like_ascend_review_options                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_out                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_strided                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_special_values                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_edge_values                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_special_scaled_modified_bessel_k0.py                              | test_special_scaled_modified_bessel_k0_empty                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_take_along_dim.py                                                 | test_take_along_dim                                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_take_along_dim.py                                                 | test_take_along_dim_dtypes                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_take_along_dim.py                                                 | test_take_along_dim_none                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_take_along_dim.py                                                 | test_take_along_dim_noncontiguous                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_take_along_dim.py                                                 | test_take_along_dim_out                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_foreach_sign_special_values.py                                    | test_sign_special_values                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mm_w8a8_int8_strides                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_scaled_empty_bias                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_tensor_scales_without_bias                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_prequantized                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_long_k_overflow                                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_mixed_scales                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_mm_w8a8_int8.py                                                   | test_graph_updates                                                     |      0 |      1 |       0 |       0 |        1 |
| tests/deprecated/test_moe_align_block_size.py                                | test_accuracy_moe_align_block_size                                     |      0 |      1 |       0 |       0 |        1 |
| tests/deprecated/test_moe_align_block_size.py                                | test_accuracy_moe_align_block_size_triton                              |      0 |      1 |       0 |       0 |        1 |
| tests/deprecated/test_moe_align_block_size.py                                | test_accuracy_moe_align_block_size_fast_paths                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_infer_size                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_conversion_and_strides                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_scalar_values                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_coalesced                                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_dispatch                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_autograd                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_pin_memory                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_large_reduction                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_size_overflow                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_cpu_transfer                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_invariant_check                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_explicit_size                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_explicit_validation                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_coo_tensor.py                                              | test_sparse_coo_tensor_explicit_pin_memory                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_max_pool1d_with_indices.py                                        | test_max_pool1d_with_indices                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py             | test_fake_quantize_per_tensor_affine_cachemask_backward                |      0 |      1 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py             | test_fake_quantize_per_tensor_affine_cachemask_backward_mask_semantics |      0 |      1 |       0 |       0 |        1 |
| tests/test_fake_quantize_per_tensor_affine_cachemask_backward.py             | test_fake_quantize_per_tensor_affine_cachemask_backward_noncontiguous  |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_explicit_size                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_infer_size                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_infer_size_trailing_empty_rows                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_aliases_values                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_rejects_dtype_mismatch                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_csc_tensor.py                                              | test_sparse_csc_tensor_roundtrip_to_csr                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_beta_zero_ignores_bias                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_vector_bias_shapes                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_scalar_bias                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_padded_vector_bias                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_out                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_out_beta_zero_ignores_bias                                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_out_vector_bias_shapes                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_out_noncontiguous                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_broadcast_bias                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_dtype_fp32_accum                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_dtype_beta_zero_ignores_bias                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_dtype_out_fp32_accum                                        |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_dtype_out_beta_zero_ignores_bias                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_large_matrix                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_large_matrix_alpha_beta                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_small_square_large                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_out_large_matrix                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_bounded_workspace                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_fallback_autotuning                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_ascend_beta_cache_transitions                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_ascend_review_empty                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_ascend_review_out_resize                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_addmm.py                                                          | test_addmm_ascend_review_beta_zero                                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte                                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_strided                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_fractional                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_requires_grad                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_complex                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_integer_limits                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_cast_Byte.py                                                      | test_cast_Byte_negative_view                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_views                                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dtype_matrix                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_resizes_mismatched_out                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_non_contiguous_gapped                     |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_transposed_view                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_complex_gapped                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_zero_tensor_rejected                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_registered_dispatch_path                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_registered_dispatch_path                  |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dispatch_non_contiguous                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dispatch_complex                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dispatch_complex_non_contiguous           |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_downstream_dispatch                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_inplace_view_rejected                         |      0 |      1 |       0 |       0 |        1 |
| tests/test_efficientzerotensor.py                                            | test_efficientzerotensor_out_dispatch_resize_growth                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner                                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_1d_dot                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_scalar                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_matvec                                                      |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_empty_contraction                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_2d                                                          |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_noncontiguous                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_inner.py                                                          | test_inner_dtype_mismatch                                              |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_tensor                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_scalar                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_scalar_tensor                                                 |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_scalar_int                                             |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_scalar_int_float_scalar                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_scalar_bool                                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_tensor_                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_tensor_scalar_                                                |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_broadcast_shape                                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_large_tensor_scalar                                           |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_large_tensor_scalar_dtypes                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_large_tensor_scalar_inplace                                   |      0 |      1 |       0 |       0 |        1 |
| tests/test_mul.py                                                            | test_mul_complex_grid_stride_review                                    |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                                       | test_sparse_compressed_tensor_explicit_size                            |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                                       | test_sparse_compressed_tensor_infer_size                               |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                                       | test_sparse_compressed_tensor_rejects_bad_layout                       |      0 |      1 |       0 |       0 |        1 |
| tests/test_sparse_compressed_tensor.py                                       | test_sparse_compressed_tensor_infer_size_scans_past_the_grid_cap       |      0 |      1 |       0 |       0 |        1 |
| tests/deprecated/test_FLA/test_chunk_gated_delta_rule_fwd.py                 | test_chunk_gated_delta_rule_fwd_accuracy                               |      0 |      0 |       0 |       1 |        1 |
| tests/deprecated/test_FLA/test_chunk_gated_delta_rule_fwd.py                 | test_chunk_gated_delta_rule_fwd_no_initial_state                       |      0 |      0 |       0 |       1 |        1 |
| tests/deprecated/test_FLA/test_chunk_gated_delta_rule_fwd.py                 | test_chunk_gated_delta_rule_fwd_with_cu_seqlens                        |      0 |      0 |       0 |       1 |        1 |
| TOTAL                                                                        |                                                                        |    100 |    231 |      40 |       3 |      374 |
