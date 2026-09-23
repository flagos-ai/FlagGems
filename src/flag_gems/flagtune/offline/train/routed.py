# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect unpartitioned routed workloads and fit independent models."""

import json
import os
import random
from pathlib import Path

from flag_gems.flagtune.inference.status import exception_reason, print_status


def run_routed_training(args, spec, *, training_api):
    """Resolve routes only inside GPU workers, then group successful samples.

    Raw recipes never choose a contract or a candidate domain in the parent.
    Sampling is performed per complete model identity after collection, so a
    partial-stage model cannot absorb rows from a public-output model.
    """
    from triton.flagtune.contract.identity import ModelIdentity
    from triton.flagtune.training.ranker import (
        export_ranker_model,
        train_xgboost_ranker,
    )

    from ..cli.pretune import resolve_max_shapes

    train = training_api

    context, _ = train.initialize_planning_context(spec)
    records = train.load_shape_records(
        Path(args.shape_config).expanduser().resolve(), spec
    )
    records = train.select_shape_records(
        records, spec, args.variant, train.parse_sort("default"), None
    )
    parallel = args.parallel or context.visible_device_count
    if not 0 < parallel <= context.visible_device_count:
        raise train.TrainError(
            "parallel exceeds available devices or no device is visible"
        )
    tokens = train.visible_device_tokens(context)[:parallel]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "op_id": spec.op_id,
                    "route_selection": "executor",
                    "shape_count": len(records),
                    "variant_filter": args.variant,
                }
            )
        )
        return 0
    run_dir = train.make_run_dir(Path(args.output).expanduser().resolve(), spec.op_id)
    database_url, database_path, database_source = train._database_url(args, run_dir)
    groups = {}
    failed = skipped = 0
    previous = os.environ.get("FLAGTREE_AABS")
    previous_progress = os.environ.get("FLAGTUNE_TRAIN_PROGRESS_INTERVAL")
    os.environ["FLAGTREE_AABS"] = "0"
    os.environ["FLAGTUNE_TRAIN_PROGRESS_INTERVAL"] = str(
        0 if args.no_progress else args.progress_interval
    )
    try:
        size = args.shape_batch_size or parallel * 4
        for start in range(0, len(records), size):
            batch = train.run_shape_config_benchmarks(
                [
                    (record.to_benchmark_shape(), None)
                    for record in records[start : start + size]
                ],
                operator_config=Path(args.flagtune_config).expanduser().resolve(),
                dtypes=args.dtypes,
                warmup=args.warmup,
                iterations=args.iterations,
                benchmark_mode=args.benchmark_mode,
                benchmark_retries=args.benchmark_retries,
                tuning_run_mode="exhaustive_collection",
                parallel=min(parallel, len(records[start : start + size])),
                gpu_tokens=tokens[: min(parallel, len(records[start : start + size]))],
                database_url=database_url,
                work_dir=run_dir / "collection_batches" / str(start),
                fail_fast=args.fail_fast,
                stream_worker_logs=not args.no_progress,
            )
            if batch.database_merge_error or any(batch.worker_returncodes):
                raise train.TrainError(
                    f"worker/database failure: {batch.database_merge_error}"
                )
            if len(batch.results) != len(records[start : start + size]):
                raise train.TrainError("collection returned an incomplete batch")
            with (run_dir / "collection_status.jsonl").open("a") as audit:
                for result in batch.results:
                    status = result.get("status")
                    audit.write(
                        json.dumps(
                            {k: v for k, v in result.items() if k != "config_timings"}
                        )
                        + "\n"
                    )
                    if status == "skipped":
                        skipped += 1
                        continue
                    if status != "ok":
                        failed += 1
                        continue
                    key = (
                        result["platform_key"],
                        spec.op_id,
                        result["tuning_variant"],
                        result["dtype_key"],
                    )
                    if key not in groups:
                        path = run_dir / "groups" / str(len(groups))
                        path.mkdir(parents=True)
                        groups[key] = {"path": path, "count": 0, "example": result}
                    group = groups[key]
                    # Persist timings incrementally rather than retaining the
                    # complete expanded candidate corpus in parent memory.
                    with (group["path"] / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                    group["count"] += 1
            train._status(
                f"Routed collection {min(start + size, len(records))}/{len(records)}; "
                f"skipped={skipped}, failed={failed}"
            )
    finally:
        if previous is None:
            os.environ.pop("FLAGTREE_AABS", None)
        else:
            os.environ["FLAGTREE_AABS"] = previous
        if previous_progress is None:
            os.environ.pop("FLAGTUNE_TRAIN_PROGRESS_INTERVAL", None)
        else:
            os.environ["FLAGTUNE_TRAIN_PROGRESS_INTERVAL"] = previous_progress
    if failed or not groups:
        raise train.TrainError(
            f"routed collection failed={failed}, successful model groups={len(groups)}; inspect {run_dir}"
        )

    models = []
    for key, group in groups.items():
        result = group["example"]
        variant_info = spec.operator_info.variants[result["variant"]]
        indices = list(range(group["count"]))
        sort = train.parse_sort(args.sort_text)
        if sort.mode == "random":
            random.Random(sort.seed).shuffle(indices)
        elif sort.mode.startswith("count_"):
            with (group["path"] / "results.jsonl").open() as stream:
                counts = [json.loads(line).get("Count") for line in stream]
            if any(count is None for count in counts):
                raise train.TrainError("count sorting requires Count for every shape")
            indices.sort(
                key=lambda i: counts[i], reverse=sort.mode == "count_descending"
            )
        selected = set(indices[: resolve_max_shapes(args.max_shapes, len(indices))])
        if not selected:
            raise train.TrainError(
                f"training selection is empty for model identity {key}"
            )
        data_path = group["path"] / "benchmark_data.jsonl"
        selected_rows = []
        protocols = {}
        with (group["path"] / "results.jsonl").open() as stream:
            for index, line in enumerate(stream):
                if index not in selected:
                    continue
                row = json.loads(line)
                if (
                    row["model_dtypes"] != result["model_dtypes"]
                    or row["gpu_metadata"] != result["gpu_metadata"]
                ):
                    raise train.TrainError(
                        f"collection mixed platform/architecture metadata or model dtypes within {key}"
                    )
                selected_rows.append(row)
                protocol = row.get("benchmark_protocol")
                if isinstance(protocol, dict):
                    protocols[json.dumps(protocol, sort_keys=True)] = protocol
        # Sort the complete per-model selection in the existing serializer so
        # duplicate normalized inputs stay in contiguous ranker groups.
        _, finite, failures = train._append_collection_rows(
            data_path,
            group["path"] / "failures.jsonl",
            selected_rows,
            variant_info,
            spec.shape.identity,
        )
        del selected_rows
        if failures:
            raise train.TrainError(f"invalid training samples for {key}")
        model, summary = train_xgboost_ranker(
            variant_info, data_path, train._training_options(args)
        )
        summary = {**summary, "benchmark_protocols": list(protocols.values())}
        identity = ModelIdentity(
            platform_key=key[0], op_id=key[1], variant=key[2], dtype_key=key[3]
        )
        exported = export_ranker_model(
            model,
            variant_info,
            run_dir,
            summary,
            identity=identity,
            dtypes=result["model_dtypes"],
            gpu=result["gpu_metadata"],
            model_version=args.model_version,
        )
        models.append(
            {
                "identity": dict(
                    zip(("platform_key", "op_id", "variant", "dtype_key"), key)
                ),
                "path": str(exported.model_path),
                "collected_shapes": group["count"],
                "training_shapes": len(selected),
                "finite_rows": finite,
                "training_summary": summary,
            }
        )
        train._status(f"Exported {key}: shapes={len(selected)}, finite_rows={finite}")
    removed = []
    cleanup_error = ""
    if not args.keep_intermediate_files:
        paths = [run_dir / "collection_batches", run_dir / "groups"]
        if database_source == "run directory":
            paths.append(database_path)
        try:
            removed = train.remove_intermediate_artifacts(run_dir, paths)
        except train.PretuneIOError as exc:
            cleanup_error = str(exc)
            print_status(
                "WARNING: intermediate-file cleanup failed",
                reason=exception_reason(exc),
                run_dir=run_dir,
                manifest=run_dir / "manifest.json",
                action=(
                    "Models were exported successfully; some intermediate files "
                    "may remain. Inspect the run directory before removing them."
                ),
            )
    train.write_manifest(
        run_dir / "manifest.json",
        {
            "route_selection": "executor",
            "database": str(database_path),
            "selected_shape_count": len(records),
            "skipped_shape_count": skipped,
            "failed_shape_count": failed,
            "models": models,
            "cleanup": {
                "keep_intermediate_files": bool(args.keep_intermediate_files),
                "removed": removed,
                "error": cleanup_error,
            },
        },
    )
    print(f"FlagTune training output: {run_dir}")
    return 0
