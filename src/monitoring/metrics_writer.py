"""Standard-library persistence helpers for training resource telemetry."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


CLIENT_RESOURCE_FIELDS = [
    "round",
    "client_id",
    "gpu_id",
    "batch_index",
    "status",
    "error_type",
    "error_message",
    "error_traceback",
    "local_num_samples",
    "learning_rate",
    "global_step",
    "training_loss",
    "train_runtime_seconds",
    "train_samples_per_second",
    "train_steps_per_second",
    "total_flos",
    "epoch",
    "trainer_fp16",
    "trainer_bf16",
    "model_load_seconds",
    "dataset_load_seconds",
    "trainer_setup_seconds",
    "trainer_train_seconds",
    "adapter_extraction_seconds",
    "update_bytes",
    "update_params",
    "telemetry_enabled",
    "telemetry_strict",
    "telemetry_sample_count",
    "telemetry_error_count",
    "telemetry_errors",
    "process_id",
    "gpu_device_index",
    "gpu_nvml_mapping_source",
    "gpu_nvml_device_index",
    "gpu_nvml_device_uuid",
    "start_time_utc",
    "end_time_utc",
    "duration_seconds",
    "psutil_available",
    "cuda_available",
    "nvml_available",
    "process_cpu_percent_mean",
    "process_cpu_percent_peak",
    "process_rss_initial_bytes",
    "process_rss_end_bytes",
    "process_rss_peak_bytes",
    "process_tree_cpu_percent_mean",
    "process_tree_cpu_percent_peak",
    "process_tree_rss_initial_bytes",
    "process_tree_rss_end_bytes",
    "process_tree_rss_peak_bytes",
    "process_child_count_peak",
    "cuda_memory_allocated_initial_bytes",
    "cuda_memory_allocated_end_bytes",
    "cuda_memory_allocated_peak_bytes",
    "cuda_memory_reserved_initial_bytes",
    "cuda_memory_reserved_end_bytes",
    "cuda_memory_reserved_peak_bytes",
    "gpu_memory_used_initial_bytes",
    "gpu_memory_used_end_bytes",
    "gpu_memory_used_peak_bytes",
    "gpu_utilization_percent_mean",
    "gpu_utilization_percent_peak",
    "gpu_power_watts_mean",
    "gpu_power_watts_peak",
    "gpu_energy_joules",
    "gpu_energy_source",
]

ROUND_RESOURCE_FIELDS = [
    "round",
    "mode",
    "status",
    "failure_phase",
    "error_type",
    "error_message",
    "learning_rate",
    "selected_clients",
    "successful_clients",
    "failed_clients",
    "skipped_clients",
    "selected_client_ids",
    "successful_client_ids",
    "failed_client_ids",
    "skipped_client_ids",
    "client_phase_seconds",
    "aggregation_seconds",
    "checkpoint_seconds",
    "round_duration_seconds",
]


def _primitive(value: Any) -> Any:
    """Convert common scalar wrappers and non-finite floats safely."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _primitive(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str, allow_nan=False, sort_keys=True)
    return str(value)


def _normalize_row(
    row: Mapping[str, Any], fieldnames: list[str]
) -> dict[str, Any]:
    unknown = sorted(set(row) - set(fieldnames))
    if unknown:
        raise ValueError(f"Telemetry row contains unknown fields: {unknown}")
    return {field: _primitive(row.get(field)) for field in fieldnames}


def write_csv_atomic(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> None:
    """Atomically replace a CSV with the supplied rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_row(row, fieldnames) for row in rows]

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_json_atomic(
    path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> None:
    """Atomically write a strict JSON document (no NaN/Infinity)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: _primitive(value) for key, value in payload.items()}

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                normalized,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


class TelemetryMetricsWriter:
    """Persist telemetry snapshots under one simulation result directory."""

    def __init__(self, results_dir: str | os.PathLike[str]) -> None:
        self.results_dir = Path(results_dir)

    def write_clients(self, rows: Iterable[Mapping[str, Any]]) -> None:
        write_csv_atomic(
            self.results_dir / "client_resource_metrics.csv",
            rows,
            CLIENT_RESOURCE_FIELDS,
        )

    def write_rounds(self, rows: Iterable[Mapping[str, Any]]) -> None:
        write_csv_atomic(
            self.results_dir / "round_resource_metrics.csv",
            rows,
            ROUND_RESOURCE_FIELDS,
        )

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        write_json_atomic(
            self.results_dir / "training_resource_summary.json", payload
        )


__all__ = [
    "CLIENT_RESOURCE_FIELDS",
    "ROUND_RESOURCE_FIELDS",
    "TelemetryMetricsWriter",
    "write_csv_atomic",
    "write_json_atomic",
]
