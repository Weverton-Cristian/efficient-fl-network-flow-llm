"""Serializable client-training results shared by sequential and parallel FL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


VALID_CLIENT_STATUSES = {"success", "skipped", "failed"}


@dataclass
class ClientTrainingResult:
    """Result returned by one client-training attempt.

    ``weights`` may contain torch tensors and is intentionally excluded from
    metric rows.  Every other field is composed of pickle/CSV-friendly values.
    """

    round_num: int
    client_id: int
    gpu_id: int | None
    status: str
    weights: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_CLIENT_STATUSES:
            raise ValueError(
                f"Invalid client result status {self.status!r}. "
                f"Expected one of {sorted(VALID_CLIENT_STATUSES)}."
            )
        if self.status == "success" and self.weights is None:
            raise ValueError("A successful client result must contain weights.")
        if self.status != "success" and self.weights is not None:
            raise ValueError("Skipped/failed client results cannot contain weights.")

    def metric_row(self) -> dict[str, Any]:
        """Return a flat row without model weights."""

        row: dict[str, Any] = dict(self.metrics)
        # Identity/error fields are authoritative and cannot be overwritten by
        # a collector metric with the same name.
        row.update({
            "round": int(self.round_num),
            "client_id": int(self.client_id),
            "gpu_id": self.gpu_id,
            "status": self.status,
            "error_type": None,
            "error_message": None,
            "error_traceback": None,
        })
        if self.error:
            row["error_type"] = self.error.get("type")
            row["error_message"] = self.error.get("message")
            row["error_traceback"] = self.error.get("traceback")
        return row


@dataclass
class PartitionedClientResults:
    """Aggregation-ready view of a collection of client results."""

    client_weights: list[Any]
    successful_client_ids: list[int]
    failure_messages: list[str]
    skipped_client_ids: list[int]


def partition_client_results(
    results: Iterable[ClientTrainingResult],
) -> PartitionedClientResults:
    """Separate successful updates while preserving ID/weight alignment."""

    client_weights: list[Any] = []
    successful_client_ids: list[int] = []
    failure_messages: list[str] = []
    skipped_client_ids: list[int] = []

    for result in results:
        if result.status == "success":
            successful_client_ids.append(result.client_id)
            client_weights.append(result.weights)
        elif result.status == "skipped":
            skipped_client_ids.append(result.client_id)
        else:
            message = "unknown error"
            if result.error:
                message = result.error.get("message") or message
            failure_messages.append(f"client_{result.client_id}: {message}")

    return PartitionedClientResults(
        client_weights=client_weights,
        successful_client_ids=successful_client_ids,
        failure_messages=failure_messages,
        skipped_client_ids=skipped_client_ids,
    )


__all__ = [
    "ClientTrainingResult",
    "PartitionedClientResults",
    "partition_client_results",
]
