import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


# Keep these standard-library persistence tests independent of the ML runtime.
_missing = object()
_blocked_modules = ("torch", "pynvml")
_previous_modules = {
    name: sys.modules.get(name, _missing) for name in _blocked_modules
}
for _name in _blocked_modules:
    sys.modules[_name] = None
try:
    from src.monitoring import metrics_writer
    from src.monitoring.metrics_writer import (
        CLIENT_RESOURCE_FIELDS,
        ROUND_RESOURCE_FIELDS,
        TelemetryMetricsWriter,
        write_csv_atomic,
        write_json_atomic,
    )
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is _missing:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class MetricsWriterTests(unittest.TestCase):
    def test_csv_uses_declared_schema_and_replaces_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "metrics.csv"
            fieldnames = ["client_id", "status", "value"]

            write_csv_atomic(
                path,
                [
                    {"client_id": 9, "status": "success", "value": 1.5},
                    {"client_id": 4, "status": "failed"},
                ],
                fieldnames,
            )
            write_csv_atomic(
                path,
                [{"client_id": 2, "status": "skipped", "value": math.nan}],
                fieldnames,
            )

            header, rows = _read_csv(path)

            self.assertEqual(header, fieldnames)
            self.assertEqual(
                rows,
                [{"client_id": "2", "status": "skipped", "value": ""}],
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_csv_rejects_unknown_fields_before_replacing_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown fields"):
                write_csv_atomic(
                    path,
                    [{"client_id": 1, "unexpected": 42}],
                    ["client_id"],
                )

            self.assertEqual(path.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_csv_replace_failure_preserves_old_file_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text("stable snapshot\n", encoding="utf-8")

            with patch.object(
                metrics_writer.os,
                "replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "replace failure"):
                    write_csv_atomic(
                        path,
                        [{"client_id": 3}],
                        ["client_id"],
                    )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "stable snapshot\n",
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_json_is_strict_atomic_and_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "summary.json"

            write_json_atomic(
                path,
                {
                    "description": "telemetria concluída",
                    "peak": math.inf,
                    "energy": 12.5,
                },
            )

            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            self.assertEqual(payload["description"], "telemetria concluída")
            self.assertIsNone(payload["peak"])
            self.assertEqual(payload["energy"], 12.5)
            self.assertNotIn("Infinity", raw)
            self.assertNotIn("NaN", raw)
            self.assertTrue(raw.endswith("\n"))
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_json_replace_failure_preserves_old_file_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            path.write_text('{"state": "old"}\n', encoding="utf-8")

            with patch.object(
                metrics_writer.os,
                "replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "replace failure"):
                    write_json_atomic(path, {"state": "new"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"state": "old"},
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_telemetry_writer_creates_all_artifacts_with_fixed_schemas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "simulation"
            writer = TelemetryMetricsWriter(results_dir)

            writer.write_clients(
                [
                    {
                        "round": 1,
                        "client_id": 8,
                        "status": "success",
                        "duration_seconds": 3.25,
                    },
                    {
                        "round": 1,
                        "client_id": 1,
                        "status": "failed",
                        "error_message": "out of memory",
                    },
                ]
            )
            writer.write_rounds(
                [
                    {
                        "round": 1,
                        "mode": "parallel",
                        "status": "completed",
                        "selected_clients": 2,
                        "successful_clients": 1,
                        "failed_clients": 1,
                        "skipped_clients": 0,
                    }
                ]
            )
            writer.write_summary(
                {
                    "client_attempts": 2,
                    "successful_clients": 1,
                    "failed_clients": 1,
                }
            )

            client_header, client_rows = _read_csv(
                results_dir / "client_resource_metrics.csv"
            )
            round_header, round_rows = _read_csv(
                results_dir / "round_resource_metrics.csv"
            )
            with (results_dir / "training_resource_summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

            self.assertEqual(client_header, CLIENT_RESOURCE_FIELDS)
            self.assertEqual(round_header, ROUND_RESOURCE_FIELDS)
            self.assertEqual(len(client_rows), 2)
            self.assertEqual(client_rows[0]["client_id"], "8")
            self.assertEqual(client_rows[1]["status"], "failed")
            self.assertEqual(client_rows[1]["error_message"], "out of memory")
            self.assertNotIn("weights", client_header)
            self.assertEqual(len(round_rows), 1)
            self.assertEqual(round_rows[0]["status"], "completed")
            self.assertEqual(round_rows[0]["selected_clients"], "2")
            self.assertEqual(
                summary,
                {
                    "client_attempts": 2,
                    "failed_clients": 1,
                    "successful_clients": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
