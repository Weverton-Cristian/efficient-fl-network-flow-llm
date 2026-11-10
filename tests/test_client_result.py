import pickle
import sys
import unittest


# Import the pure telemetry contracts without loading an installed torch/NVML
# stack through src.monitoring.__init__.  The modules under test do not need it.
_missing = object()
_blocked_modules = ("torch", "pynvml")
_previous_modules = {
    name: sys.modules.get(name, _missing) for name in _blocked_modules
}
for _name in _blocked_modules:
    sys.modules[_name] = None
try:
    from src.monitoring.client_result import (
        ClientTrainingResult,
        partition_client_results,
    )
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is _missing:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous


class ClientTrainingResultTests(unittest.TestCase):
    def test_partition_preserves_successful_id_and_weight_alignment(self):
        first_weights = {"adapter": [9.0]}
        second_weights = {"adapter": [2.0]}
        results = [
            ClientTrainingResult(
                round_num=3,
                client_id=9,
                gpu_id=1,
                status="success",
                weights=first_weights,
            ),
            ClientTrainingResult(
                round_num=3,
                client_id=4,
                gpu_id=0,
                status="failed",
                error={"type": "RuntimeError", "message": "out of memory"},
            ),
            ClientTrainingResult(
                round_num=3,
                client_id=7,
                gpu_id=None,
                status="skipped",
            ),
            ClientTrainingResult(
                round_num=3,
                client_id=2,
                gpu_id=0,
                status="success",
                weights=second_weights,
            ),
            ClientTrainingResult(
                round_num=3,
                client_id=5,
                gpu_id=1,
                status="failed",
            ),
        ]

        partitioned = partition_client_results(results)

        # Completion order is deliberately non-sorted.  The two parallel lists
        # must retain that same order so FedAvg associates each update with the
        # correct client sample count.
        self.assertEqual(partitioned.successful_client_ids, [9, 2])
        self.assertEqual(
            partitioned.client_weights,
            [first_weights, second_weights],
        )
        self.assertIs(partitioned.client_weights[0], first_weights)
        self.assertIs(partitioned.client_weights[1], second_weights)
        self.assertEqual(partitioned.skipped_client_ids, [7])
        self.assertEqual(
            partitioned.failure_messages,
            ["client_4: out of memory", "client_5: unknown error"],
        )

    def test_metric_row_is_flat_and_never_contains_weights(self):
        result = ClientTrainingResult(
            round_num=2,
            client_id=11,
            gpu_id=1,
            status="failed",
            metrics={
                "duration_seconds": 4.5,
                "gpu_energy_joules": None,
                "status": "success",
                "client_id": 999,
            },
            error={
                "type": "ValueError",
                "message": "invalid batch",
                "traceback": "traceback text",
            },
        )

        row = result.metric_row()

        self.assertEqual(row["round"], 2)
        self.assertEqual(row["client_id"], 11)
        self.assertEqual(row["gpu_id"], 1)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error_type"], "ValueError")
        self.assertEqual(row["error_message"], "invalid batch")
        self.assertEqual(row["error_traceback"], "traceback text")
        self.assertEqual(row["duration_seconds"], 4.5)
        self.assertIsNone(row["gpu_energy_joules"])
        self.assertNotIn("weights", row)

    def test_result_is_pickle_safe_for_process_pool_transport(self):
        original = ClientTrainingResult(
            round_num=1,
            client_id=6,
            gpu_id=0,
            status="success",
            weights={"adapter": [1, 2, 3]},
            metrics={"training_loss": 0.25},
        )

        restored = pickle.loads(pickle.dumps(original))

        self.assertEqual(restored, original)
        self.assertEqual(restored.weights, {"adapter": [1, 2, 3]})

    def test_invalid_status_and_weight_combinations_are_rejected(self):
        invalid_cases = [
            {"status": "unknown", "weights": None},
            {"status": "success", "weights": None},
            {"status": "failed", "weights": {"adapter": [1]}},
            {"status": "skipped", "weights": {"adapter": [1]}},
        ]

        for values in invalid_cases:
            with self.subTest(**values):
                with self.assertRaises(ValueError):
                    ClientTrainingResult(
                        round_num=1,
                        client_id=0,
                        gpu_id=None,
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
