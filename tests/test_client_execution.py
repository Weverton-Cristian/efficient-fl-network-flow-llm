import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _load_server_without_ml_dependencies():
    """Import the orchestration module with tiny stand-ins for the ML stack."""

    fake_torch = types.ModuleType("torch")

    class FakeTensor:
        pass

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def device_count():
            return 0

        @staticmethod
        def set_device(device):
            return None

        @staticmethod
        def manual_seed_all(seed):
            return None

        @staticmethod
        def empty_cache():
            return None

    fake_torch.Tensor = FakeTensor
    fake_torch.cuda = FakeCuda
    fake_torch.device = lambda value: value
    fake_torch.manual_seed = lambda seed: None

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.random = types.SimpleNamespace(seed=lambda seed: None)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = lambda path: None

    fake_model_loader = types.ModuleType("src.models.model_loader")
    fake_model_loader.initialize_global_model = lambda config: (None, None)

    fake_client = types.ModuleType("src.federated_learning.client")
    fake_client.ClientTrainer = object

    replacements = {
        "torch": fake_torch,
        "numpy": fake_numpy,
        "datasets": fake_datasets,
        "src.models.model_loader": fake_model_loader,
        "src.federated_learning.client": fake_client,
    }
    sys.modules.pop("src.federated_learning.server", None)
    with patch.dict(sys.modules, replacements):
        return importlib.import_module("src.federated_learning.server")


server_module = _load_server_without_ml_dependencies()


class _FakeMonitor:
    instances = []

    def __init__(self, config, gpu_id=None):
        self.enabled = True
        self.synchronize_cuda = False
        self.gpu_id = gpu_id
        self.started = False
        self.stopped = False
        self._metrics = None
        self.__class__.instances.append(self)

    @property
    def metrics(self):
        return self._metrics

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True
        self._metrics = {
            "duration_seconds": 2.5,
            "telemetry_enabled": True,
            "raw_samples": [{"elapsed_seconds": 0.0}],
        }
        return dict(self._metrics)


class _SuccessfulTrainer:
    received_config = None

    def __init__(self, client_id, config, gpu_id):
        self.__class__.received_config = config
        self.last_training_metrics = {"training_loss": 0.25}

    def train(self, round_num, learning_rate):
        return {"adapter": [1.0]}


class _FailingTrainer(_SuccessfulTrainer):
    def train(self, round_num, learning_rate):
        raise RuntimeError("simulated client failure")


class _FakeGlobalModel:
    def __init__(self):
        self.saved_paths = []
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def save_pretrained(self, path):
        self.saved_paths.append(path)


def _read_single_csv_row(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def _make_server(temp_dir):
    instance = object.__new__(server_module.FederatedServer)
    instance.config = {
        "num_rounds": 1,
        "results_path": temp_dir,
        "simulation_name": "telemetry-test",
    }
    instance.device = "cpu"
    instance.global_model = _FakeGlobalModel()
    instance.communication_metrics = []
    instance.telemetry_enabled = True
    instance.client_resource_metrics = []
    instance.round_resource_metrics = []
    instance.telemetry_writer = server_module.TelemetryMetricsWriter(
        Path(temp_dir) / "telemetry-test"
    )
    instance._select_clients_for_round = Mock(return_value=[2, 7])
    instance._get_learning_rate = Mock(return_value=0.01)
    instance._aggregate_models = Mock()
    instance._record_round_communication = Mock()
    return instance


class ExecuteClientTrainingTests(unittest.TestCase):
    def setUp(self):
        _FakeMonitor.instances = []

    def test_success_returns_metrics_and_strips_raw_samples(self):
        config = {
            "telemetry_enabled": "true",
            "telemetry_synchronize_cuda": "false",
        }
        with patch.object(server_module, "ResourceMonitor", _FakeMonitor), patch.object(
            server_module, "ClientTrainer", _SuccessfulTrainer
        ):
            result = server_module.execute_client_training(
                client_id=4,
                config=config,
                round_num=2,
                learning_rate=0.01,
                gpu_id=1,
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.weights, {"adapter": [1.0]})
        self.assertEqual(result.metrics["training_loss"], 0.25)
        self.assertEqual(result.metrics["duration_seconds"], 2.5)
        self.assertNotIn("raw_samples", result.metrics)
        self.assertTrue(_FakeMonitor.instances[0].started)
        self.assertTrue(_FakeMonitor.instances[0].stopped)
        self.assertIs(_SuccessfulTrainer.received_config["telemetry_enabled"], True)
        self.assertIs(
            _SuccessfulTrainer.received_config["telemetry_synchronize_cuda"],
            False,
        )

    def test_training_failure_is_returned_and_monitor_still_stops(self):
        with patch.object(server_module, "ResourceMonitor", _FakeMonitor), patch.object(
            server_module, "ClientTrainer", _FailingTrainer
        ):
            result = server_module.execute_client_training(
                client_id=8,
                config={},
                round_num=1,
                learning_rate=0.1,
            )

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.weights)
        self.assertEqual(result.error["type"], "RuntimeError")
        self.assertIn("simulated client failure", result.error["message"])
        self.assertTrue(_FakeMonitor.instances[0].stopped)

    def test_monitor_configuration_failure_becomes_client_failure(self):
        with patch.object(
            server_module,
            "ResourceMonitor",
            side_effect=ValueError("invalid telemetry interval"),
        ):
            result = server_module.execute_client_training(
                client_id=3,
                config={},
                round_num=1,
                learning_rate=0.1,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error["type"], "ValueError")
        self.assertIn("invalid telemetry interval", result.error["message"])


class SequentialTelemetryFlowTests(unittest.TestCase):
    def test_sequential_round_persists_success_failure_and_completed_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            results = [
                server_module.ClientTrainingResult(
                    round_num=1,
                    client_id=2,
                    gpu_id=None,
                    status="success",
                    weights={"adapter": [1.0]},
                    metrics={"duration_seconds": 1.0},
                ),
                server_module.ClientTrainingResult(
                    round_num=1,
                    client_id=7,
                    gpu_id=None,
                    status="failed",
                    error={"type": "RuntimeError", "message": "client failed"},
                ),
            ]

            with patch.object(
                server_module,
                "execute_client_training",
                side_effect=results,
            ):
                instance._run_sequential_training()

            instance._aggregate_models.assert_called_once_with(
                [{"adapter": [1.0]}],
                [2],
            )
            round_row = _read_single_csv_row(
                Path(temp_dir)
                / "telemetry-test"
                / "round_resource_metrics.csv"
            )
            self.assertEqual(round_row["status"], "completed")
            self.assertEqual(round_row["successful_clients"], "1")
            self.assertEqual(round_row["failed_clients"], "1")
            with (
                Path(temp_dir)
                / "telemetry-test"
                / "training_resource_summary.json"
            ).open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["completed_rounds"], 1)
            self.assertEqual(summary["aborted_rounds"], 0)

    def test_all_clients_without_updates_persist_aborted_round_before_raise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            results = [
                server_module.ClientTrainingResult(
                    round_num=1,
                    client_id=2,
                    gpu_id=None,
                    status="failed",
                    error={"type": "RuntimeError", "message": "client failed"},
                ),
                server_module.ClientTrainingResult(
                    round_num=1,
                    client_id=7,
                    gpu_id=None,
                    status="skipped",
                ),
            ]

            with patch.object(
                server_module,
                "execute_client_training",
                side_effect=results,
            ):
                with self.assertRaisesRegex(RuntimeError, "no selected client"):
                    instance._run_sequential_training()

            round_row = _read_single_csv_row(
                Path(temp_dir)
                / "telemetry-test"
                / "round_resource_metrics.csv"
            )
            self.assertEqual(round_row["status"], "aborted")
            self.assertEqual(round_row["failure_phase"], "client_training")
            instance._aggregate_models.assert_not_called()

    def test_aggregation_failure_persists_failed_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            instance._aggregate_models.side_effect = RuntimeError(
                "aggregation failed"
            )
            result = server_module.ClientTrainingResult(
                round_num=1,
                client_id=2,
                gpu_id=None,
                status="success",
                weights={"adapter": [1.0]},
            )
            instance._select_clients_for_round.return_value = [2]

            with patch.object(
                server_module,
                "execute_client_training",
                return_value=result,
            ):
                with self.assertRaisesRegex(RuntimeError, "aggregation failed"):
                    instance._run_sequential_training()

            round_row = _read_single_csv_row(
                Path(temp_dir)
                / "telemetry-test"
                / "round_resource_metrics.csv"
            )
            self.assertEqual(round_row["status"], "failed")
            self.assertEqual(round_row["failure_phase"], "aggregation")
            self.assertEqual(round_row["error_type"], "RuntimeError")

    def test_checkpoint_failure_persists_failed_round(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            instance.global_model.save_pretrained = Mock(
                side_effect=OSError("checkpoint disk failure")
            )
            result = server_module.ClientTrainingResult(
                round_num=1,
                client_id=2,
                gpu_id=None,
                status="success",
                weights={"adapter": [1.0]},
            )
            instance._select_clients_for_round.return_value = [2]

            with patch.object(
                server_module,
                "execute_client_training",
                return_value=result,
            ):
                with self.assertRaisesRegex(OSError, "checkpoint disk failure"):
                    instance._run_sequential_training()

            round_row = _read_single_csv_row(
                Path(temp_dir)
                / "telemetry-test"
                / "round_resource_metrics.csv"
            )
            self.assertEqual(round_row["status"], "failed")
            self.assertEqual(round_row["failure_phase"], "checkpoint")
            self.assertEqual(round_row["error_type"], "OSError")

    def test_disabled_telemetry_preserves_training_without_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            instance.telemetry_enabled = False
            instance._select_clients_for_round.return_value = [2]
            result = server_module.ClientTrainingResult(
                round_num=1,
                client_id=2,
                gpu_id=None,
                status="success",
                weights={"adapter": [1.0]},
            )

            with patch.object(
                server_module,
                "execute_client_training",
                return_value=result,
            ):
                instance._run_sequential_training()

            result_dir = Path(temp_dir) / "telemetry-test"
            self.assertFalse((result_dir / "client_resource_metrics.csv").exists())
            self.assertFalse((result_dir / "round_resource_metrics.csv").exists())
            self.assertFalse(
                (result_dir / "training_resource_summary.json").exists()
            )


class _ImmediateFuture:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _ImmediateExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, args):
        try:
            return _ImmediateFuture(result=function(args))
        except Exception as exc:
            return _ImmediateFuture(error=exc)


class ParallelTelemetryFlowTests(unittest.TestCase):
    def test_parallel_round_transports_results_and_preserves_weight_id_alignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = _make_server(temp_dir)
            instance.config["max_parallel_gpus"] = 2
            instance._select_clients_for_round.return_value = [2, 7, 9]

            def client_result(client_id, config, round_num, learning_rate, *, gpu_id):
                if client_id == 7:
                    return server_module.ClientTrainingResult(
                        round_num=round_num,
                        client_id=client_id,
                        gpu_id=gpu_id,
                        status="failed",
                        error={"type": "RuntimeError", "message": "worker failed"},
                    )
                return server_module.ClientTrainingResult(
                    round_num=round_num,
                    client_id=client_id,
                    gpu_id=gpu_id,
                    status="success",
                    weights={"adapter": [float(client_id)]},
                )

            with patch.object(
                server_module,
                "execute_client_training",
                side_effect=client_result,
            ), patch.object(
                server_module.torch.cuda,
                "device_count",
                return_value=2,
            ), patch.object(
                server_module.concurrent.futures,
                "ProcessPoolExecutor",
                _ImmediateExecutor,
            ), patch.object(
                server_module.concurrent.futures,
                "as_completed",
                side_effect=lambda futures: list(futures),
            ):
                instance._run_parallel_training()

            instance._aggregate_models.assert_called_once_with(
                [{"adapter": [2.0]}, {"adapter": [9.0]}],
                [2, 9],
            )
            round_row = _read_single_csv_row(
                Path(temp_dir)
                / "telemetry-test"
                / "round_resource_metrics.csv"
            )
            self.assertEqual(round_row["mode"], "parallel")
            self.assertEqual(round_row["status"], "completed")
            self.assertEqual(round_row["successful_client_ids"], "[2, 9]")
            self.assertEqual(round_row["failed_client_ids"], "[7]")


if __name__ == "__main__":
    unittest.main()
