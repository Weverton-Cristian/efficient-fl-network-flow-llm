import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.monitoring import resource_monitor
from src.monitoring.resource_monitor import ResourceMonitor


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.cpu_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_calls += 1
        return float(self.cpu_calls * 10)

    def memory_info(self):
        return SimpleNamespace(rss=1_024 * self.cpu_calls)

    def children(self, recursive=False):
        return []


class _FakePsutil:
    Process = _FakeProcess


class _FakeCuda:
    synchronized = []
    reset = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 1

    @staticmethod
    def device_count():
        return 2

    @classmethod
    def synchronize(cls, device):
        cls.synchronized.append(device)

    @classmethod
    def reset_peak_memory_stats(cls, device):
        cls.reset.append(device)

    @staticmethod
    def memory_allocated(device):
        return 100 + device

    @staticmethod
    def memory_reserved(device):
        return 200 + device

    @staticmethod
    def max_memory_allocated(device):
        return 300 + device

    @staticmethod
    def max_memory_reserved(device):
        return 400 + device


class _FakeTorch:
    cuda = _FakeCuda


class _FakeTreeParent(_FakeProcess):
    def __init__(self, pid):
        super().__init__(pid)
        self.child_a = _FakeProcess(101)
        self.child_b = _FakeProcess(102)

    def children(self, recursive=False):
        if not recursive:
            raise AssertionError("the process tree must be collected recursively")
        return [self.child_a, self.child_b]


class _FakeTreePsutil:
    parent = None

    @classmethod
    def Process(cls, pid):
        cls.parent = _FakeTreeParent(pid)
        return cls.parent


class _FakeNvml:
    initialized = False
    shutdown = False
    energy = 1_000

    @classmethod
    def nvmlInit(cls):
        cls.initialized = True

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index):
        return f"gpu-{index}"

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        return SimpleNamespace(used=512)

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        return SimpleNamespace(gpu=75)

    @staticmethod
    def nvmlDeviceGetPowerUsage(handle):
        return 50_000

    @classmethod
    def nvmlDeviceGetTotalEnergyConsumption(cls, handle):
        value = cls.energy
        cls.energy += 500
        return value

    @classmethod
    def nvmlShutdown(cls):
        cls.shutdown = True


class _MappingCuda(_FakeCuda):
    reported_uuid = None

    @classmethod
    def get_device_properties(cls, device):
        return SimpleNamespace(uuid=cls.reported_uuid)


class _MappingTorch:
    cuda = _MappingCuda


class _MappingNvml(_FakeNvml):
    device_uuids = {
        0: "GPU-zero-0000",
        1: "GPU-one-1111",
        2: "GPU-two-2222",
        3: "GPU-three-3333",
    }
    index_handle_calls = []
    uuid_handle_calls = []
    sampled_handles = []

    @classmethod
    def reset(cls):
        cls.initialized = False
        cls.shutdown = False
        cls.energy = 1_000
        cls.index_handle_calls = []
        cls.uuid_handle_calls = []
        cls.sampled_handles = []

    @classmethod
    def nvmlDeviceGetHandleByIndex(cls, index):
        cls.index_handle_calls.append(index)
        if index not in cls.device_uuids:
            raise ValueError("unknown NVML index")
        return f"handle-{index}"

    @classmethod
    def nvmlDeviceGetHandleByUUID(cls, device_uuid):
        if isinstance(device_uuid, bytes):
            device_uuid = device_uuid.decode("utf-8")
        cls.uuid_handle_calls.append(device_uuid)
        for index, actual_uuid in cls.device_uuids.items():
            if device_uuid == actual_uuid:
                return f"handle-{index}"
        raise ValueError("unknown NVML UUID")

    @classmethod
    def nvmlDeviceGetCount(cls):
        return len(cls.device_uuids)

    @classmethod
    def nvmlDeviceGetUUID(cls, handle):
        return cls.device_uuids[int(handle.split("-")[1])]

    @staticmethod
    def nvmlDeviceGetIndex(handle):
        return int(handle.split("-")[1])

    @classmethod
    def nvmlDeviceGetMemoryInfo(cls, handle):
        cls.sampled_handles.append(handle)
        return SimpleNamespace(used=512)


class _NeverStopsThread:
    def join(self, timeout=None):
        return None

    def is_alive(self):
        return True


def _enabled_config(**overrides):
    config = {
        "telemetry_enabled": True,
        "telemetry_sample_interval_seconds": 10,
        "telemetry_collect_cpu": True,
        "telemetry_collect_gpu": True,
        "telemetry_collect_power": True,
        "telemetry_save_raw_samples": True,
    }
    config.update(overrides)
    return config


class ResourceMonitorTests(unittest.TestCase):
    def test_disabled_monitor_does_not_initialize_optional_collectors(self):
        with patch.object(resource_monitor, "_psutil", None), patch.object(
            resource_monitor, "_torch", None
        ), patch.object(resource_monitor, "_pynvml", None):
            monitor = ResourceMonitor({"telemetry_enabled": False})
            monitor.start()
            metrics = monitor.stop()

        self.assertFalse(metrics["telemetry_enabled"])
        self.assertEqual(metrics["telemetry_sample_count"], 0)
        self.assertGreaterEqual(metrics["duration_seconds"], 0)
        self.assertEqual(metrics["telemetry_error_count"], 0)

    def test_non_strict_mode_degrades_gracefully_without_dependencies(self):
        with patch.object(resource_monitor, "_psutil", None), patch.object(
            resource_monitor, "_torch", None
        ), patch.object(resource_monitor, "_pynvml", None):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=0)
            monitor.start()
            metrics = monitor.stop()

        self.assertGreaterEqual(metrics["telemetry_sample_count"], 2)
        self.assertIsNone(metrics["process_rss_peak_bytes"])
        self.assertIsNone(metrics["cuda_memory_allocated_peak_bytes"])
        self.assertIsNone(metrics["gpu_energy_joules"])
        self.assertIn("psutil_unavailable", metrics["telemetry_errors"])
        self.assertIn("torch_unavailable", metrics["telemetry_errors"])

    def test_strict_mode_rejects_missing_requested_dependency(self):
        with patch.object(resource_monitor, "_psutil", None):
            monitor = ResourceMonitor(
                _enabled_config(
                    telemetry_strict=True,
                    telemetry_collect_gpu=False,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "psutil_unavailable"):
                monitor.start()

    def test_collects_process_cuda_and_nvml_metrics(self):
        _FakeCuda.synchronized = []
        _FakeCuda.reset = []
        _FakeNvml.initialized = False
        _FakeNvml.shutdown = False
        _FakeNvml.energy = 1_000

        with patch.dict(os.environ, {}, clear=True), patch.object(
            resource_monitor, "_psutil", _FakePsutil
        ), patch.object(resource_monitor, "_torch", _FakeTorch), patch.object(
            resource_monitor, "_pynvml", _FakeNvml
        ):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=1)
            monitor.start()
            metrics = monitor.stop()

        self.assertEqual(metrics["process_cpu_percent_peak"], 30.0)
        self.assertEqual(metrics["process_rss_initial_bytes"], 2_048.0)
        self.assertEqual(metrics["cuda_memory_allocated_peak_bytes"], 301.0)
        self.assertEqual(metrics["cuda_memory_reserved_peak_bytes"], 401.0)
        self.assertEqual(metrics["gpu_memory_used_peak_bytes"], 512.0)
        self.assertEqual(metrics["gpu_utilization_percent_mean"], 75.0)
        self.assertEqual(metrics["gpu_power_watts_mean"], 50.0)
        self.assertEqual(metrics["gpu_energy_joules"], 0.5)
        self.assertEqual(
            metrics["gpu_energy_source"], "nvml_total_energy_counter"
        )
        self.assertTrue(metrics["nvml_available"])
        self.assertEqual(
            metrics["gpu_nvml_mapping_source"], "logical_index_fallback"
        )
        self.assertEqual(metrics["gpu_nvml_device_index"], 1)
        self.assertEqual(len(metrics["raw_samples"]), 2)
        self.assertEqual(_FakeCuda.synchronized, [1, 1])
        self.assertEqual(_FakeCuda.reset, [1])
        self.assertTrue(_FakeNvml.shutdown)

    def test_process_tree_metrics_include_recursive_children(self):
        with patch.object(
            resource_monitor, "_psutil", _FakeTreePsutil
        ), patch.object(resource_monitor, "_torch", None):
            monitor = ResourceMonitor(
                _enabled_config(telemetry_collect_gpu=False), gpu_id=None
            )
            monitor.start()
            metrics = monitor.stop()

        # Existing parent-only metrics remain unchanged.
        self.assertEqual(metrics["process_cpu_percent_peak"], 30.0)
        self.assertEqual(metrics["process_rss_initial_bytes"], 2_048.0)
        self.assertEqual(metrics["process_rss_end_bytes"], 3_072.0)
        # The tree is parent + two recursively discovered children.
        self.assertEqual(metrics["process_tree_cpu_percent_mean"], 75.0)
        self.assertEqual(metrics["process_tree_cpu_percent_peak"], 90.0)
        self.assertEqual(metrics["process_tree_rss_initial_bytes"], 6_144.0)
        self.assertEqual(metrics["process_tree_rss_end_bytes"], 9_216.0)
        self.assertEqual(metrics["process_tree_rss_peak_bytes"], 9_216.0)
        self.assertEqual(metrics["process_child_count_peak"], 2)
        self.assertEqual(set(monitor._child_processes), {101, 102})
        self.assertEqual(monitor._child_processes[101].cpu_calls, 3)
        self.assertEqual(monitor._child_processes[102].cpu_calls, 3)

    def test_torch_uuid_has_priority_for_nvml_mapping(self):
        _MappingNvml.reset()
        _MappingCuda.reported_uuid = "GPU-three-3333"
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True
        ), patch.object(resource_monitor, "_psutil", _FakePsutil), patch.object(
            resource_monitor, "_torch", _MappingTorch
        ), patch.object(resource_monitor, "_pynvml", _MappingNvml):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=0)
            monitor.start()
            metrics = monitor.stop()

        self.assertEqual(metrics["gpu_nvml_mapping_source"], "torch_cuda_uuid")
        self.assertEqual(metrics["gpu_nvml_device_index"], 3)
        self.assertEqual(metrics["gpu_nvml_device_uuid"], "GPU-three-3333")
        self.assertEqual(set(_MappingNvml.sampled_handles), {"handle-3"})

    def test_raw_torch_uuid_bytes_are_normalized(self):
        raw_uuid = bytes.fromhex("00112233445566778899aabbccddeeff")
        self.assertEqual(
            ResourceMonitor._normalize_uuid(raw_uuid),
            "GPU-00112233-4455-6677-8899-aabbccddeeff",
        )

    def test_visible_devices_physical_index_maps_logical_cuda_device(self):
        _MappingNvml.reset()
        _MappingCuda.reported_uuid = None
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "3,1"}, clear=True
        ), patch.object(resource_monitor, "_psutil", _FakePsutil), patch.object(
            resource_monitor, "_torch", _MappingTorch
        ), patch.object(resource_monitor, "_pynvml", _MappingNvml):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=1)
            monitor.start()
            metrics = monitor.stop()

        self.assertEqual(
            metrics["gpu_nvml_mapping_source"], "cuda_visible_devices_index"
        )
        self.assertEqual(metrics["gpu_nvml_device_index"], 1)
        self.assertEqual(metrics["gpu_nvml_device_uuid"], "GPU-one-1111")
        self.assertEqual(set(_MappingNvml.sampled_handles), {"handle-1"})

    def test_visible_devices_uuid_prefix_is_resolved_uniquely(self):
        _MappingNvml.reset()
        _MappingCuda.reported_uuid = None
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-three"}, clear=True
        ), patch.object(resource_monitor, "_psutil", _FakePsutil), patch.object(
            resource_monitor, "_torch", _MappingTorch
        ), patch.object(resource_monitor, "_pynvml", _MappingNvml):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=0)
            monitor.start()
            metrics = monitor.stop()

        self.assertEqual(
            metrics["gpu_nvml_mapping_source"], "cuda_visible_devices_uuid"
        )
        self.assertEqual(metrics["gpu_nvml_device_index"], 3)
        self.assertEqual(metrics["gpu_nvml_device_uuid"], "GPU-three-3333")
        self.assertEqual(set(_MappingNvml.sampled_handles), {"handle-3"})

    def test_unresolvable_explicit_visible_devices_skips_nvml_safely(self):
        _MappingNvml.reset()
        _MappingCuda.reported_uuid = None
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "not-a-real-device"}, clear=True
        ), patch.object(resource_monitor, "_psutil", _FakePsutil), patch.object(
            resource_monitor, "_torch", _MappingTorch
        ), patch.object(resource_monitor, "_pynvml", _MappingNvml):
            monitor = ResourceMonitor(_enabled_config(), gpu_id=0)
            monitor.start()
            metrics = monitor.stop()

        self.assertFalse(metrics["nvml_available"])
        self.assertIsNone(metrics["gpu_nvml_mapping_source"])
        self.assertIsNone(metrics["gpu_memory_used_peak_bytes"])
        self.assertIn("nvml_device_mapping", metrics["telemetry_errors"])
        self.assertEqual(_MappingNvml.sampled_handles, [])
        self.assertTrue(_MappingNvml.shutdown)

    def test_sampling_thread_timeout_is_reported_without_concurrent_teardown(self):
        monitor = ResourceMonitor({"telemetry_enabled": False})
        monitor.start()
        monitor.enabled = True
        monitor._nvml_initialized = True
        monitor._thread = _NeverStopsThread()
        with patch.object(monitor, "_shutdown_nvml") as shutdown:
            metrics = monitor.stop()

        self.assertIn("sampling_thread_timeout", metrics["telemetry_errors"])
        self.assertEqual(metrics["telemetry_sample_count"], 0)
        shutdown.assert_not_called()

    def test_sampling_thread_timeout_surfaces_in_strict_mode(self):
        monitor = ResourceMonitor(
            {"telemetry_enabled": False, "telemetry_strict": True}
        )
        monitor.start()
        monitor.enabled = True
        monitor._thread = _NeverStopsThread()
        with self.assertRaisesRegex(RuntimeError, "sampling_thread_timeout"):
            monitor.stop()

    def test_context_manager_and_lifecycle(self):
        monitor = ResourceMonitor(
            _enabled_config(
                telemetry_collect_cpu=False,
                telemetry_collect_gpu=False,
                telemetry_save_raw_samples=False,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "before start"):
            monitor.stop()

        with monitor:
            pass

        self.assertIsNotNone(monitor.metrics)
        self.assertGreaterEqual(monitor.metrics["telemetry_sample_count"], 2)
        self.assertNotIn("raw_samples", monitor.metrics)
        self.assertEqual(monitor.stop(), monitor.metrics)

    def test_configuration_validation(self):
        for value in (0, -1, "not-a-number", None):
            with self.subTest(interval=value):
                with self.assertRaisesRegex(ValueError, "positive number"):
                    ResourceMonitor(
                        {
                            "telemetry_enabled": True,
                            "telemetry_sample_interval_seconds": value,
                        }
                    )
        for value in (-1, True, "1"):
            with self.subTest(gpu_id=value):
                with self.assertRaisesRegex(ValueError, "gpu_id"):
                    ResourceMonitor(_enabled_config(), gpu_id=value)

    def test_energy_falls_back_to_power_integration(self):
        samples = [
            {"elapsed_seconds": 0.0, "gpu_power_watts": 10.0},
            {"elapsed_seconds": 2.0, "gpu_power_watts": 20.0},
            {"elapsed_seconds": 3.0, "gpu_power_watts": 40.0},
        ]
        energy, source = ResourceMonitor._calculate_energy(samples)
        self.assertAlmostEqual(energy, 60.0)
        self.assertEqual(source, "power_integration")


if __name__ == "__main__":
    unittest.main()
