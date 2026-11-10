"""Lightweight, optional resource telemetry for local training processes.

The monitor deliberately has no mandatory third-party dependency.  Process
metrics are collected when :mod:`psutil` is available, CUDA allocator metrics
when :mod:`torch` can see a GPU, and device-level metrics when NVML is
available.  This makes it safe to import the module in data-processing and CPU
only environments.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any, Mapping

try:  # Optional: the project must still import without profiling dependencies.
    import psutil as _psutil
except (ImportError, OSError):  # pragma: no cover - depends on host environment
    _psutil = None

try:  # Optional for CPU-only preprocessing and test environments.
    import torch as _torch
except (ImportError, OSError):  # pragma: no cover - depends on host environment
    _torch = None

try:  # Package name is nvidia-ml-py; its import name is pynvml.
    import pynvml as _pynvml
except (ImportError, OSError):  # pragma: no cover - depends on host environment
    _pynvml = None


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _config_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{key} must be a boolean, got {value!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


class ResourceMonitor:
    """Sample CPU, RAM and GPU resources used during one unit of work.

    Configuration is intentionally flat so the class can receive the existing
    experiment configuration directly.  Supported keys are:

    ``telemetry_enabled`` (default ``False``)
        Disable all dependency initialization and sampling when false.
    ``telemetry_strict`` (default ``False``)
        Raise when a requested collector is unavailable or fails.  In the
        default non-strict mode, available metrics are still returned and the
        failures are described by ``telemetry_errors``.
    ``telemetry_sample_interval_seconds`` (default ``0.5``)
        Background sampling period.
    ``telemetry_collect_cpu`` / ``telemetry_collect_gpu``
        Enable process or GPU collectors (both default to ``True``).
        CPU/RSS output includes both the worker process and an aggregate of its
        recursive child tree (for example, PyTorch DataLoader workers).
    ``telemetry_collect_power`` (default ``True``)
        Collect NVML power and energy data when GPU collection is enabled.
    ``telemetry_save_raw_samples`` (default ``False``)
        Include the samples in the result returned by :meth:`stop`.
    ``telemetry_synchronize_cuda`` (default ``True``)
        Synchronize the selected CUDA device at the measurement boundaries.

    CUDA allocator metrics are process-local. NVML metrics describe the whole
    physical device. The logical CUDA device is matched to NVML by torch UUID
    or ``CUDA_VISIBLE_DEVICES`` when available, and the mapping is reported in
    the resulting metadata.

    The monitor is reusable, but each ``start`` must be paired with ``stop``.
    It can also be used as a context manager; retrieve the result from
    :attr:`metrics` after leaving the context.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        gpu_id: int | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.enabled = _config_bool(self.config, "telemetry_enabled", False)
        self.strict = _config_bool(self.config, "telemetry_strict", False)
        self.collect_cpu = _config_bool(
            self.config, "telemetry_collect_cpu", True
        )
        self.collect_gpu = _config_bool(
            self.config, "telemetry_collect_gpu", True
        )
        self.collect_power = _config_bool(
            self.config, "telemetry_collect_power", True
        )
        self.save_raw_samples = _config_bool(
            self.config, "telemetry_save_raw_samples", False
        )
        self.synchronize_cuda = _config_bool(
            self.config, "telemetry_synchronize_cuda", True
        )

        try:
            self.sample_interval = float(
                self.config.get("telemetry_sample_interval_seconds", 0.5)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "telemetry_sample_interval_seconds must be a positive number"
            ) from exc
        if self.sample_interval <= 0:
            raise ValueError(
                "telemetry_sample_interval_seconds must be a positive number"
            )

        configured_gpu_id = self.config.get("telemetry_gpu_id")
        if gpu_id is None and configured_gpu_id is not None:
            try:
                gpu_id = int(configured_gpu_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("telemetry_gpu_id must be an integer") from exc
        if gpu_id is not None and (
            isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
        ):
            raise ValueError("gpu_id must be a non-negative integer or None")
        self.gpu_id = gpu_id

        self._process: Any = None
        self._child_processes: dict[int, Any] = {}
        self._nvml_handle: Any = None
        self._nvml_initialized = False
        self._nvml_available = False
        self._nvml_mapping_source: str | None = None
        self._nvml_device_index: int | None = None
        self._nvml_device_uuid: str | None = None
        self._cuda_available = False
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._error_keys: set[str] = set()
        self._fatal_error: RuntimeError | None = None
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._defer_nvml_shutdown_to_sampler = False
        self._sampler_finished = False
        self._thread_state_lock = threading.Lock()
        self._started = False
        self._running = False
        self._start_monotonic: float | None = None
        self._start_time_utc: str | None = None
        self._metrics: dict[str, Any] | None = None

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Return a defensive copy of the most recently completed metrics."""

        return dict(self._metrics) if self._metrics is not None else None

    @property
    def raw_samples(self) -> list[dict[str, float]]:
        """Return copies of samples collected so far (mainly for diagnostics)."""

        with self._sample_lock:
            return [dict(sample) for sample in self._samples]

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> "ResourceMonitor":
        """Start a new measurement window and return ``self``."""

        if self._running:
            raise RuntimeError("ResourceMonitor is already running")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "the previous ResourceMonitor sampling thread is still running"
            )

        self._reset_run_state()
        self._started = True
        self._running = True
        self._start_monotonic = time.perf_counter()
        self._start_time_utc = _utc_now()

        if not self.enabled:
            return self

        try:
            self._initialize_collectors()
            self._synchronize_cuda_if_requested()
            self._reset_cuda_peaks()
            self._collect_sample()
            self._thread = threading.Thread(
                target=self._sampling_loop,
                name=f"resource-monitor-{os.getpid()}",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self._running = False
            self._shutdown_nvml()
            raise
        return self

    def stop(self) -> dict[str, Any]:
        """Stop sampling, summarize the window and return serializable metrics."""

        if not self._started:
            raise RuntimeError("ResourceMonitor.stop() called before start()")
        if not self._running:
            if self._metrics is None:
                raise RuntimeError("ResourceMonitor is not running")
            return dict(self._metrics)

        self._stop_event.set()
        sampling_thread_alive = False
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval * 2.0))
            sampling_thread_alive = self._thread.is_alive()
            if sampling_thread_alive:
                with self._thread_state_lock:
                    if self._sampler_finished:
                        # The thread has left its sampling section and cannot
                        # touch NVML again, even if is_alive() has not flipped.
                        sampling_thread_alive = False
                    else:
                        # The sampler may be blocked inside a driver call. It
                        # must own teardown after that call returns.
                        self._defer_nvml_shutdown_to_sampler = True
                if sampling_thread_alive:
                    self._collector_error(
                        "sampling_thread_timeout",
                        "resource sampling thread did not stop before the "
                        "timeout; the final sample was skipped and NVML "
                        "teardown was deferred",
                        from_thread=True,
                    )

        if self.enabled and not sampling_thread_alive:
            self._synchronize_cuda_if_requested()
            self._collect_sample()

        end_monotonic = time.perf_counter()
        self._running = False
        if not sampling_thread_alive:
            self._shutdown_nvml()
        duration = max(0.0, end_monotonic - (self._start_monotonic or end_monotonic))
        self._metrics = self._summarize(duration, _utc_now())

        if self.strict and self._fatal_error is not None:
            error = self._fatal_error
            self._fatal_error = None
            raise error
        return dict(self._metrics)

    def _reset_run_state(self) -> None:
        self._process = None
        self._child_processes = {}
        self._nvml_handle = None
        self._nvml_initialized = False
        self._nvml_available = False
        self._nvml_mapping_source = None
        self._nvml_device_index = None
        self._nvml_device_uuid = None
        self._cuda_available = False
        self._samples = []
        self._errors = []
        self._error_keys = set()
        self._fatal_error = None
        self._stop_event = threading.Event()
        self._thread = None
        self._defer_nvml_shutdown_to_sampler = False
        self._sampler_finished = False
        self._thread_state_lock = threading.Lock()
        self._metrics = None

    def _initialize_collectors(self) -> None:
        if self.collect_cpu:
            if _psutil is None:
                self._collector_error(
                    "psutil_unavailable",
                    "psutil is not installed; CPU and RSS metrics are unavailable",
                )
            else:
                try:
                    self._process = _psutil.Process(os.getpid())
                    # The first non-blocking call establishes psutil's baseline.
                    self._process.cpu_percent(interval=None)
                    self._prime_process_children()
                except Exception as exc:
                    self._collector_exception("psutil_initialization", exc)

        if not self.collect_gpu:
            return
        if _torch is None:
            self._collector_error(
                "torch_unavailable",
                "torch is not installed; CUDA allocator metrics are unavailable",
            )
        else:
            try:
                self._cuda_available = bool(_torch.cuda.is_available())
            except Exception as exc:
                self._collector_exception("cuda_detection", exc)

        if not self._cuda_available:
            self._collector_error(
                "cuda_unavailable",
                "CUDA is unavailable; GPU telemetry was skipped",
            )
            return

        if self.gpu_id is None:
            try:
                self.gpu_id = int(_torch.cuda.current_device())
            except Exception as exc:
                self._collector_exception("cuda_device_detection", exc)
                return

        try:
            device_count = int(_torch.cuda.device_count())
            if self.gpu_id >= device_count:
                self._collector_error(
                    "invalid_gpu_id",
                    f"GPU {self.gpu_id} does not exist (device count: {device_count})",
                )
                self._cuda_available = False
                return
        except Exception as exc:
            self._collector_exception("cuda_device_count", exc)
            return

        if _pynvml is None:
            self._collector_error(
                "pynvml_unavailable",
                "pynvml is not installed; GPU utilization, driver VRAM, power and "
                "energy metrics are unavailable",
            )
            return

        try:
            _pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as exc:
            try:
                self._collector_exception("nvml_initialization", exc)
            finally:
                self._shutdown_nvml()
            return

        try:
            (
                self._nvml_handle,
                self._nvml_mapping_source,
                self._nvml_device_index,
                self._nvml_device_uuid,
            ) = self._resolve_nvml_device()
            self._nvml_available = True
        except Exception as exc:
            try:
                self._collector_exception("nvml_device_mapping", exc)
            finally:
                self._shutdown_nvml()

    def _prime_process_children(self) -> None:
        """Prime psutil CPU counters for the current recursive child tree."""

        if self._process is None:
            return
        try:
            children = self._process.children(recursive=True)
        except Exception as exc:
            self._collector_exception("process_children_discovery", exc)
            return

        for child in children:
            try:
                pid = int(child.pid)
                if pid in self._child_processes:
                    continue
                self._child_processes[pid] = child
                child.cpu_percent(interval=None)
            except Exception as exc:
                if not self._is_expected_process_exit(exc):
                    self._collector_exception("process_child_prime", exc)

    def _collect_process_tree_sample(
        self,
        sample: dict[str, float],
        *,
        process_cpu_percent: float | None,
        process_rss_bytes: float | None,
    ) -> None:
        """Add parent-plus-descendant aggregates to one process sample."""

        if self._process is None:
            return
        try:
            discovered_children = self._process.children(recursive=True)
        except Exception as exc:
            self._collector_exception(
                "process_children_discovery", exc, from_thread=True
            )
            return

        children_by_pid: dict[int, Any] = {}
        for child in discovered_children:
            try:
                children_by_pid[int(child.pid)] = child
            except (AttributeError, TypeError, ValueError):
                continue

        active_pids = set(children_by_pid)
        for stale_pid in set(self._child_processes) - active_pids:
            self._child_processes.pop(stale_pid, None)

        tree_cpu = process_cpu_percent
        tree_rss = process_rss_bytes
        for pid, discovered_child in children_by_pid.items():
            child = self._child_processes.get(pid)
            newly_discovered = child is None
            if newly_discovered:
                child = discovered_child
                self._child_processes[pid] = child
                try:
                    # A non-blocking first call establishes psutil's baseline.
                    child.cpu_percent(interval=None)
                except Exception as exc:
                    if not self._is_expected_process_exit(exc):
                        self._collector_exception(
                            "process_child_prime", exc, from_thread=True
                        )
                    continue

            try:
                child_cpu = 0.0 if newly_discovered else float(
                    child.cpu_percent(interval=None)
                )
                child_rss = float(child.memory_info().rss)
            except Exception as exc:
                if not self._is_expected_process_exit(exc):
                    self._collector_exception(
                        "process_child_sample", exc, from_thread=True
                    )
                continue

            if tree_cpu is not None:
                tree_cpu += child_cpu
            if tree_rss is not None:
                tree_rss += child_rss

        sample["process_child_count"] = float(len(active_pids))
        if tree_cpu is not None:
            sample["process_tree_cpu_percent"] = tree_cpu
        if tree_rss is not None:
            sample["process_tree_rss_bytes"] = tree_rss

    @staticmethod
    def _is_expected_process_exit(exc: Exception) -> bool:
        if _psutil is None:
            return False
        expected_types = tuple(
            exception_type
            for exception_type in (
                getattr(_psutil, "NoSuchProcess", None),
                getattr(_psutil, "ZombieProcess", None),
            )
            if isinstance(exception_type, type)
        )
        return bool(expected_types) and isinstance(exc, expected_types)

    def _resolve_nvml_device(self) -> tuple[Any, str, int | None, str | None]:
        """Resolve the CUDA logical device to the matching NVML handle safely."""

        if _pynvml is None or self.gpu_id is None:
            raise RuntimeError("NVML or the CUDA logical device is unavailable")

        torch_uuid = self._torch_cuda_device_uuid()
        if torch_uuid is not None:
            resolved = self._nvml_handle_from_uuid(torch_uuid, allow_prefix=False)
            if resolved is None:
                raise RuntimeError(
                    f"torch reported CUDA UUID {torch_uuid!r}, but NVML could not "
                    "resolve it"
                )
            handle, index, actual_uuid = resolved
            return handle, "torch_cuda_uuid", index, actual_uuid or torch_uuid

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            tokens = [token.strip() for token in visible_devices.split(",")]
            if not tokens or any(not token for token in tokens):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES is explicit but empty or malformed"
                )
            if self.gpu_id >= len(tokens):
                raise RuntimeError(
                    f"CUDA logical device {self.gpu_id} is outside explicit "
                    f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
                )

            token = tokens[self.gpu_id]
            try:
                physical_index = int(token)
            except ValueError:
                physical_index = None

            if physical_index is not None:
                if physical_index < 0:
                    raise RuntimeError(
                        f"invalid physical GPU index {physical_index} in "
                        "CUDA_VISIBLE_DEVICES"
                    )
                handle = _pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                return (
                    handle,
                    "cuda_visible_devices_index",
                    physical_index,
                    self._nvml_uuid_for_handle(handle),
                )

            resolved = self._nvml_handle_from_uuid(token, allow_prefix=True)
            if resolved is None:
                raise RuntimeError(
                    f"CUDA_VISIBLE_DEVICES token {token!r} could not be resolved "
                    "to exactly one NVML device"
                )
            handle, index, actual_uuid = resolved
            return handle, "cuda_visible_devices_uuid", index, actual_uuid

        handle = _pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        return (
            handle,
            "logical_index_fallback",
            self.gpu_id,
            self._nvml_uuid_for_handle(handle),
        )

    def _torch_cuda_device_uuid(self) -> str | None:
        if _torch is None or self.gpu_id is None:
            return None
        get_properties = getattr(_torch.cuda, "get_device_properties", None)
        if get_properties is None:
            return None
        try:
            properties = get_properties(self.gpu_id)
        except Exception:
            return None
        return self._normalize_uuid(getattr(properties, "uuid", None))

    def _nvml_handle_from_uuid(
        self, device_uuid: str, *, allow_prefix: bool
    ) -> tuple[Any, int | None, str | None] | None:
        if _pynvml is None:
            return None

        get_by_uuid = getattr(_pynvml, "nvmlDeviceGetHandleByUUID", None)
        if get_by_uuid is not None:
            candidates: tuple[Any, ...] = (device_uuid, device_uuid.encode("utf-8"))
            for candidate in candidates:
                try:
                    handle = get_by_uuid(candidate)
                    return (
                        handle,
                        self._nvml_index_for_handle(handle),
                        self._nvml_uuid_for_handle(handle) or device_uuid,
                    )
                except Exception:
                    continue

        get_count = getattr(_pynvml, "nvmlDeviceGetCount", None)
        if get_count is None:
            return None
        target = self._canonical_uuid(device_uuid)
        matches: list[tuple[Any, int, str]] = []
        try:
            device_count = int(get_count())
        except Exception:
            return None
        for index in range(device_count):
            try:
                handle = _pynvml.nvmlDeviceGetHandleByIndex(index)
                actual_uuid = self._nvml_uuid_for_handle(handle)
            except Exception:
                continue
            if actual_uuid is None:
                continue
            actual = self._canonical_uuid(actual_uuid)
            matches_uuid = actual == target
            if allow_prefix:
                matches_uuid = matches_uuid or actual.startswith(target)
            if matches_uuid:
                matches.append((handle, index, actual_uuid))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _normalize_uuid(value: Any) -> str | None:
        if isinstance(value, bytes):
            if len(value) == 16:
                return f"GPU-{uuidlib.UUID(bytes=value)}"
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @classmethod
    def _canonical_uuid(cls, value: Any) -> str:
        normalized = cls._normalize_uuid(value) or ""
        lowered = normalized.lower()
        return lowered[4:] if lowered.startswith("gpu-") else lowered

    def _nvml_uuid_for_handle(self, handle: Any) -> str | None:
        if _pynvml is None:
            return None
        get_uuid = getattr(_pynvml, "nvmlDeviceGetUUID", None)
        if get_uuid is None:
            return None
        try:
            return self._normalize_uuid(get_uuid(handle))
        except Exception:
            return None

    @staticmethod
    def _nvml_index_for_handle(handle: Any) -> int | None:
        if _pynvml is None:
            return None
        get_index = getattr(_pynvml, "nvmlDeviceGetIndex", None)
        if get_index is None:
            return None
        try:
            return int(get_index(handle))
        except Exception:
            return None

    def _sampling_loop(self) -> None:
        try:
            while not self._stop_event.wait(self.sample_interval):
                self._collect_sample()
                if self._fatal_error is not None:
                    self._stop_event.set()
                    return
        finally:
            with self._thread_state_lock:
                self._sampler_finished = True
                if self._defer_nvml_shutdown_to_sampler:
                    self._shutdown_nvml()

    def _collect_sample(self) -> None:
        if self._start_monotonic is None:
            return
        sample: dict[str, float] = {
            "elapsed_seconds": max(0.0, time.perf_counter() - self._start_monotonic)
        }

        if self._process is not None:
            process_cpu_percent: float | None = None
            process_rss_bytes: float | None = None
            try:
                process_cpu_percent = float(
                    self._process.cpu_percent(interval=None)
                )
                process_rss_bytes = float(self._process.memory_info().rss)
                sample["process_cpu_percent"] = process_cpu_percent
                sample["process_rss_bytes"] = process_rss_bytes
            except Exception as exc:
                self._collector_exception("process_sample", exc, from_thread=True)
            self._collect_process_tree_sample(
                sample,
                process_cpu_percent=process_cpu_percent,
                process_rss_bytes=process_rss_bytes,
            )

        if self._cuda_available and _torch is not None and self.gpu_id is not None:
            cuda_metrics = (
                ("cuda_memory_allocated_bytes", "memory_allocated"),
                ("cuda_memory_reserved_bytes", "memory_reserved"),
                ("cuda_memory_allocated_peak_bytes", "max_memory_allocated"),
                ("cuda_memory_reserved_peak_bytes", "max_memory_reserved"),
            )
            for metric_name, function_name in cuda_metrics:
                try:
                    function = getattr(_torch.cuda, function_name)
                    sample[metric_name] = float(function(self.gpu_id))
                except Exception as exc:
                    self._collector_exception(
                        f"torch_{function_name}", exc, from_thread=True
                    )

        if self._nvml_handle is not None and _pynvml is not None:
            try:
                memory = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                sample["gpu_memory_used_bytes"] = float(memory.used)
            except Exception as exc:
                self._collector_exception("nvml_memory", exc, from_thread=True)
            try:
                utilization = _pynvml.nvmlDeviceGetUtilizationRates(
                    self._nvml_handle
                )
                sample["gpu_utilization_percent"] = float(utilization.gpu)
            except Exception as exc:
                self._collector_exception(
                    "nvml_utilization", exc, from_thread=True
                )

            if self.collect_power:
                try:
                    sample["gpu_power_watts"] = float(
                        _pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                    ) / 1000.0
                except Exception as exc:
                    self._collector_exception("nvml_power", exc, from_thread=True)
                energy_function = getattr(
                    _pynvml, "nvmlDeviceGetTotalEnergyConsumption", None
                )
                if energy_function is not None:
                    try:
                        sample["gpu_total_energy_millijoules"] = float(
                            energy_function(self._nvml_handle)
                        )
                    except Exception as exc:
                        # Not every NVIDIA GPU supports the native energy counter.
                        self._collector_exception(
                            "nvml_energy_counter", exc, from_thread=True
                        )

        with self._sample_lock:
            self._samples.append(sample)

    def _synchronize_cuda_if_requested(self) -> None:
        if not (
            self.enabled
            and self.collect_gpu
            and self.synchronize_cuda
            and self._cuda_available
            and _torch is not None
            and self.gpu_id is not None
        ):
            return
        try:
            _torch.cuda.synchronize(self.gpu_id)
        except Exception as exc:
            self._collector_exception("cuda_synchronize", exc)

    def _reset_cuda_peaks(self) -> None:
        if not (
            self._cuda_available and _torch is not None and self.gpu_id is not None
        ):
            return
        try:
            _torch.cuda.reset_peak_memory_stats(self.gpu_id)
        except Exception as exc:
            self._collector_exception("cuda_reset_peak_memory", exc)

    def _collector_exception(
        self, key: str, exc: Exception, *, from_thread: bool = False
    ) -> None:
        self._collector_error(key, f"{type(exc).__name__}: {exc}", from_thread)

    def _collector_error(
        self, key: str, message: str, from_thread: bool = False
    ) -> None:
        if key not in self._error_keys:
            self._error_keys.add(key)
            self._errors.append(f"{key}: {message}")
        if self.strict:
            error = RuntimeError(f"Telemetry collector failed ({key}): {message}")
            if from_thread:
                self._fatal_error = error
            else:
                raise error

    def _shutdown_nvml(self) -> None:
        if not self._nvml_initialized or _pynvml is None:
            return
        try:
            _pynvml.nvmlShutdown()
        except Exception as exc:
            # Teardown must not hide already collected experiment metrics.
            if self.strict and self._fatal_error is None:
                self._fatal_error = RuntimeError(
                    f"Telemetry collector failed (nvml_shutdown): {exc}"
                )
            elif not self.strict:
                self._collector_exception("nvml_shutdown", exc)
        finally:
            self._nvml_initialized = False
            self._nvml_handle = None

    def _summarize(self, duration: float, end_time_utc: str) -> dict[str, Any]:
        with self._sample_lock:
            samples = [dict(sample) for sample in self._samples]

        metrics: dict[str, Any] = {
            "telemetry_enabled": self.enabled,
            "telemetry_strict": self.strict,
            "telemetry_sample_count": len(samples),
            "telemetry_error_count": len(self._errors),
            "telemetry_errors": " | ".join(self._errors),
            "process_id": os.getpid(),
            "gpu_device_index": self.gpu_id,
            "start_time_utc": self._start_time_utc,
            "end_time_utc": end_time_utc,
            "duration_seconds": duration,
            "psutil_available": _psutil is not None,
            "cuda_available": self._cuda_available,
            "nvml_available": self._nvml_available,
            "gpu_nvml_mapping_source": self._nvml_mapping_source,
            "gpu_nvml_device_index": self._nvml_device_index,
            "gpu_nvml_device_uuid": self._nvml_device_uuid,
        }

        self._add_series_summary(
            metrics,
            samples,
            "process_cpu_percent",
            mean_name="process_cpu_percent_mean",
            peak_name="process_cpu_percent_peak",
        )
        self._add_initial_end_peak(
            metrics, samples, "process_rss_bytes", "process_rss"
        )
        self._add_series_summary(
            metrics,
            samples,
            "process_tree_cpu_percent",
            mean_name="process_tree_cpu_percent_mean",
            peak_name="process_tree_cpu_percent_peak",
        )
        self._add_initial_end_peak(
            metrics, samples, "process_tree_rss_bytes", "process_tree_rss"
        )
        child_counts = self._values(samples, "process_child_count")
        metrics["process_child_count_peak"] = (
            int(max(child_counts)) if child_counts else None
        )
        self._add_initial_end_peak(
            metrics,
            samples,
            "cuda_memory_allocated_bytes",
            "cuda_memory_allocated",
        )
        self._add_initial_end_peak(
            metrics,
            samples,
            "cuda_memory_reserved_bytes",
            "cuda_memory_reserved",
        )
        self._add_initial_end_peak(
            metrics, samples, "gpu_memory_used_bytes", "gpu_memory_used"
        )

        # CUDA exposes allocator peaks directly, which may be higher than any
        # periodic sample. Prefer those counters for the final peak values.
        allocated_peaks = self._values(samples, "cuda_memory_allocated_peak_bytes")
        if allocated_peaks:
            metrics["cuda_memory_allocated_peak_bytes"] = max(allocated_peaks)
        reserved_peaks = self._values(samples, "cuda_memory_reserved_peak_bytes")
        if reserved_peaks:
            metrics["cuda_memory_reserved_peak_bytes"] = max(reserved_peaks)

        self._add_series_summary(
            metrics,
            samples,
            "gpu_utilization_percent",
            mean_name="gpu_utilization_percent_mean",
            peak_name="gpu_utilization_percent_peak",
        )
        self._add_series_summary(
            metrics,
            samples,
            "gpu_power_watts",
            mean_name="gpu_power_watts_mean",
            peak_name="gpu_power_watts_peak",
        )
        energy, source = self._calculate_energy(samples)
        metrics["gpu_energy_joules"] = energy
        metrics["gpu_energy_source"] = source

        if self.save_raw_samples:
            metrics["raw_samples"] = samples
        return metrics

    @staticmethod
    def _values(samples: list[dict[str, float]], key: str) -> list[float]:
        return [sample[key] for sample in samples if key in sample]

    @classmethod
    def _add_series_summary(
        cls,
        metrics: dict[str, Any],
        samples: list[dict[str, float]],
        sample_key: str,
        *,
        mean_name: str,
        peak_name: str,
    ) -> None:
        values = cls._values(samples, sample_key)
        metrics[mean_name] = _mean(values)
        metrics[peak_name] = _maximum(values)

    @classmethod
    def _add_initial_end_peak(
        cls,
        metrics: dict[str, Any],
        samples: list[dict[str, float]],
        sample_key: str,
        output_prefix: str,
    ) -> None:
        values = cls._values(samples, sample_key)
        metrics[f"{output_prefix}_initial_bytes"] = values[0] if values else None
        metrics[f"{output_prefix}_end_bytes"] = values[-1] if values else None
        metrics[f"{output_prefix}_peak_bytes"] = _maximum(values)

    @classmethod
    def _calculate_energy(
        cls, samples: list[dict[str, float]]
    ) -> tuple[float | None, str]:
        native = [
            sample
            for sample in samples
            if "gpu_total_energy_millijoules" in sample
        ]
        if len(native) >= 2:
            delta_millijoules = (
                native[-1]["gpu_total_energy_millijoules"]
                - native[0]["gpu_total_energy_millijoules"]
            )
            if delta_millijoules >= 0:
                return delta_millijoules / 1000.0, "nvml_total_energy_counter"

        power_samples = [
            sample
            for sample in samples
            if "gpu_power_watts" in sample and "elapsed_seconds" in sample
        ]
        if len(power_samples) < 2:
            return None, "unavailable"

        energy_joules = 0.0
        for previous, current in zip(power_samples, power_samples[1:]):
            elapsed = current["elapsed_seconds"] - previous["elapsed_seconds"]
            if elapsed > 0:
                average_power = (
                    previous["gpu_power_watts"] + current["gpu_power_watts"]
                ) / 2.0
                energy_joules += average_power * elapsed
        return energy_joules, "power_integration"


__all__ = ["ResourceMonitor"]
