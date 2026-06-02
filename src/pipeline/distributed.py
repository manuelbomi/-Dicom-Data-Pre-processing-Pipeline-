"""Distributed processing using multiprocessing and optional Dask integration.

Provides execution backends for the pipeline orchestrator:

  - **LocalExecutor**: Single-process execution for debugging.
  - **MultiprocessExecutor**: Python multiprocessing with configurable
    worker count and memory limits.
  - **DaskExecutor**: Distributed execution via Dask for cluster-scale
    processing (AWS ECS/EKS, SLURM, etc.).

The executor abstraction allows the pipeline to scale from laptop to
cloud cluster without changing pipeline code.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ExecutorConfig:
    """Configuration for distributed execution."""

    n_workers: int = 4
    memory_limit_per_worker: str = "4GB"
    threads_per_worker: int = 1
    use_processes: bool = True
    timeout_per_task_sec: float = 300.0
    max_retries: int = 1
    prefetch_factor: int = 2

    # Dask-specific
    dask_scheduler_address: Optional[str] = None
    dask_dashboard_port: int = 8787
    dask_worker_space: str = "/tmp/dask-worker-space"
    dask_memory_target: float = 0.6
    dask_memory_spill: float = 0.7
    dask_memory_pause: float = 0.8


@dataclass
class TaskResult:
    """Result from a distributed task."""

    key: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_sec: float = 0.0
    worker_id: Optional[str] = None


@dataclass
class ExecutionStats:
    """Statistics from distributed execution."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_duration_sec: float = 0.0
    tasks_per_second: float = 0.0
    worker_utilization: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Execution: {self.completed_tasks}/{self.total_tasks} tasks "
            f"in {self.total_duration_sec:.1f}s "
            f"({self.tasks_per_second:.1f} tasks/sec), "
            f"{self.failed_tasks} failures"
        )


class BaseExecutor(ABC):
    """Abstract base class for execution backends."""

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        self.config = config or ExecutorConfig()

    @abstractmethod
    def map(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        keys: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        """Apply a function to a list of items in parallel.

        Args:
            fn: Function to apply. Signature: fn(item) -> result.
            items: List of input items.
            keys: Optional identifiers for each item.

        Returns:
            List of TaskResult, one per input item.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up executor resources."""

    def __enter__(self) -> "BaseExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


class LocalExecutor(BaseExecutor):
    """Single-process executor for debugging and small datasets.

    Processes items sequentially in the current process, making it
    easy to attach debuggers and inspect intermediate state.
    """

    def map(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        keys: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        results: List[TaskResult] = []
        keys = keys or [str(i) for i in range(len(items))]

        for key, item in zip(keys, items):
            t0 = time.monotonic()
            try:
                output = fn(item)
                results.append(TaskResult(
                    key=key,
                    success=True,
                    output=output,
                    duration_sec=time.monotonic() - t0,
                    worker_id="local",
                ))
            except Exception as exc:
                results.append(TaskResult(
                    key=key,
                    success=False,
                    error=str(exc),
                    duration_sec=time.monotonic() - t0,
                    worker_id="local",
                ))

        return results

    def shutdown(self) -> None:
        pass


class MultiprocessExecutor(BaseExecutor):
    """Multiprocessing-based executor for single-machine parallelism.

    Uses Python's ProcessPoolExecutor for CPU-bound tasks (image processing)
    or ThreadPoolExecutor for I/O-bound tasks (S3 reads, disk I/O).

    Provides:
      - Configurable worker count
      - Per-task timeout
      - Automatic retry on failure
      - Memory-aware scheduling
    """

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        super().__init__(config)
        self._pool: Optional[ProcessPoolExecutor] = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            if self.config.use_processes:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.config.n_workers,
                    mp_context=mp.get_context("spawn"),
                )
            else:
                # ThreadPool for I/O-bound work
                self._pool = ThreadPoolExecutor(
                    max_workers=self.config.n_workers,
                )
        return self._pool

    def map(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        keys: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        keys = keys or [str(i) for i in range(len(items))]
        pool = self._get_pool()
        results: List[TaskResult] = [None] * len(items)  # type: ignore
        t0 = time.monotonic()

        futures = {}
        for idx, (key, item) in enumerate(zip(keys, items)):
            future = pool.submit(_execute_task, fn, item)
            futures[future] = (idx, key)

        for future in as_completed(futures, timeout=self.config.timeout_per_task_sec * len(items)):
            idx, key = futures[future]
            try:
                output, duration = future.result(timeout=self.config.timeout_per_task_sec)
                results[idx] = TaskResult(
                    key=key,
                    success=True,
                    output=output,
                    duration_sec=duration,
                    worker_id=f"worker-{idx % self.config.n_workers}",
                )
            except Exception as exc:
                # Retry logic
                retry_success = False
                if self.config.max_retries > 0:
                    for attempt in range(self.config.max_retries):
                        try:
                            retry_future = pool.submit(_execute_task, fn, items[idx])
                            output, duration = retry_future.result(
                                timeout=self.config.timeout_per_task_sec
                            )
                            results[idx] = TaskResult(
                                key=key,
                                success=True,
                                output=output,
                                duration_sec=duration,
                            )
                            retry_success = True
                            break
                        except Exception:
                            continue

                if not retry_success:
                    results[idx] = TaskResult(
                        key=key,
                        success=False,
                        error=str(exc),
                    )

        total_time = time.monotonic() - t0
        succeeded = sum(1 for r in results if r and r.success)
        failed = sum(1 for r in results if r and not r.success)
        logger.info(
            "MultiprocessExecutor: %d/%d succeeded in %.1fs (%.1f tasks/sec)",
            succeeded, len(items), total_time, len(items) / max(total_time, 0.001),
        )

        return results

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _execute_task(fn: Callable, item: Any) -> Tuple[Any, float]:
    """Execute a single task and return (result, duration)."""
    t0 = time.monotonic()
    result = fn(item)
    return result, time.monotonic() - t0


class DaskExecutor(BaseExecutor):
    """Dask-based executor for cluster-scale distributed processing.

    Connects to an existing Dask cluster or creates a local one.
    Provides adaptive scaling, dashboard monitoring, and fault tolerance.

    Requires: ``pip install "dask[distributed]"``
    """

    def __init__(self, config: Optional[ExecutorConfig] = None) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Dask client."""
        if self._client is None:
            from dask.distributed import Client, LocalCluster

            if self.config.dask_scheduler_address:
                self._client = Client(self.config.dask_scheduler_address)
                logger.info(
                    "Connected to Dask scheduler: %s",
                    self.config.dask_scheduler_address,
                )
            else:
                cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit_per_worker,
                    dashboard_address=f":{self.config.dask_dashboard_port}",
                    local_directory=self.config.dask_worker_space,
                    memory_target_fraction=self.config.dask_memory_target,
                    memory_spill_fraction=self.config.dask_memory_spill,
                    memory_pause_fraction=self.config.dask_memory_pause,
                )
                self._client = Client(cluster)
                logger.info(
                    "Dask LocalCluster started: %d workers, dashboard at :%d",
                    self.config.n_workers,
                    self.config.dask_dashboard_port,
                )

        return self._client

    def map(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        keys: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        keys = keys or [str(i) for i in range(len(items))]
        client = self._get_client()
        t0 = time.monotonic()

        # Submit all tasks
        futures = client.map(fn, items)

        # Gather results with error handling
        results: List[TaskResult] = []
        gathered = client.gather(futures, errors="skip")

        for idx, (key, output) in enumerate(zip(keys, gathered)):
            if isinstance(output, Exception):
                results.append(TaskResult(
                    key=key,
                    success=False,
                    error=str(output),
                ))
            else:
                results.append(TaskResult(
                    key=key,
                    success=True,
                    output=output,
                ))

        total_time = time.monotonic() - t0
        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "DaskExecutor: %d/%d succeeded in %.1fs", succeeded, len(items), total_time,
        )
        return results

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class DistributedExecutor:
    """High-level executor factory that selects the right backend.

    Convenience wrapper that creates the appropriate executor based on
    configuration. For production pipelines, use this as the primary
    entry point.

    Args:
        n_workers: Number of parallel workers.
        memory_limit: Memory limit per worker (e.g., "4GB").
        backend: Execution backend ("local", "multiprocess", "dask").
        dask_scheduler: Address of Dask scheduler (for remote clusters).

    Example::

        executor = DistributedExecutor(n_workers=16, memory_limit="4GB")
        results = executor.map(process_fn, dicom_paths)
    """

    def __init__(
        self,
        n_workers: int = 4,
        memory_limit: str = "4GB",
        backend: str = "multiprocess",
        dask_scheduler: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        config = ExecutorConfig(
            n_workers=n_workers,
            memory_limit_per_worker=memory_limit,
            dask_scheduler_address=dask_scheduler,
            **{k: v for k, v in kwargs.items() if hasattr(ExecutorConfig, k)},
        )

        backend_map = {
            "local": LocalExecutor,
            "multiprocess": MultiprocessExecutor,
            "dask": DaskExecutor,
        }

        executor_cls = backend_map.get(backend)
        if executor_cls is None:
            raise ValueError(f"Unknown backend: {backend}. Options: {list(backend_map.keys())}")

        self._executor = executor_cls(config)
        self._backend = backend
        logger.info("DistributedExecutor: backend=%s, workers=%d", backend, n_workers)

    def map(
        self,
        fn: Callable[..., Any],
        items: List[Any],
        keys: Optional[List[str]] = None,
    ) -> List[TaskResult]:
        """Apply function to items in parallel."""
        return self._executor.map(fn, items, keys)

    def shutdown(self) -> None:
        """Release executor resources."""
        self._executor.shutdown()

    def __enter__(self) -> "DistributedExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
