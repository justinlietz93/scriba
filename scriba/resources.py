"""Resource limits for low-end machines."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import TranscribeError


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_resource_limits(raw_threads: str | None) -> int:
    if raw_threads in (None, ""):
        threads = 2
        override_existing = False
    else:
        try:
            threads = int(str(raw_threads).strip())
        except ValueError as exc:
            raise TranscribeError(f"invalid thread count: {raw_threads}") from exc
        if threads < 1:
            raise TranscribeError("thread count must be at least 1")
        override_existing = True

    value = str(threads)
    os.environ["SCRIBA_THREADS"] = value
    for name in THREAD_ENV_VARS:
        if override_existing:
            os.environ[name] = value
        else:
            os.environ.setdefault(name, value)
    apply_cpu_affinity(threads)
    return threads


def configured_threads() -> int:
    try:
        return max(1, int(os.environ.get("SCRIBA_THREADS", "2")))
    except ValueError:
        return 1


def apply_cpu_affinity(threads: int) -> None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return
    try:
        current_cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return

    if not current_cpus:
        return
    target_cpus = set(current_cpus[: max(1, min(threads, len(current_cpus)))])

    task_root = Path("/proc/self/task")
    tids: list[int] = []
    if task_root.exists():
        for task_dir in task_root.iterdir():
            if task_dir.name.isdigit():
                tids.append(int(task_dir.name))
    if not tids:
        tids = [0]

    for tid in tids:
        try:
            os.sched_setaffinity(tid, target_cpus)
        except OSError:
            continue
