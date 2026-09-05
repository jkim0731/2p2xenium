"""Multiprocessing-worker resolution shared by every stage that parallelizes independent units of
work (grid-search candidates, tiles, sections). Two user-facing knobs (see `resolve_num_cpus`):

    num_cpus       None, 0, or a value exceeding this machine's usable CPU count -> "auto" (every
                   usable core); 1 -> no multiprocessing at all (run serially, in-process); any
                   other positive int <= the usable CPU count -> exactly that many worker processes
    reserve_cpus   how many cores to withhold from "usable" (default 0 -- usable == every physical
                   core). Set > 0 in a shared/orchestrated environment where saturating every
                   physical core with worker processes leaves nothing for the OS/orchestration
                   process itself (the thing that submitted this job, e.g. a Code Ocean
                   computation) -- on a fully-loaded machine that can silently kill the run partway
                   through with no traceback (observed during development of the
                   ophys-xenium-autocoreg capsule).

An explicit `num_cpus` at or below the usable count (`os.cpu_count() - reserve_cpus`, floored at 1)
is never reduced further -- only a request that would hit or exceed the machine's usable count gets
capped back down to it.
"""
import os
from concurrent.futures import ProcessPoolExecutor

DEFAULT_RESERVE_CPUS = 0


def resolve_num_cpus(num_cpus, reserve_cpus=DEFAULT_RESERVE_CPUS):
    """Resolve a user-requested worker count against this machine's availability. Returns `None`
    (meaning: run serially, no `ProcessPoolExecutor` at all -- not even a 1-worker pool, which
    would still pay fork/pickle overhead for zero parallelism gain) or a positive int >= 2 (the
    worker count to hand `ProcessPoolExecutor`). See the module docstring for the fallback rules
    and what `reserve_cpus` is for."""
    if num_cpus is not None and num_cpus < 0:
        raise ValueError(f"num_cpus must be >= 0 (0 or blank = auto), got {num_cpus}")
    if reserve_cpus < 0:
        raise ValueError(f"reserve_cpus must be >= 0, got {reserve_cpus}")
    available = os.cpu_count() or 1
    usable = max(1, available - reserve_cpus)
    if num_cpus is None or num_cpus == 0 or num_cpus > usable:
        num_cpus = usable
    return None if num_cpus == 1 else num_cpus


def pool_map(fn, items, num_cpus, reserve_cpus=DEFAULT_RESERVE_CPUS, initializer=None, initargs=(),
            chunksize=1):
    """Map `fn` over `items`, either serially (in-process) or via a `ProcessPoolExecutor`,
    depending on `resolve_num_cpus(num_cpus, reserve_cpus)`. `initializer`/`initargs` (the
    read-only fork-shared-array pattern several stages use, so large arrays are pickled once per
    worker rather than once per task) are called once directly, in-process, for the serial path --
    exactly what they'd do inside a single worker process, and with the identical order-preserving
    semantics as `ProcessPoolExecutor.map`."""
    workers = resolve_num_cpus(num_cpus, reserve_cpus=reserve_cpus)
    if workers is None:
        if initializer is not None:
            initializer(*initargs)
        return list(map(fn, items))
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))
