"""Multiprocessing-worker resolution shared by every stage that parallelizes independent units of
work (grid-search candidates, tiles, sections). One user-facing knob, `num_cpus`, with 3 fallback
rules (see `resolve_num_cpus`):
    None, 0, or a value exceeding the machine's CPU count -> "auto" (use every available core)
    1                                                      -> no multiprocessing at all (run
                                                              serially, in-process)
    any other positive int <= the machine's CPU count      -> exactly that many worker processes
"""
import os
from concurrent.futures import ProcessPoolExecutor


def resolve_num_cpus(num_cpus):
    """Resolve a user-requested worker count against this machine's availability. Returns `None`
    (meaning: run serially, no `ProcessPoolExecutor` at all -- not even a 1-worker pool, which
    would still pay fork/pickle overhead for zero parallelism gain) or a positive int >= 2 (the
    worker count to hand `ProcessPoolExecutor`). See the module docstring for the fallback rules."""
    if num_cpus is not None and num_cpus < 0:
        raise ValueError(f"num_cpus must be >= 0 (0 or blank = auto), got {num_cpus}")
    available = os.cpu_count() or 1
    if num_cpus is None or num_cpus == 0 or num_cpus > available:
        num_cpus = available
    return None if num_cpus == 1 else num_cpus


def pool_map(fn, items, num_cpus, initializer=None, initargs=(), chunksize=1):
    """Map `fn` over `items`, either serially (in-process) or via a `ProcessPoolExecutor`,
    depending on `resolve_num_cpus(num_cpus)`. `initializer`/`initargs` (the read-only
    fork-shared-array pattern several stages use, so large arrays are pickled once per worker
    rather than once per task) are called once directly, in-process, for the serial path --
    exactly what they'd do inside a single worker process, and with the identical order-preserving
    semantics as `ProcessPoolExecutor.map`."""
    workers = resolve_num_cpus(num_cpus)
    if workers is None:
        if initializer is not None:
            initializer(*initargs)
        return list(map(fn, items))
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))
