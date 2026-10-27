"""Bounded day-level concurrency with conservative adaptive backoff."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os


def transient(error):
    message = repr(error).lower()
    return any(s in message for s in ('429', 'rate limit', 'ratelimit', 'rate_limit', 'overload',
        '503', '502', '504', 'timed out', 'timeout', 'connection reset', 'resource_exhausted', 'resource pressure'))


def quota(error):
    return any(s in repr(error).lower() for s in ('usage limit', 'quota exceeded', 'spending limit', 'insufficient credits'))


@dataclass
class AdaptiveConcurrency:
    workers: int = 2
    stable_days: int = 0
    maximum: int = 4

    def failure(self):
        self.workers = max(1, self.workers // 2)
        self.stable_days = 0

    def success(self, pressure=False):
        if pressure:
            self.failure()
            return
        self.stable_days += 1
        if self.stable_days >= 3 and self.workers < self.maximum:
            self.workers += 1
            self.stable_days = 0


def host_pressure(backend=None):
    # Cursor inference runs remotely. Local CPU load is not provider capacity.
    # Actual request failures still trigger AdaptiveConcurrency.failure().
    if backend == 'cursor':
        return False
    return os.getloadavg()[0] > (os.cpu_count() or 1) * .9


def run_batch(ids, work, workers):
    # All in-flight tasks finish before any retry or clock advance.
    errors = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='trading-agent') as pool:
        pending = {pool.submit(work, aid): aid for aid in ids}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                errors[pending[future]] = error
    return errors
