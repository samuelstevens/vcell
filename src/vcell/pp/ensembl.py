import concurrent.futures
import typing as tp

import beartype


@beartype.beartype
class EnsemblExecutor(concurrent.futures.Executor):
    def __init__(
        self,
        max_workers: int = 8,
        rate: float = 15.0,  # requests per second
        burst: int | None = None,  # token bucket capacity; default = ceil(rate)
        response_adapter: tp.Callable[[tp.Any], dict[str, str]] | None = None,
    ):
        """Thread-pool executor with rate limiting."""

    def submit(
        self, fn: tp.Callable[..., tp.Any], /, *args: tp.Any, **kwargs: tp.Any
    ) -> concurrent.futures.Future:
        """Schedule a call; returns a Future. Respects rate+burst and max_workers."""

    def set_rate(self, rps: float, burst: int | None = None) -> None:
        """Manually adjust RPS/burst."""

    def stats(self) -> dict[str, tp.Any]:
        """Return current tokens, effective rps, in_flight, queued, next_permit_at, per-bucket info."""

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None: ...
