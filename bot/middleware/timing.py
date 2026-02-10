"""Response-time logging for every pipeline step.

Usage inside handlers:

    timer = Timer(update)
    ...
    timer.checkpoint("url_fetch")       # logs elapsed since last checkpoint
    ...
    timer.checkpoint("perplexity_api")
    ...
    timer.done()                        # logs total wall-clock time
"""

import time
import logging

logger = logging.getLogger(__name__)


class Timer:
    """Lightweight per-request stopwatch with named checkpoints."""

    __slots__ = ("_start", "_last", "_update_id", "_checkpoints")

    def __init__(self, update):
        now = time.monotonic()
        self._start = now
        self._last = now
        self._update_id = update.update_id if update else 0
        self._checkpoints: list[tuple[str, float]] = []

    def checkpoint(self, name: str) -> float:
        """Record a named checkpoint.  Returns ms since previous checkpoint."""
        now = time.monotonic()
        step_ms = (now - self._last) * 1000
        self._checkpoints.append((name, step_ms))
        self._last = now
        return step_ms

    def done(self) -> float:
        """Finalise and log the full breakdown.  Returns total ms."""
        total_ms = (time.monotonic() - self._start) * 1000
        parts = " | ".join(f"{name}={ms:.0f}ms" for name, ms in self._checkpoints)
        logger.info(
            f"[perf] update={self._update_id} total={total_ms:.0f}ms  {parts}"
        )
        return total_ms
