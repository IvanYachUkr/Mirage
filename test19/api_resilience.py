"""Shared API resilience utilities for Gemini calls.

Policy:
- hard timeout per call (default 70s)
- retry on 503/unavailable/rate-limit and timeout/hang
- bounded exponential backoff with jitter
- lightweight structured stats for reporting
"""

from __future__ import annotations

import concurrent.futures
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


_RETRIABLE_TOKENS = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINEEXCEEDED",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
    "TIMED OUT",
    "SERVICE UNAVAILABLE",
)


@dataclass
class ApiCallStats:
    attempts: int = 0
    retries: int = 0
    timeout_retries: int = 0
    service_retries: int = 0
    other_retries: int = 0
    sleep_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ApiCallError(RuntimeError):
    """Raised when all retry attempts are exhausted."""



def is_retriable_error(exc: BaseException) -> bool:
    msg = str(exc).upper()
    return any(tok in msg for tok in _RETRIABLE_TOKENS)



def _call_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        result = fut.result(timeout=timeout_sec)
        pool.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError as exc:
        # Do NOT wait=True — the hung HTTP thread may never return.
        # cancel_futures=True added in Python 3.9; safe to call regardless.
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
        raise TimeoutError(f"API call exceeded {timeout_sec:.0f}s") from exc



def generate_content_resilient(
    *,
    client,
    model: str,
    contents: Any,
    config: Any,
    timeout_sec: float = 70.0,
    max_attempts: int = 5,
    base_delay_sec: float = 4.0,
    max_delay_sec: float = 35.0,
    jitter_ratio: float = 0.25,
    rng_seed: Optional[int] = None,
    on_retry: Optional[Callable[[int, int, BaseException, float], None]] = None,
) -> tuple[Any, ApiCallStats]:
    """Robust Gemini content generation with timeout + retries.

    Returns (response, stats). Raises ApiCallError after exhausting attempts.
    """
    stats = ApiCallStats()
    rng = random.Random(rng_seed)
    last_err: Optional[BaseException] = None

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        stats.attempts += 1
        try:
            resp = _call_with_timeout(
                lambda: client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                timeout_sec=timeout_sec,
            )
            return resp, stats
        except BaseException as exc:  # noqa: BLE001
            last_err = exc
            stats.errors.append(str(exc))

            if isinstance(exc, TimeoutError):
                stats.timeout_retries += 1
                retriable = True
            else:
                retriable = is_retriable_error(exc)
                if retriable:
                    stats.service_retries += 1
                else:
                    stats.other_retries += 1

            if (not retriable) or attempt >= max_attempts:
                break

            stats.retries += 1
            delay = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
            jitter = delay * jitter_ratio * rng.random()
            sleep_for = delay + jitter
            stats.sleep_seconds += sleep_for
            if on_retry is not None:
                on_retry(attempt, max_attempts, exc, sleep_for)
            time.sleep(sleep_for)

    raise ApiCallError(
        f"API call failed after {stats.attempts} attempts: {last_err}"
    ) from last_err



def usage_tokens_and_cost(
    response: Any,
    *,
    input_per_million: float,
    output_per_million: float,
) -> tuple[int, int, float]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, 0.0
    inp = int(getattr(usage, "prompt_token_count", 0) or 0)
    out = int(getattr(usage, "candidates_token_count", 0) or 0)
    cost = inp / 1_000_000 * input_per_million + out / 1_000_000 * output_per_million
    return inp, out, cost
