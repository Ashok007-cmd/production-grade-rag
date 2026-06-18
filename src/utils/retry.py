"""Retry helpers shared across LLM-calling components."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exception types that should never be retried — they indicate programming
# errors or unrecoverable state, not transient infrastructure issues.
_NEVER_RETRY: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    ImportError,
    FileNotFoundError,
    NotImplementedError,
)


def retry_with_backoff(
    fn: Callable[[], T],
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
) -> T:
    """Execute ``fn`` with exponential backoff on transient failures.

    Re-raises immediately (without retrying) for ``ValueError``, ``TypeError``,
    ``ImportError``, ``FileNotFoundError``, and similar programming errors that
    are not transient. Retries up to ``retries`` times for everything else,
    sleeping ``backoff_in_seconds * 2**attempt`` between attempts.
    """
    x = 0
    while True:
        try:
            return fn()
        except _NEVER_RETRY:
            raise
        except Exception as e:
            if x == retries:
                logger.error("Failed after %d retries: %s", retries, e)
                raise
            sleep = backoff_in_seconds * (2**x)
            logger.warning(
                "Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                x + 1,
                retries,
                e,
                sleep,
            )
            time.sleep(sleep)
            x += 1


async def async_retry_with_backoff(
    fn: Callable[[], Coroutine[Any, Any, T]],
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
) -> T:
    """Execute async ``fn`` with exponential backoff on transient failures.

    Same retry semantics as ``retry_with_backoff`` but for coroutines.
    """
    x = 0
    while True:
        try:
            return await fn()
        except _NEVER_RETRY:
            raise
        except Exception as e:
            if x == retries:
                logger.error("Failed after %d retries: %s", retries, e)
                raise
            sleep = backoff_in_seconds * (2**x)
            logger.warning(
                "Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                x + 1,
                retries,
                e,
                sleep,
            )
            await asyncio.sleep(sleep)
            x += 1
