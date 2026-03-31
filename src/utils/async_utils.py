"""Async utility classes for concurrent operations.

This module provides utilities for managing async operations,
including rate limiting and concurrency control.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("src.utils.async_utils")


class AsyncRateLimiter:
    """Async rate limiter using token bucket algorithm.

    Limits the rate of async operations to avoid exceeding API rate limits.
    Uses a token bucket algorithm that allows for burst capacity while
    maintaining an average rate limit.

    Attributes:
        rate: Maximum requests per second
        max_tokens: Maximum tokens in the bucket (burst capacity)
        tokens: Current available tokens
        last_update: Time of last token update
        lock: Async lock for thread safety
    """

    def __init__(self, rate: float, max_tokens: Optional[float] = None) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Maximum requests per second (0 or negative = unlimited)
            max_tokens: Maximum burst capacity (defaults to rate)
        """
        self.rate: float = rate
        self.max_tokens: float = (
            max_tokens if max_tokens is not None else max(rate, 1.0)
        )
        self.tokens: float = self.max_tokens
        self.last_update: float = time.monotonic()
        self.lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        """Acquire n tokens, waiting if necessary.

        This method blocks until the requested number of tokens are available.
        Tokens are replenished based on elapsed time since the last update.

        Args:
            n: Number of tokens to acquire (default 1). If 0, returns immediately.
        """
        if self.rate <= 0 or n <= 0:
            return  # No rate limiting or nothing to acquire

        async with self.lock:
            now: float = time.monotonic()
            # Add tokens based on elapsed time
            elapsed: float = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < n:
                # Wait for enough tokens to become available
                wait_time: float = (n - self.tokens) / self.rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {n} tokens")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= n

    def reset(self) -> None:
        """Reset the rate limiter to full capacity.

        This is useful when starting a new batch of operations.
        """
        self.tokens = self.max_tokens
        self.last_update = time.monotonic()


class AsyncSemaphoreContext:
    """Context manager combining semaphore and rate limiter.

    This class provides a convenient way to limit both concurrency
    and rate of async operations.

    Attributes:
        semaphore: Asyncio semaphore for concurrency control
        rate_limiter: Optional rate limiter for rate control
    """

    def __init__(
        self,
        max_concurrency: int,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            max_concurrency: Maximum number of concurrent operations
            rate_limiter: Optional rate limiter instance
        """
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter: Optional[AsyncRateLimiter] = rate_limiter

    async def __aenter__(self) -> "AsyncSemaphoreContext":
        """Enter the context, acquiring semaphore and rate limit tokens."""
        await self.semaphore.acquire()
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, releasing the semaphore."""
        self.semaphore.release()
