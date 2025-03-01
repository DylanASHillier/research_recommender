"""Miscellaneous utilities for multiplayer games."""

import asyncio


class AsyncRateLimiter:
    """Async context manager for rate limiting.

    Similar to external libraries like `aiolimiter`
    Example usage:
    ```python
    rate_limiter = AsyncRateLimiter(rate=10)  # 10 Hz
    while True:
        async with rate_limiter:
            # Your code here
    ```
    """

    def __init__(self, rate: float):
        self.period = 1 / rate
        self.loop = asyncio.get_event_loop()
        self.start_time = ...

    async def __aenter__(self):
        self.start_time = self.loop.time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        elapsed = self.loop.time() - self.start_time
        await asyncio.sleep(max(0, self.period - elapsed))
