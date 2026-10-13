"""Small standard-library compatibility helpers for Python 3.7 and newer."""
import asyncio
import contextvars
import functools


async def to_thread(func, *args, **kwargs):
    """Run blocking work without blocking the event loop, preserving context.

    asyncio.to_thread was added in Python 3.9. This equivalent uses the
    executor and contextvars APIs already available in Python 3.7.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, call)
