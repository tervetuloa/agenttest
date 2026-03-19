from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def mock_agent(response: Any = None, side_effect: Exception | None = None) -> Callable[[F], F]:
    """
    Decorator that replaces an agent function with deterministic behavior.

    Args:
        response: Static return value to use instead of running the wrapped function
        side_effect: Optional exception to raise when the function is called
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if side_effect is not None:
                raise side_effect
            if response is not None:
                return response
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
