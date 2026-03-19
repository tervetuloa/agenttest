from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseInterceptor(ABC):
    """Base interface for framework interceptors."""

    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped multi-agent system."""
        raise NotImplementedError
