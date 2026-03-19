from __future__ import annotations

from typing import Any

from agenttest.interceptors.base import BaseInterceptor


class CrewAIInterceptor(BaseInterceptor):
    """Placeholder interceptor for future CrewAI integration."""

    def __init__(self, crew: Any):
        self.crew = crew

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("CrewAIInterceptor is not implemented in this MVP")
