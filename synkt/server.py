"""Real-time trace streaming server for synkt UI.

Streams test execution data via Server-Sent Events (SSE).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="synkt Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory trace queue shared between endpoints
trace_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()


@app.get("/stream")
async def stream_traces() -> EventSourceResponse:
    """SSE endpoint that streams trace snapshots to the UI."""

    async def event_generator():  # type: ignore[no-untyped-def]
        while True:
            trace = await trace_queue.get()
            yield {"event": "trace", "data": json.dumps(trace)}

    return EventSourceResponse(event_generator())


@app.post("/trace")
async def receive_trace(trace: dict[str, Any]) -> dict[str, str]:
    """Receive a trace snapshot from test execution."""
    await trace_queue.put(trace)
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check."""
    return {"status": "healthy", "queue_size": trace_queue.qsize()}


if __name__ == "__main__":
    import uvicorn

    print("synkt server starting on http://localhost:8000")
    print("Open http://localhost:3000 to view dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8000)
