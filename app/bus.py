import asyncio
from typing import List


class EventBus:
    """In-process pub/sub. One publisher, many subscribers (SSE clients).

    Each subscriber gets its own bounded queue. Slow subscribers drop
    events rather than blocking the pipeline — the dashboard is
    best-effort, not a system of record.
    """

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def publish(self, event: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # drop for slow subscribers


bus = EventBus()
