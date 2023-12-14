import time
from typing import Any, Awaitable, Callable, Dict

from aioprometheus import Histogram, MetricsMiddleware

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class ExtendedMetricsMiddleware(MetricsMiddleware):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_latency = Histogram(
            "request_latency_seconds",
            "Latency of HTTP requests in seconds",
            const_labels=self.const_labels,
            registry=self.registry
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        start_time = time.time()
        try:
            await super().__call__(scope, receive, send)
        finally:
            duration = time.time() - start_time
            labels = {
                "method": scope["method"],
                "path": self.get_full_or_template_path(scope)
            }

            self.request_latency.observe(value=duration, labels=labels)
