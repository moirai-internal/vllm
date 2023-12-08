import time
from typing import Any, Awaitable, Callable, Dict

from aioprometheus import Histogram, MetricsMiddleware
from fastapi import Request

from vllm.logger import init_logger

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class OpcRequestIdLoggingMiddleware:
    def __init__(self, name: str):
        self.logger_name = name

    async def __call__(self, request: Request, call_next):
        logger = init_logger(self.logger_name)
        opc_request_id = request.headers.get("opc-request-id", "unknown")
        bound_logger = logger.bind(opc_request_id=opc_request_id)
        bound_logger.info("Starting request")
        try:
            response = await call_next(request)
            bound_logger.info("Finished request", status_code=response.status_code)
            return response
        except Exception as e:
            bound_logger.error("Exception during request", error=str(e))
            raise


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
