import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import Request
from loguru import logger as loguru_logger
from prometheus_client import Histogram

from vllm.logger import init_logger

# Create a context variable to store the request ID
request_id_ctx_var = ContextVar("request_id", default="unknown")

logger = init_logger(__name__)


@asynccontextmanager
async def request_id_context(request_id: str):
    """Context manager to bind request ID to all loggers within the context."""
    # Bind the request ID at context entry
    token = request_id_ctx_var.set(request_id)
    with loguru_logger.contextualize(opc_request_id=request_id):
        try:
            yield
        finally:
            # Reset the context variable
            request_id_ctx_var.reset(token)


async def log_opc_header(request: Request, call_next):
    # Log at the start and end of a POST request
    if request.method == "POST":
        # Get the request ID from header or generate a default
        opc_request_id = request.headers.get("opc-request-id", "unknown")

        async with request_id_context(opc_request_id):
            logger.info(
                "POST Request Start - opc-request-id: %s",
                opc_request_id,
            )

            try:
                response = await call_next(request)
                logger.info(
                    "POST Request End - opc-request-id: %s, status_code: %s",
                    opc_request_id,
                    response.status_code,
                )
                return response
            except Exception as e:
                logger.error(
                    "Exception during POST request with opc-request-id: %s, error: %s",  # noqa: E501
                    opc_request_id,
                    str(e),
                )
                raise

    # For non-POST requests, just pass through
    return await call_next(request)


class LatencyMetricsMiddleware:

    def __init__(self, app):
        self.app = app
        self.request_latency = Histogram(
            "http_request_latency_seconds",
            "Latency of HTTP requests in seconds",
            ["method", "path"],  # Define the labels
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )

    async def __call__(self, scope, receive, send):
        start_time = time.time()
        try:
            # Process the request
            await self.app(scope, receive, send)
        except Exception:
            # Re-raise the exception without recording latency
            raise
        else:
            duration = time.time() - start_time
            method = scope["method"]
            path = self.get_full_or_template_path(scope)
            self.request_latency.labels(method=method,
                                        path=path).observe(duration)

    def get_full_or_template_path(self, scope):
        """Return the full path or the template path if available."""
        root_path = scope.get("root_path", "")
        path = scope.get("path", "")
        full_path = f"{root_path}{path}"

        routes = scope.get("app").routes

        for route in routes:
            match, _child_scope = route.matches(scope)
            # Enum value 2 represents the route template Match.FULL
            if match.value == 2:
                return route.path
        return full_path
