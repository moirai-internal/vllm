import time

from prometheus_client import Histogram
from fastapi import Request

from vllm.logger import init_logger

logger = init_logger(__name__)


async def log_opc_header(request: Request, call_next):
    # Log at the start and end of a POST request
    if request.method == "POST":
        opc_request_id = request.headers.get("opc-request-id", "unknown")
        logger.info(f"POST Request Start - opc-request-id: {opc_request_id}",
                    extra={"opc-request-id": opc_request_id})

        try:
            response = await call_next(request)
            logger.info(
                f"POST Request End - opc-request-id: {opc_request_id}, "
                f"status_code: {response.status_code}",
                extra={"opc-request-id": opc_request_id})
            return response
        except Exception as e:
            logger.error(
                f"Exception during POST request with "
                f"opc-request-id: {opc_request_id}, error: {e}",
                extra={"opc-request-id": opc_request_id})
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
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])

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
        """ Return the full path or the template path if available. """
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
