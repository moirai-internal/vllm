import time

from aioprometheus import Histogram
from fastapi import Request

from vllm.logger import init_logger

logger = init_logger(__name__)


async def log_opc_header(request: Request, call_next):
    # Log at the start and end of a POST request
    if request.method == "POST":
        opc_request_id = request.headers.get("opc-request-id", "unknown")
        logger.bind(opc_request_id=opc_request_id).info(
            f"POST Request Start - opc-request-id: {opc_request_id}")

        try:
            response = await call_next(request)
            logger.bind(opc_request_id=opc_request_id).info(
                f"POST Request End - opc-request-id: {opc_request_id}, "
                f"status_code: {response.status_code}")
            return response
        except Exception as e:
            logger.bind(opc_request_id=opc_request_id).error(
                f"Exception during POST request with "
                f"opc-request-id: {opc_request_id}, error: {e}")
            raise

    # For non-POST requests, just pass through
    return await call_next(request)


class LatencyMetricsMiddleware:

    def __init__(self, app):
        self.app = app
        # Create a Histogram metric for request latency
        self.request_latency = Histogram("request_latency_seconds",
                                         "Latency of requests in seconds")

    async def __call__(self, scope, receive, send):
        start_time = time.time()
        try:
            # Process the request
            await self.app(scope, receive, send)
        except Exception:
            # Re-raise the exception without recording latency
            raise
        else:
            # Calculate the time taken and record it in the Summary metric
            duration = time.time() - start_time
            labels = {
                "method": scope["method"],
                "path": self.get_full_or_template_path(scope)
            }
            self.request_latency.observe(value=duration, labels=labels)

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
