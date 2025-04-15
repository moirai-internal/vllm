import time

from fastapi import Request
from vllm.logger import init_logger

logger = init_logger(__name__)


async def log_opc_header(request: Request, call_next):
    # Log at the start and end of a POST request
    if request.method == "POST":
        opc_request_id = request.headers.get("opc-request-id", "unknown")
        logger.info(f"POST Request Start - opc-request-id: {opc_request_id}")

        try:
            response = await call_next(request)
            logger.info(
                f"POST Request End - opc-request-id: {opc_request_id}, "
                f"status_code: {response.status_code}")
            return response
        except Exception as e:
            logger.error(
                f"Exception during POST request with "
                f"opc-request-id: {opc_request_id}, error: {e}")
            raise

    # For non-POST requests, just pass through
    return await call_next(request)
