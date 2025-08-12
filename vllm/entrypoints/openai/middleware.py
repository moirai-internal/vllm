# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import Request

from vllm.logger import init_logger

logger = init_logger(__name__)


async def log_opc_header(request: Request, call_next):
    # Log at the start and end of a POST request
    if request.method == "POST":
        opc_request_id = request.headers.get("opc-request-id", "unknown")
        logger.info("POST Request Start - opc-request-id: %s", opc_request_id)

        try:
            response = await call_next(request)
            logger.info(
                "POST Request End - opc-request-id: %s, "
                "status_code: %d", opc_request_id, response.status_code)
            return response
        except Exception as e:
            logger.error(
                "Exception during POST request with "
                "opc-request-id: %s, error: %s", opc_request_id, e)
            raise

    # For non-POST requests, just pass through
    return await call_next(request)
