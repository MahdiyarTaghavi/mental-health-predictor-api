from fastapi import Request
from utils.loggings.logging_context import request_context

async def log_request_context_middleware(request: Request, call_next):
    """
    Middleware to capture request-specific information (endpoint, method, client IP).
    This data is safely isolated per request using contextvars.
    """
    # Extract useful details
    context_data = {
        "endpoint": request.url.path,
        "method": request.method,
        "client_ip": request.client.host if request.client else None,
    }

    # Set in context variable
    request_context.set(context_data)

    # Proceed with request
    response = await call_next(request)
    return response