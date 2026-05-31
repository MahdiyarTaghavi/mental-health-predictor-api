from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as aioredis


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, redis_host: str, redis_port: int, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis = aioredis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        redis_key = f"rate_limit:{client_ip}"

        # Increment the request count for the IP
        request_count = await self.redis.incr(redis_key)

        if request_count == 1:
            # Set expiration for the key on first increment
            await self.redis.expire(redis_key, self.window_seconds)

        if request_count > self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        response = await call_next(request)
        return response
