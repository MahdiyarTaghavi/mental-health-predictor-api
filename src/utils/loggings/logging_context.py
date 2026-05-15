import contextvars

# Context variable that stores request info per async task
request_context = contextvars.ContextVar("request_context", default=None)