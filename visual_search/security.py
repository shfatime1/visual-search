import hashlib
import hmac
import ipaddress
import secrets
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse
from .config import MAX_IMAGE_BYTES


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        limit = MAX_IMAGE_BYTES + 65536
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > limit:
                return await JSONResponse({"detail": "Request exceeds the 5 MiB image limit."}, status_code=413)(scope, receive, send)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


class LocalSessions:
    def __init__(self, settings):
        self.settings = settings
        self.secret = secrets.token_bytes(32)
        self.requests = defaultdict(deque)

    def sign(self, value):
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def require_local(self, request: Request):
        try:
            is_local = ipaddress.ip_address(request.client.host).is_loopback
        except (ValueError, AttributeError):
            is_local = False
        expected = urlsplit(self.settings.ui_origin)
        if not is_local or request.headers.get("host") != expected.netloc or request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "The browser UI is available only on the configured local origin.")

    def issue(self, request):
        self.require_local(request)
        body = f"{int(time.time()) + 3600}.{secrets.token_hex(24)}"
        cookie = body + "." + self.sign(body)
        return cookie, self.sign("csrf:" + cookie)

    def authorize(self, request):
        self.require_local(request)
        if request.headers.get("origin") != self.settings.ui_origin:
            raise HTTPException(403, "Invalid browser origin.")
        cookie = request.cookies.get("visual_session", "")
        try:
            body, signature = cookie.rsplit(".", 1)
            expiry = int(body.split(".")[0])
            valid = hmac.compare_digest(self.sign(body), signature) and expiry > time.time()
        except ValueError:
            valid = False
        csrf = request.headers.get("x-csrf-token", "")
        if not valid or not hmac.compare_digest(csrf, self.sign("csrf:" + cookie)):
            raise HTTPException(403, "Browser session expired. Reload the page.")

    def rate_limit(self, request):
        now = time.monotonic()
        # Bounded local memory; stale client entries are discarded.
        for key in list(self.requests):
            if not self.requests[key] or self.requests[key][-1] < now - 60:
                del self.requests[key]
        key = request.client.host if request.client else "unknown"
        queue = self.requests[key]
        while queue and queue[0] < now - 60:
            queue.popleft()
        if len(queue) >= 20:
            raise HTTPException(429, "Search limit reached. Try again in a minute.", headers={"Retry-After": "60"})
        queue.append(now)
