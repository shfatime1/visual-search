"""Bounded HTTP download with DNS validation and connections pinned to public IPs."""
import http.client
import ipaddress
import socket
import ssl
import time
from urllib.parse import urlsplit, urljoin
from .config import MAX_IMAGE_BYTES


def resolve_public(url):
    if len(url) > 2048 or any(ord(c) < 33 or ord(c) == 127 for c in url) or "\\" in url:
        raise ValueError("Invalid image URL.")
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = (parsed.hostname or "").encode("idna").decode("ascii")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid image URL.") from exc
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("Use a public HTTP or HTTPS image URL without credentials.")
    if port != (443 if parsed.scheme == "https" else 80):
        raise ValueError("Only standard HTTP/HTTPS ports are allowed.")
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("Image host could not be resolved.") from exc
    if not addresses:
        raise ValueError("Image host has no addresses.")
    for _, _, _, _, sockaddr in addresses:
        addr = ipaddress.ip_address(sockaddr[0])
        if (not addr.is_global or addr.is_multicast or addr.is_reserved
                or getattr(addr, "ipv4_mapped", None) is not None
                or getattr(addr, "sixtofour", None) is not None
                or getattr(addr, "teredo", None) is not None):
            raise ValueError("Private, local, and special-use image addresses are blocked.")
    return parsed, host, port, addresses[0]


class PinnedConnection(http.client.HTTPConnection):
    def __init__(self, host, port, address, secure, timeout):
        super().__init__(host, port, timeout=timeout)
        self.address, self.secure = address, secure

    def connect(self):
        family, socktype, proto, _, sockaddr = self.address
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(self.timeout)
            sock.connect(sockaddr)
            if self.secure:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
            self.sock = sock
        except BaseException:
            sock.close()
            raise


def download_image(url):
    deadline = time.monotonic() + 25
    try:
        for _ in range(4):
            parsed, host, port, address = resolve_public(url)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Image download timed out.")
            connection = PinnedConnection(host, port, address, parsed.scheme == "https", min(8, remaining))
            try:
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                connection.request("GET", target, headers={"Accept": "image/jpeg,image/png,image/webp", "Accept-Encoding": "identity", "User-Agent": "VisualSearch/1.0"})
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location:
                        raise ValueError("Image redirect has no destination.")
                    url = urljoin(url, location)
                    continue
                if response.status != 200:
                    raise ValueError(f"Image server returned HTTP {response.status}.")
                mime = response.getheader("Content-Type", "").split(";")[0].strip().lower()
                if mime not in {"image/jpeg", "image/png", "image/webp"}:
                    raise ValueError("URL must return a JPEG, PNG, or WebP image.")
                length = response.getheader("Content-Length")
                if length and int(length) > MAX_IMAGE_BYTES:
                    raise ValueError("Remote image exceeds 5 MiB.")
                body = bytearray()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError("Image download timed out.")
                    if connection.sock:
                        connection.sock.settimeout(min(8, remaining))
                    chunk = response.read1(min(65536, MAX_IMAGE_BYTES + 1 - len(body)))
                    if not chunk:
                        return bytes(body)
                    body.extend(chunk)
                    if len(body) > MAX_IMAGE_BYTES:
                        raise ValueError("Remote image exceeds 5 MiB.")
            finally:
                connection.close()
        raise ValueError("Too many image redirects.")
    except (OSError, http.client.HTTPException, UnicodeError) as exc:
        raise ValueError("Could not download the remote image safely.") from exc
