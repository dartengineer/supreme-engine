"""Tiny HTML -> PDF service using WeasyPrint. Runs in Docker so Windows doesn't need the GTK libraries.

POST /pdf   body: HTML (UTF-8)  ->  application/pdf
GET  /health                    ->  "ok"

Safety: WeasyPrint is never allowed to fetch URLs (templates embed everything they need), requests are
size-limited, and the container is published on 127.0.0.1 only.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from weasyprint import HTML
from weasyprint.urls import default_url_fetcher

MAX_BYTES = 2 * 1024 * 1024


def no_network_fetcher(url, *args, **kwargs):
    # Allow inline data: URIs only; refuse http(s), file and everything else.
    if url.startswith("data:"):
        return default_url_fetcher(url, *args, **kwargs)
    raise ValueError(f"Fetching {url!r} is not allowed")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/pdf":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES:
            return self._send(413, b"HTML missing or too large", "text/plain")
        html = self.rfile.read(length).decode("utf-8")
        try:
            pdf = HTML(string=html, url_fetcher=no_network_fetcher).write_pdf()
        except Exception as e:  # report rendering errors instead of dropping the connection
            return self._send(500, f"Rendering failed: {e}".encode(), "text/plain")
        self._send(200, pdf, "application/pdf")

    def log_message(self, fmt, *args):  # quieter logs
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 5001), Handler).serve_forever()
