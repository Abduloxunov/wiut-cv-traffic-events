"""Local static server with HTTP Range support, so the label tool can seek in videos.

python -m http.server ignores Range requests, and browsers then cannot seek in a served MP4.
Serves on 127.0.0.1 only (footage never leaves this machine).

  python tools/serve.py --dir .. --port 8766
  then http://127.0.0.1:8766/submission/tools/label_tool.html?video=/proxies/C3897.MP4&import=/proxies/proposals_C3897.json
"""
import argparse
import functools
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else size - 1
        if start >= size:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source, outputfile):
        left = getattr(self, "_remaining", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(1 << 16, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    handler = functools.partial(RangeHandler, directory=a.dir)
    print(f"serving {os.path.abspath(a.dir)} on http://127.0.0.1:{a.port}")
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
