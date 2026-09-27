"""Website + live-demo server (standard library only; the Hugging Face Space runs this).

  GET  /                      the website (demo/site), with HTTP Range support so videos can seek
  POST /api/jobs              body = the video file (header X-Filename); starts processing -> {"id": ...}
  GET  /api/jobs/<id>         {"state": queued|running|done|error, "progress": 0..1, "message", "position", "result"}
  GET  /jobs/<id>/<file>      annotated.mp4 / events.json of a finished job

One job runs at a time (CPU); later uploads wait in a queue and see their position. Uploads and results are deleted
after JOB_TTL seconds.

  python demo/server.py --port 7860
"""
import argparse
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import process

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
MAX_MB = 10_000               # disk safety only: any size is accepted, the pipeline scales it down
JOB_TTL = 3600
WORK = Path(tempfile.gettempdir()) / "traffic_demo_jobs"

jobs = {}
order = []
todo = queue.Queue()
lock = threading.Lock()


def worker():
    while True:
        jid = todo.get()
        job = jobs.get(jid)
        if job is None:
            continue
        job.update(state="running", message="Starting")

        def progress(frac, msg, job=job):
            job.update(progress=round(float(frac), 3), message=msg)
        try:
            result, video, _ = process.run(job["path"], job["dir"], progress=progress)
            job.update(state="done", progress=1.0, message="Done", result=result, zones=f"/jobs/{jid}/zones.jpg",
                       video=f"/jobs/{jid}/annotated.mp4" if video.exists() else None,
                       json=f"/jobs/{jid}/events.json")
        except ValueError as exc:
            job.update(state="error", message=str(exc))
        except Exception:
            traceback.print_exc()
            job.update(state="error", message="Processing failed on this file. Please try another clip "
                                              "(H.264 .mp4, up to 2 minutes).")
        finally:
            try:
                os.remove(job["path"])
            except OSError:
                pass
            with lock:
                if jid in order:
                    order.remove(jid)


def janitor():
    while True:
        time.sleep(300)
        now = time.time()
        for jid, job in list(jobs.items()):
            if now - job["created"] > JOB_TTL and job["state"] in ("done", "error"):
                shutil.rmtree(job["dir"], ignore_errors=True)
                jobs.pop(jid, None)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SITE), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        if self.path.split("?")[0].endswith((".json", ".js", ".html", ".css", "/")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/jobs":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._json({"error": "Empty upload."}, 400)
        if n > MAX_MB * 1_000_000:
            return self._json({"error": f"File is larger than {MAX_MB} MB."}, 413)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", self.headers.get("X-Filename", "upload.mp4"))[-80:]
        jid = uuid.uuid4().hex[:12]
        d = WORK / jid
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        with open(path, "wb") as f:
            left = n
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        jobs[jid] = dict(state="queued", progress=0.0, message="Waiting for the previous upload to finish",
                         path=str(path), dir=str(d), created=time.time(), result=None)
        with lock:
            order.append(jid)
        todo.put(jid)
        return self._json({"id": jid})

    def do_GET(self):
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", self.path)
        if m:
            job = jobs.get(m.group(1))
            if job is None:
                return self._json({"error": "Unknown or expired job."}, 404)
            with lock:
                pos = order.index(m.group(1)) if m.group(1) in order else 0
            out = {k: job.get(k) for k in ("state", "progress", "message", "result", "video", "json", "zones")}
            out["position"] = pos
            return self._json(out)
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/(annotated\.mp4|events\.json|zones\.jpg)", self.path.split("?")[0])
        if m:
            job = jobs.get(m.group(1))
            if job is None:
                return self._json({"error": "Unknown or expired job."}, 404)
            return self._serve_file(Path(job["dir"]) / m.group(2))
        return super().do_GET()

    def _serve_file(self, path):
        """Serve a file with Range support (browsers need it to seek in videos)."""
        if not path.exists():
            return self._json({"error": "not ready"}, 404)
        size = path.stat().st_size
        rng = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        start, end = 0, size - 1
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
            else:
                start = max(0, size - int(rng.group(2)))
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def send_head(self):
        """Static site files, with Range support for the sample videos."""
        path = Path(self.translate_path(self.path))
        if path.is_file() and self.headers.get("Range"):
            self._serve_file(path)
            return None
        return super().send_head()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = p.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=janitor, daemon=True).start()
    print(f"serving {SITE} and the demo API on http://0.0.0.0:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
