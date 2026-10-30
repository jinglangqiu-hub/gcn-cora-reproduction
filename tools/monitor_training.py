"""Read-only, localhost training monitor. Uses the Python standard library only."""

import argparse
import csv
import io
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


PROJECT = Path(__file__).resolve().parents[1]
FIELDS = ("epoch", "eval_train_objective", "val_objective",
          "eval_train_accuracy", "val_accuracy", "epoch_seconds")


def read_json(path):
    # Training may be writing a JSON file at the instant we poll it.
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_metrics(path):
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    # Ignore a trailing partial write; it will be available on the next poll.
    end = content.rfind("\n")
    if end < 0:
        return []
    rows = []
    for raw in csv.DictReader(io.StringIO(content[:end + 1])):
        try:
            row = {key: float(raw[key]) for key in FIELDS}
            if not all(math.isfinite(value) for value in row.values()):
                continue
            if not row["epoch"].is_integer() or row["epoch"] <= 0:
                continue
            if rows and row["epoch"] <= rows[-1]["epoch"]:
                continue
            row["epoch"] = int(row["epoch"])
            rows.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def snapshot(root, requested=""):
    candidates = []
    if root.is_dir():
        for folder in root.iterdir():
            if not folder.is_dir() or folder.resolve().parent != root.resolve():
                continue
            metrics = folder / "metrics.csv"
            config = read_json(folder / "config.json")
            if not metrics.exists() and config.get("protocol") != "official_code_last":
                continue
            try:
                modified = (metrics if metrics.exists() else folder).stat().st_mtime
            except OSError:
                continue
            candidates.append((modified, folder, config))
    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    selected = next((item for item in candidates if item[1].name == requested), None)
    if selected is None and not requested and candidates:
        selected = candidates[0]
    state = {"runs": [item[1].name for item in candidates], "run": None,
             "rows": [], "config": {}, "status": {}, "result": {}, "age_seconds": None}
    if selected:
        _, folder, config = selected
        state.update(run=folder.name, config=config, rows=read_metrics(folder / "metrics.csv"),
                     status=read_json(folder / "status.json"), result=read_json(folder / "result.json"))
        try:
            state["age_seconds"] = max(0, time.time() - (folder / "metrics.csv").stat().st_mtime)
        except OSError:
            pass
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=PROJECT / "runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-file", type=Path, help="Optional server PID/URL record")
    args = parser.parse_args()
    root = args.runs_root.resolve()
    page_path = Path(__file__).with_name("training_monitor.html")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/":
                body, content_type, status = page_path.read_bytes(), "text/html; charset=utf-8", 200
            elif request.path == "/api/state":
                requested = parse_qs(request.query).get("run", [""])[0]
                body = json.dumps(snapshot(root, requested), ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
                content_type, status = "application/json; charset=utf-8", 200
            else:
                body, content_type, status = b"Not found", "text/plain", 404
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *values):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    info = {"pid": os.getpid(), "url": f"http://127.0.0.1:{server.server_port}",
            "runs_root": str(root)}
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
