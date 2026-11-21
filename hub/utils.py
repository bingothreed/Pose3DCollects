"""Small shared helpers (no third-party imports)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def now() -> float:
    return time.time()


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} PB"


def file_info(path: Path) -> dict:
    try:
        st = path.stat()
        return {"exists": True, "size": st.st_size, "size_text": human_size(st.st_size)}
    except OSError:
        return {"exists": False, "size": 0, "size_text": "—"}


def http_json(url: str, method: str = "GET", payload=None, timeout: float = 5.0,
              headers: dict | None = None):
    """Return parsed JSON. Raises RuntimeError with a readable message on failure."""
    body = None
    hdrs = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"cannot reach {url}: {exc}") from exc


def http_bytes(url: str, timeout: float = 10.0) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc


def urlopen_stream(url: str, headers: dict | None = None, timeout: float = 60.0):
    """Open a URL and return the raw response so callers can stream it.

    The caller owns closing the returned object.
    """
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=20)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def popen_kwargs(new_group: bool = True) -> dict:
    kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        flags = 0
        if new_group:
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["creationflags"] = flags
    return kwargs


def tail_file(path: Path, limit: int = 60) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def resolve_command(base_python: str) -> list[str]:
    """Turn a configured interpreter string ("py -3.12", "C:\\...\\python.exe") into argv."""
    import shlex
    base_python = (base_python or "python").strip()
    if not base_python:
        base_python = "python"
    if os.name == "nt" and (base_python.startswith("py ") or base_python == "py"):
        return shlex.split(base_python, posix=False)
    return shlex.split(base_python)


_VERSION_CACHE: dict[str, tuple[float, str]] = {}


def python_version(python: str, timeout: float = 15.0) -> str:
    cached = _VERSION_CACHE.get(python)
    if cached and time.time() - cached[0] < (300 if cached[1] else 10):
        return cached[1]
    version = ""
    try:
        out = subprocess.run([python, "-c", "import sys; print(sys.version.split()[0])"],
                             capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            version = out.stdout.strip()
    except Exception:
        pass
    _VERSION_CACHE[python] = (time.time(), version)
    return version


def probe_python_candidates(configured: str = "") -> list[str]:
    """Working base interpreters for creating a venv, best first."""
    seen: set[str] = set()
    found: list[str] = []
    candidates = [configured, "py -3.12", "py -3.11", "py -3.10", "python"]
    if sys.executable:
        candidates.append(sys.executable)
    for item in candidates:
        item = (item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        argv = resolve_command(item)
        try:
            out = subprocess.run(argv + ["-c", "import sys; print(sys.version_info[:2])"],
                                 capture_output=True, text=True, timeout=12)
        except Exception:
            continue
        if out.returncode == 0 and "(" in out.stdout:
            found.append(item)
    return found
