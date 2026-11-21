"""Pose3D Collects desktop shell.

The UI is still the same local web app. This launcher opens it in a native
WebView2 window through pywebview. If that is unavailable it falls back to
Chrome/Edge's standalone app window (`--app`, no tabs or address bar), and
finally to the default browser.

The Flask hub is started inside this process when no healthy hub is already
running on the configured port. Closing the window then stops the hub.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
APP_MARKER = "Pose3D Collects"
DEFAULT_PORT = 7860


def hub_healthy(port: int, timeout: float = 0.9) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("app") == APP_MARKER
    except Exception:
        return False


def wait_healthy(port: int, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if hub_healthy(port):
            return True
        time.sleep(0.2)
    return False


def find_browser() -> str:
    candidates: list[Path] = []
    for env_name in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(env_name)
        if not base:
            continue
        candidates.extend([
            Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ])
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


def start_hub(port: int) -> tuple[str, object | None, bool]:
    """Return (url, server, owned). Reuses a healthy hub when there is one."""
    for candidate in range(port, port + 20):
        if hub_healthy(candidate):
            return f"http://127.0.0.1:{candidate}", None, False
    try:
        import app as hub_module  # the Flask application module
        from werkzeug.serving import make_server
    except Exception as exc:
        raise RuntimeError(f"无法加载 Hub：{exc}") from exc

    for candidate in range(port, port + 20):
        try:
            server = make_server("127.0.0.1", candidate, hub_module.app, threaded=True)
        except OSError:
            continue
        threading.Thread(target=server.serve_forever, daemon=True, name="pose3d-hub").start()
        if wait_healthy(candidate):
            return f"http://127.0.0.1:{candidate}", server, True
        server.shutdown()
        server.server_close()
    raise RuntimeError("找不到可用端口来启动 Hub")


def webview_version() -> str:
    try:
        return importlib.metadata.version("pywebview")
    except importlib.metadata.PackageNotFoundError:
        return ""


def open_native_window(url: str, auto_close: float = 0.0) -> bool:
    try:
        import webview
    except Exception as exc:
        print(f"[Pose3D] pywebview 不可用（{exc}），改用浏览器应用窗口。")
        return False
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True
    except Exception:
        pass
    storage = DATA_DIR / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    try:
        window = webview.create_window(
            "Pose3D Collects", url,
            width=1500, height=950, min_size=(1080, 680),
            background_color="#07080c", text_select=True,
        )
        if auto_close and window is not None:
            threading.Timer(auto_close, window.destroy).start()
        webview.start(private_mode=False, storage_path=str(storage), debug=False)
        return True
    except Exception as exc:
        print(f"[Pose3D] 原生窗口启动失败（{exc}），改用浏览器应用窗口。")
        return False


def open_app_window(url: str) -> bool:
    browser = find_browser()
    if not browser:
        return False
    profile = DATA_DIR / "desktop-profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser, f"--app={url}", "--window-size=1500,950",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
        "--disable-background-mode", "--disable-features=Translate",
    ]
    print(f"[Pose3D] 使用浏览器应用窗口：{browser}")
    try:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return True
    except Exception as exc:
        print(f"[Pose3D] 浏览器应用窗口启动失败（{exc}）。")
        return False


def open_default_browser(url: str) -> None:
    print("[Pose3D] 未找到可用的独立窗口，改用系统默认浏览器。")
    webbrowser.open(url)
    print("[Pose3D] 按 Ctrl+C 或关闭此控制台即可停止服务。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose3D Collects desktop window")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check", action="store_true", help="只检查桌面窗口组件，不打开窗口")
    parser.add_argument("--auto-close", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit("Pose3D Collects 需要 Python 3.10 或更高版本")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    if args.check:
        print(json.dumps({
            "existing_hub": hub_healthy(args.port),
            "hub_url": f"http://127.0.0.1:{args.port}" if hub_healthy(args.port) else "",
            "pywebview": webview_version(),
            "browser": find_browser(),
        }, ensure_ascii=False, indent=2))
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        url, server, owned = start_hub(args.port)
    except RuntimeError as exc:
        raise SystemExit(f"[Pose3D] {exc}") from exc
    print(f"[Pose3D] Hub: {url}")
    try:
        if not open_native_window(url, args.auto_close) and not open_app_window(url):
            open_default_browser(url)
    finally:
        if owned and server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
