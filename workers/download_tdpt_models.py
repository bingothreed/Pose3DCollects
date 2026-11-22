"""Download the TDPT (ThreeDPoseTracker) VNect model into the engine's models dir.

Source: digital-standard/ThreeDPoseTracker (public GitHub repo, raw CDN).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib import request

MODEL_URL = ("https://raw.githubusercontent.com/digital-standard/ThreeDPoseTracker/"
             "master/Assets/Model/MobileNet3D3_6.onnx")
EXPECTED_SIZE = 33183458


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--proxy", default="", help="HTTP proxy, e.g. http://127.0.0.1:21882")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "MobileNet3D3_6.onnx"

    if target.is_file() and target.stat().st_size == EXPECTED_SIZE:
        print(f"模型已存在且大小正确: {target}", flush=True)
        return

    handlers = []
    proxy = (args.proxy or "").strip()
    if proxy:
        handlers.append(request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = request.build_opener(*handlers)

    print(f"下载 {MODEL_URL}", flush=True)
    print(f"目标 {target}", flush=True)
    tmp = target.with_suffix(".onnx.part")
    done = 0
    try:
        with opener.open(MODEL_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with tmp.open("wb") as out:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total and done % (total // 20 + 1) < 256 * 1024:
                        print(f"  {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB",
                              flush=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if done != EXPECTED_SIZE:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"下载大小不符: 期望 {EXPECTED_SIZE}，实际 {done}")
    tmp.replace(target)
    print(f"完成: {target} ({done / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        sys.exit(1)
