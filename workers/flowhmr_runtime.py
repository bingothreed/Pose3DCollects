"""Configure FlowHMR's Windows inference runtime (PyTorch) automatically.

Run with FlowHMR's own Python from its project root. The heavy lifting is done
by FlowHMR's RuntimeEnvironments, so downloads, SHA-256 checks, GPU probes and
the slot layout match the FlowHMR-Windows desktop app exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _fmt_bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(value or 0)
    unit = 0
    while number >= 1024 and unit < len(units) - 1:
        number /= 1024
        unit += 1
    return f"{number:.1f} {units[unit]}" if unit else f"{int(number)} {units[unit]}"


def _runtime_label(name: str) -> str:
    if name == "cpu":
        return "CPU"
    if name == "xpu":
        return "Intel XPU"
    if name == "system":
        return "现有项目环境"
    if name.startswith("cu") and name[2:].isdigit():
        return f"CUDA {name[2:-1]}.{name[-1]}"
    if name.startswith("rocm"):
        return f"ROCm ({name})"
    return name or "—"


def _choose(details: dict) -> dict | None:
    """Prefer an installed compatible slot, then the recommended download."""
    options = [item for item in details.get("options", []) if item.get("id")]
    compatible = [item for item in options if item.get("compatible", True)]
    installed = [item for item in compatible if item.get("installed")]
    selected = details.get("selected")
    if selected:
        for item in installed:
            if item["id"] == selected:
                return item
    for item in installed:
        if item.get("recommended"):
            return item
    for item in installed:
        if item["id"] != "system":
            return item
    for item in compatible:
        if item.get("recommended"):
            return item
    for item in compatible:
        if item["id"] == "cpu":
            return item
    for item in compatible:
        if item["id"] != "system":
            return item
    return None


class Progress:
    def __init__(self) -> None:
        self.last_key = None
        self.last_at = 0.0

    def __call__(self, phase: str, filename: str, size: int, total: int) -> None:
        key = (phase, filename, total)
        now = time.monotonic()
        percent = (size / total * 100.0) if total else None
        changed = key != self.last_key
        due = now - self.last_at >= 1.0
        if not (changed or due or (percent is not None and percent >= 100)):
            return
        self.last_key = key
        self.last_at = now
        if phase == "download":
            detail = f" {percent:.0f}% ({_fmt_bytes(size)} / {_fmt_bytes(total)})" if percent is not None else ""
            print(f"下载 {filename}{detail}", flush=True)
        elif phase == "install":
            detail = f" {percent:.0f}%" if percent is not None else ""
            print(f"解包 {filename}{detail}", flush=True)
        elif phase == "test":
            print("验证运行环境（实际加载 PyTorch 并测试算子）…", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="FlowHMR project directory")
    parser.add_argument("--status", action="store_true", help="print runtime state as JSON")
    parser.add_argument("--dry-run", action="store_true", help="only show the chosen runtime")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[错误] FlowHMR 目录不存在: {root}", flush=True)
        return 2
    sys.path.insert(0, str(root))
    os.chdir(root)

    try:
        from flowhmr.runtime_environments import RUNTIME
    except Exception as exc:  # noqa: BLE001 - report the engine's import problem
        print(f"[错误] 无法加载 FlowHMR 运行环境模块: {type(exc).__name__}: {exc}", flush=True)
        return 2

    RUNTIME.configure_process()
    details = RUNTIME.details()
    summary = RUNTIME.summary()

    if args.status:
        payload = {
            "summary": summary,
            "gpu": details.get("gpu"),
            "options": [
                {key: item.get(key) for key in ("id", "label", "installed", "compatible", "recommended", "bytes", "detail")}
                for item in details.get("options", [])
            ],
            "directory": details.get("directory"),
            "download": details.get("download"),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0

    gpu = details.get("gpu")
    if gpu:
        print(f"检测到显卡: {gpu.get('name')}（驱动 {gpu.get('driver')}）", flush=True)

    if summary.get("ready"):
        print(f"推理运行环境已就绪: {summary.get('label') or _runtime_label(summary.get('active') or '')}", flush=True)
        print("[runtime] already ready", flush=True)
        return 0

    if not details.get("supported"):
        print("[错误] 当前系统不支持自动配置运行环境（需要 Windows x64 + Python 3.10）。", flush=True)
        return 2

    target = _choose({**details, "selected": summary.get("selected")})
    if target is None:
        print("[错误] 没有找到与当前设备兼容的运行环境，请检查显卡驱动。", flush=True)
        return 2
    label = target.get("label") or _runtime_label(target["id"])
    installed = bool(target.get("installed"))

    if args.dry_run:
        action = "选择已安装的" if installed else "下载并安装"
        print(f"[dry-run] 将{action}运行环境: {label} ({target['id']})", flush=True)
        return 0

    print(f"目标运行环境: {label} ({target['id']}){'（已安装）' if installed else ''}", flush=True)
    try:
        if not installed:
            size = target.get("bytes")
            suffix = f"，约 {_fmt_bytes(size)}" if size else ""
            print(f"开始下载并安装 {label}{suffix}（支持断点续传）…", flush=True)
            RUNTIME.install(target["id"], Progress(), lambda: None)
            print("解包与验证完成，正在启用…", flush=True)
        RUNTIME.select(target["id"])
    except Exception as exc:  # noqa: BLE001 - keep the setup log readable
        print(f"[错误] 运行环境配置失败: {type(exc).__name__}: {exc}", flush=True)
        return 1

    print(f"已启用运行环境: {label}", flush=True)
    print(f"[runtime] restart required: {target['id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
