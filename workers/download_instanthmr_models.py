"""Download InstantHMR ONNX models into a local models directory.

Runs inside the InstantHMR virtual environment (needs huggingface_hub).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "momolesang/InstantHMR"


def download(filename: str, models_dir: Path) -> Path:
    target = models_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        print(f"已存在，跳过: {target} ({target.stat().st_size:,} bytes)", flush=True)
        return target
    from huggingface_hub import hf_hub_download

    print(f"下载 {REPO_ID}/{filename} …", flush=True)
    path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(models_dir)))
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    print(f"完成: {target} ({target.stat().st_size:,} bytes)", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--fit", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--body", action="store_true", help="also download mhr-body.onnx")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    name = f"instanthmr-{args.model}.onnx"
    if args.fit:
        name = f"instanthmr-{args.model}-fit{args.fit}.onnx"
    download(name, models_dir)
    if args.body:
        download("mhr-body.onnx", models_dir)
    print("模型准备完成。", flush=True)


if __name__ == "__main__":
    main()
