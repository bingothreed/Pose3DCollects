"""Download GEM-X Body Pose models (runs inside the GEM-X virtual environment).

- YOLOX HumanArt detector: OpenMMLab archive, end2end.onnx extracted.
- Body Pose checkpoint: nvidia/GEM-X on Hugging Face (~3.5 GB).
- SOMA 0.3.3 assets: nvidia/SOMA-X revision v0.3.3.

Existing complete files are left untouched, so the task is resumable.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

YOLOX_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "yolox_x_8xb8-300e_humanart-a39d44ed.zip")
GEMX_REPO = "nvidia/GEM-X"
SOMA_REPO = "nvidia/SOMA-X"
SOMA_REVISION = "v0.3.3"
SOMA_FILES = [
    "SOMA_neutral.npz",
    "SOMA_template_rig.usda",
    "SOMA_procedural_transforms.json",
    "correctives_model.pt",
]


def log(message: str) -> None:
    print(message, flush=True)


def _proxy_handler(proxy: str):
    if not proxy:
        return urllib.request.build_opener()
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    return urllib.request.build_opener(handler)


def _complete(path: Path, size: int | None) -> bool:
    return path.is_file() and (size is None or path.stat().st_size == size)


def download_url(url: str, dest: Path, *, proxy: str, expected: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _complete(dest, expected):
        log(f"[跳过] 已存在: {dest}")
        return dest
    attempts = [False, True] if proxy else [False]
    for use_proxy in attempts:
        try:
            log(f"下载 {url}")
            opener = _proxy_handler(proxy if use_proxy else "")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with opener.open(url, timeout=60) as response, tmp.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        log(f"  {done * 100 // total}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")
            if expected and tmp.stat().st_size != expected:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"大小不符: {tmp} != {expected}")
            tmp.replace(dest)
            return dest
        except Exception as exc:
            if use_proxy is attempts[-1]:
                raise RuntimeError(f"下载失败: {url}: {exc}") from exc
            log(f"[重试] 直连失败（{exc}），改用代理 {proxy}")
    raise RuntimeError(f"下载失败: {url}")


def download_yolox(models_dir: Path, proxy: str) -> Path:
    target = models_dir / "yolox_humanart.onnx"
    if target.is_file() and target.stat().st_size > 100_000_000:
        log(f"[跳过] 已存在: {target}")
        return target
    archive = models_dir / "yolox_humanart.zip"
    download_url(YOLOX_URL, archive, proxy=proxy, expected=368200375)
    log("解压 YOLOX end2end.onnx …")
    with ZipFile(archive) as zip_file:
        member = next(name for name in zip_file.namelist() if name.endswith("/end2end.onnx"))
        with zip_file.open(member) as source, target.open("wb") as out:
            copyfileobj(source, out)
    archive.unlink(missing_ok=True)
    return target


def download_hf(models_dir: Path, soma_dir: Path, proxy: str) -> None:
    from huggingface_hub import hf_hub_download

    def hub(repo: str, filename: str, *, revision: str | None = None,
            local_dir: Path, min_size: int = 1000) -> Path:
        dest = local_dir / filename
        if dest.is_file() and dest.stat().st_size >= min_size:
            log(f"[跳过] 已存在: {dest}")
            return dest
        log(f"下载 {repo} / {filename}"
            + (f"（revision {revision}）" if revision else ""))
        path = Path(hf_hub_download(
            repo_id=repo, filename=filename, revision=revision,
            local_dir=str(local_dir), local_dir_use_symlinks=False,
        ))
        if path != dest and path.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.replace(dest)
        return dest

    for filename in ("body_s240000.ckpt", "body_config.json"):
        hub(GEMX_REPO, filename, local_dir=models_dir)
    for filename in SOMA_FILES:
        hub(SOMA_REPO, filename, revision=SOMA_REVISION, local_dir=soma_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--soma-dir", required=True)
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    soma_dir = Path(args.soma_dir)
    download_yolox(models_dir, args.proxy)
    download_hf(models_dir, soma_dir, args.proxy)
    log("GEM-X 模型准备完成。")
    log("models: " + str(models_dir))
    log("soma:   " + str(soma_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
