"""Install only the verified SAM 2.1 inference package, like the HaMeR component.

The portable build does not vendor the 60 MB SAM 2 repository.  This module
fetches the pinned upstream archive, verifies its SHA-256, and extracts only the
``sam2`` Python package (code + Hydra configs) plus a tiny ``iopath`` shim so the
official model code runs without Hydra and without the CUDA post-processing
extension that FlowHMR never calls.
"""
from hashlib import sha256
import io
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

from . import model_files as files, model_sources as sources


SOURCE_PREFIX = f"sam2-{sources.SAM2_SOURCE_REVISION}/sam2/"
KEEP_SUFFIXES = (".py", ".yaml", ".yml")
REQUIRED = (
    "sam2/__init__.py",
    "sam2/sam2_video_predictor.py",
    "sam2/modeling/sam2_base.py",
    "sam2/utils/misc.py",
    "sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
    "iopath/common/file_io.py",
)
# Only ``g_pathmgr.open`` is reached from SAM 2's Hiera backbone (a pretrained
# backbone file is never used here because FlowHMR loads the official checkpoint).
_IOPATH_SHIM = {
    "iopath/__init__.py": '"""Minimal iopath package installed by FlowHMR for SAM 2."""\n',
    "iopath/common/__init__.py": '"""Common file helpers installed by FlowHMR for SAM 2."""\n',
    "iopath/common/file_io.py": '''"""Just enough of iopath for the official SAM 2 inference code."""

import builtins
import os


class PathManager:
    """Small stand-in for iopath's PathManager; FlowHMR only opens local paths."""

    @staticmethod
    def open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        return builtins.open(path, mode, buffering, encoding, errors, newline)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def isfile(path):
        return os.path.isfile(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def get_local_path(path, **kwargs):
        return str(path)

    @classmethod
    def mkdirs(cls, path):
        os.makedirs(path, exist_ok=True)


g_pathmgr = PathManager()
''',
}


class BadArchive(ValueError):
    pass


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.close()
            tmp.replace(path)
        finally:
            temporary.close()
            tmp.unlink(missing_ok=True)


def runtime_installed():
    try:
        revision = (files.SAM2_RUNTIME / "revision.txt").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return False
    return revision == sources.SAM2_SOURCE_REVISION and all(
        (files.SAM2_RUNTIME / name).is_file() for name in REQUIRED)


def code_available():
    """True when the installed component or an editable SAM 2 package is importable."""
    if runtime_installed():
        return True
    from importlib.util import find_spec

    try:
        return find_spec("sam2") is not None
    except (ImportError, ValueError):
        return False


def install_code(source_archive, *, check=lambda: None):
    data = source_archive.read_bytes()
    if sha256(data).hexdigest() != sources.SAM2_SOURCE_SHA256:
        raise BadArchive("SAM 2 source archive SHA-256 mismatch")
    extracted = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if not name.startswith(SOURCE_PREFIX) or name.endswith("/"):
                    continue
                relative = name[len(SOURCE_PREFIX):]
                if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                    continue
                if not relative.endswith(KEEP_SUFFIXES):
                    continue
                check()
                _atomic_write(files.SAM2_RUNTIME / "sam2" / relative, archive.read(name))
                extracted += 1
    except zipfile.BadZipFile as error:
        raise BadArchive("SAM 2 source archive is not a ZIP file") from error
    if not extracted:
        raise BadArchive("SAM 2 source archive contained no inference package")
    for relative, text in _IOPATH_SHIM.items():
        check()
        _atomic_write(files.SAM2_RUNTIME / relative, text.encode("utf-8"))
    _atomic_write(files.SAM2_RUNTIME / "revision.txt",
                  (sources.SAM2_SOURCE_REVISION + "\n").encode("ascii"))


def crossing_issue():
    """Return ``(code, message)`` when the crossing optimisation cannot run."""
    if not files.crossing_optimization_available():
        return ("crossing_model_missing",
                "请先在右上角「模型」下载可选的 SAM 2.1 人物分割模型，再开启人物交叉优化。")
    if not code_available():
        return ("crossing_code_missing",
                "请先在右上角「模型」下载 SAM 2.1 运行组件，再开启人物交叉优化。")
    return None


def crossing_ready():
    return crossing_issue() is None
