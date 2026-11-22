"""Just enough of iopath for the official SAM 2 inference code."""

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
