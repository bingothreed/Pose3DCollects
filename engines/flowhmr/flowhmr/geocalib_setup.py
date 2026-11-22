"""Pinned GeoCalib inference runtime and feature-specific readiness."""
from importlib import import_module, util
from pathlib import Path
import sys

from . import model_files as files, model_integrity, model_sources

SOURCE_REVISION = '1dedb29f94fc11b896c5f572005978aeebccd67f'
SOURCE_ROOT = files.PROJECT_ROOT / 'third_party/geocalib_inference'


def runtime_issue(device):
    if str(device).startswith('xpu'):
        return 'unsupported_device'
    if not (SOURCE_ROOT / 'geocalib/extractor.py').is_file():
        return 'runtime_missing'
    if any(util.find_spec(name) is None for name in ('kornia', 'kornia_rs')):
        return 'runtime_missing'
    return None


def readiness(device, *, verify=False):
    path = files.resolve_model(files.GEOCALIB)
    verdict = model_integrity.check(path)
    if not verdict.ok:
        return 'weights_missing' if verdict.state == 'missing' else 'weights_corrupt'
    if verify:
        try:
            if model_integrity.digest(path) != model_sources.PUBLIC_ASSETS['geocalib'][0].sha256:
                return 'weights_corrupt'
        except OSError:
            return 'weights_corrupt'
    return runtime_issue(device)


def load_class():
    # Never resolve the research checkout or optional_packages. Reject an
    # already imported foreign package instead of mixing two implementations.
    existing = sys.modules.get('geocalib')
    if existing and not Path(existing.__file__).resolve().is_relative_to(SOURCE_ROOT.resolve()):
        raise ImportError('A different GeoCalib package is already loaded; restart the application.')
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    module = import_module('geocalib')
    if not Path(module.__file__).resolve().is_relative_to(SOURCE_ROOT.resolve()):
        raise ImportError('GeoCalib did not resolve to the bundled inference runtime.')
    return module.GeoCalib
