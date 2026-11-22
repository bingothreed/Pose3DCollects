"""Read the established motion ground for archived camera previews and exports."""
import json
from pathlib import Path
import numpy as np


def published_floor(directory,correction,fallback):
    """The final motion's coordinate declaration outranks legacy job metadata."""
    with np.load(Path(directory)/'motion.npz',allow_pickle=False) as z:
        if 'ground_height_m' in z:
            value=float(z['ground_height_m'])
            if not np.isfinite(value):raise ValueError('Invalid published ground')
            return value
    return float((correction or {}).get('floor_m',fallback))


def stored_floor(directory):
    """Read the matching world declaration for a persisted job or scene."""
    root=Path(directory)
    if (root/'scene.json').is_file():
        return float(json.loads((root/'scene.json').read_text(encoding='utf8')).get('ground_m',0.))
    job=json.loads((root/'job.json').read_text(encoding='utf8')) if (root/'job.json').is_file() else {}
    return published_floor(root,job.get('correction'),0.)
