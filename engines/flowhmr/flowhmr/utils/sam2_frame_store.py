"""Bounded JPEG frame cache for SAM 2 video prediction on long clips.

The official SAM 2 loader materialises every frame as a 1024x1024 float32 tensor.
For a few hundred 1080p frames that is several gigabytes, so the GUI's crossing
optimisation keeps only a few decoded frames alive and reads the rest from the
JPEG sequence on demand.  Image normalisation matches the upstream loader
exactly, so model inputs are unchanged.
"""
from collections import OrderedDict
from pathlib import Path
import shutil

import torch


def _load_img_as_tensor(path, image_size):
    # Imported lazily so the module (and its unit tests) do not require the
    # SAM 2 package unless video prediction actually runs.
    from flowhmr.utils.sam2_builder import ensure_importable

    ensure_importable()
    from sam2.utils.misc import _load_img_as_tensor as loader

    return loader(path, image_size)


class FrameStore:
    """Lazy, bounded frame sequence with the upstream normalisation."""

    def __init__(self, directory, image_size, keep=4):
        self.files = sorted(Path(directory).glob("*.jpg"), key=lambda path: int(path.stem))
        if not self.files:
            raise ValueError(f"no JPEG frames in {directory}")
        self.image_size = image_size
        self.keep = max(1, int(keep))
        self.cache = OrderedDict()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        index = int(index)
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        value, _, _ = _load_img_as_tensor(str(self.files[index]), self.image_size)
        value = value.to(dtype=torch.float32)
        value = (value - self.mean) / self.std
        self.cache[index] = value
        while len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return value


def bounded_state(predictor, frame_dir, bootstrap_dir, keep=4):
    """Create an inference state that reads frames on demand.

    The official constructor is used once on a one-frame directory so object
    bookkeeping stays pristine; only the image sequence is then swapped for the
    bounded store.  Frame 0 is compared bit-exactly against the official loader.
    """
    store = FrameStore(frame_dir, predictor.image_size, keep=keep)
    bootstrap = Path(bootstrap_dir)
    bootstrap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(store.files[0], bootstrap / "00000.jpg")
    state = predictor.init_state(video_path=str(bootstrap), offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)
    torch.testing.assert_close(store[0], state["images"][0], rtol=0, atol=0)
    state["images"] = store
    state["num_frames"] = len(store)
    return state
