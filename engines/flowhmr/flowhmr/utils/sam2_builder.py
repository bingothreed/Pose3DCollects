"""Build the official SAM 2.1 video predictor without Hydra.

The upstream ``sam2.build_sam`` helper depends on Hydra for config composition.
The portable runtime does not ship Hydra, so this module loads the same YAML
config with OmegaConf (part of the pinned runtime) and instantiates the model
with a small recursive ``_target_`` resolver that covers SAM 2's configuration.
"""
import importlib
import sys
from pathlib import Path

import torch

from flowhmr import model_files as files

_TARGET_KEYS = ("_target_", "_convert_", "_partial_")


def runtime_available():
    return (files.SAM2_RUNTIME / "sam2" / "__init__.py").is_file()


def ensure_importable():
    """Put the installed SAM 2 runtime (and its iopath shim) on ``sys.path``."""
    runtime = files.SAM2_RUNTIME
    if runtime_available() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))


def _config_path(relative):
    runtime = files.SAM2_RUNTIME
    if runtime_available():
        return runtime / "sam2" / relative
    import sam2  # development environment with an editable SAM 2 install

    return Path(sam2.__path__[0]) / relative


def _resolve(target):
    module_name, _, attribute = str(target).rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _instantiate(value):
    if isinstance(value, dict):
        target = value.get("_target_")
        if target:
            kwargs = {key: _instantiate(item) for key, item in value.items() if key not in _TARGET_KEYS}
            return _resolve(target)(**kwargs)
        return {key: _instantiate(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_instantiate(item) for item in value)
    return value


def _model_config(relative, apply_postprocessing):
    from omegaconf import OmegaConf

    path = _config_path(relative)
    if not path.is_file():
        raise FileNotFoundError(f"SAM 2 config is missing: {path}")
    config = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    model = config["model"]
    model["_target_"] = "sam2.sam2_video_predictor.SAM2VideoPredictor"
    if apply_postprocessing:
        # Mirrors the overrides in sam2.build_sam.build_sam2_video_predictor.
        extra = dict(model.get("sam_mask_decoder_extra_args") or {})
        extra.update({"dynamic_multimask_via_stability": True,
                      "dynamic_multimask_stability_delta": 0.05,
                      "dynamic_multimask_stability_thresh": 0.98})
        model["sam_mask_decoder_extra_args"] = extra
        model["binarize_mask_from_pts_for_mem_enc"] = True
        model["fill_hole_area"] = 8
    return model


def build_video_predictor(config_relative, ckpt_path, device, *, apply_postprocessing=True):
    """Equivalent of ``sam2.build_sam.build_sam2_video_predictor`` minus Hydra."""
    ensure_importable()
    model = _instantiate(_model_config(config_relative, apply_postprocessing))
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"SAM 2 checkpoint does not match the model: "
                           f"{len(missing)} missing / {len(unexpected)} unexpected keys")
    model = model.to(device)
    model.eval()
    return model
