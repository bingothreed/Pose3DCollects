"""Download sources and the verified upstream versions used by FlowHMR."""

from dataclasses import dataclass

from . import model_files as files


FLOWHMR_REPO = "fafsaf1/flowhmr-0.46B"
FLOWHMR_REVISION = "1eb105e554e0cfc1846ea5feac8c2d516a36d319"
HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"
MODELSCOPE = "https://www.modelscope.cn"
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth"
REGRESSOR_URL = "https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
MANO_URL = "https://mano.is.tue.mpg.de/download.php"
SMPLH_DOWNLOAD = ("Extended SMPL+H model", "smplh.tar.xz")
MANO_DOWNLOAD = ("Models & Code", f"{files.MANO_BUNDLE}.zip")
HAMER_PROJECT_URL = "https://github.com/geopavlakos/hamer#demo"
HAMER_DEMO_URL = "https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT"
HAMER_ARCHIVE_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
HAMER_MIRROR_REPO = "Leo-TX/hamer"
HAMER_MIRROR_REVISION = "f64df318d49e7e014a6a7f5d0547cba87d6e4317"
HAMER_REVISION = "3a01849f4148352e9260b69bf28b65d1671a4905"
HAMER_SOURCE_URL = f"https://codeload.github.com/geopavlakos/hamer/zip/{HAMER_REVISION}"
HAMER_SOURCE_SHA256 = "fba22b57f7a89a0af981ee064b06603a4c527bfd347331a0b369e5e4dcba5194"
SAM_REPO = "facebook/sam-3d-body-dinov3"
VGGT_REPO = "facebook/VGGT-Omega"
# Optional SAM 2.1 person segmentation used by the crossing optimisation.
SAM2_REPO = "KyanChen/sam2.1-hiera-small"
SAM2_CODE_REPO = "facebookresearch/sam2"
SAM2_SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_SOURCE_URL = f"https://codeload.github.com/{SAM2_CODE_REPO}/zip/{SAM2_SOURCE_REVISION}"
SAM2_SOURCE_SHA256 = "fe93082a71a885a427894b1eab76341768781b6b58a298a0717e03862097d137"
SAM2_SOURCE_SIZE = 55952852
GEOCALIB_URL = "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar"

# The publishers' ModelScope files match the original model weights byte for byte.
PERCEPTION_DOWNLOADS = (
    (SAM_REPO, "model.ckpt", files.SAM3D_FILES[0], 2109129346,
     "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"),
    (SAM_REPO, "model_config.yaml", files.SAM3D_FILES[1], 1488,
     "1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d"),
    (SAM_REPO, "assets/mhr_model.pt", files.SAM3D_FILES[2], 696110248,
     "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"),
    (VGGT_REPO, "vggt_omega_1b_512.pt", files.VGGT, 4576706117,
     "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"),
)


def modelscope_file(repo, name):
    return f"{MODELSCOPE}/models/{repo}/resolve/master/{name}"


@dataclass(frozen=True)
class Asset:
    relative: str
    size: int
    sha256: str | None
    urls: tuple[str, ...] = ()


def _flowhmr_asset(relative, size, sha256):
    suffix = f"/{FLOWHMR_REPO}/resolve/{FLOWHMR_REVISION}/{relative}"
    return Asset(relative, size, sha256, (HF_MIRROR + suffix, HF_OFFICIAL + suffix))


_perception_assets = tuple(Asset(target, size, digest, (modelscope_file(repo, name),))
                           for repo, name, target, size, digest in PERCEPTION_DOWNLOADS)
PUBLIC_ASSETS = {
    "geocalib": (Asset(files.GEOCALIB, 116074121,
        "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac", (GEOCALIB_URL,)),),
    "flowhmr": (
        _flowhmr_asset(files.FLOWHMR_CONFIG, 3067, "5deae01800c031975558bf9675819e8d1e24803414f692b8b5c2196b266c64d7"),
        _flowhmr_asset(files.FLOWHMR_CKPT, 1841702021, "b7eb06b119feaf5ad7287bafdb2fff38d42dfc22ea9d256f8cdd2ada26dfc53a"),
    ),
    # Optional base checkpoint for the torso-motion optimization (same official repo).
    "flowhmr_base": (
        _flowhmr_asset(files.FLOWHMR_BASE_CONFIG, 2227, "2dc27e1216606fe0d50882f3bc92603004974acaccf4919b159908312a2f491e"),
        _flowhmr_asset(files.FLOWHMR_BASE_CKPT, 1841700869, "77b903ba545669bafd6f7709c0f1748f5fa030a4f25f67392c8c082a21f5e672"),
    ),
    "yolox": (Asset(files.YOLOX, 434357141, "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917", (YOLOX_URL,)),),
    "sam": _perception_assets[:3],
    "vggt": _perception_assets[3:],
    "joint-regressor": (Asset(files.JOINT_REGRESSOR, 662187, "70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf", (REGRESSOR_URL,)),),
    # Optional SAM 2.1 checkpoint for the experimental crossing optimisation.
    "sam2": (Asset(files.SAM2_CROSSING, 184416285,
                   "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
                   (modelscope_file(SAM2_REPO, "sam2.1_hiera_small.pt"),)),),
}

HAMER_CODE = Asset(".downloads/hamer-source.zip", 2314326, HAMER_SOURCE_SHA256, (HAMER_SOURCE_URL,))
# Optional SAM 2.1 inference code, installed on demand like the HaMeR component.
SAM2_CODE = Asset(".downloads/sam2-source.zip", SAM2_SOURCE_SIZE, SAM2_SOURCE_SHA256, (SAM2_SOURCE_URL,))
HAMER_ARCHIVE = Asset(".downloads/hamer_demo_data.tar.gz", 6037554929, None, (
    HAMER_ARCHIVE_URL,
    "https://drive.usercontent.google.com/download?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT&export=download&confirm=t",
))
HAMER_MEMBERS = {
    "_DATA/hamer_ckpts/checkpoints/hamer.ckpt": Asset(
        files.HAMER_FILES[0], 2689536166, "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53"),
    "_DATA/hamer_ckpts/model_config.yaml": Asset(
        files.HAMER_FILES[1], 2072, "0e5eeb82752e47dfd01db8e13ccc4c5eba9bf83f53da8285523b8d3e87247aa3"),
    "_DATA/data/mano_mean_params.npz": Asset(
        files.HAMER_FILES[2], 1178, "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"),
}


def _hamer_direct_asset(asset):
    # The mirror's three files were compared byte for byte with the author
    # package. Always verify the pinned official digest before installing.
    name = asset.relative.rsplit("/", 1)[-1]
    suffix = f"/{HAMER_MIRROR_REPO}/resolve/{HAMER_MIRROR_REVISION}/{name}"
    return Asset(asset.relative, asset.size, asset.sha256, (HF_MIRROR + suffix, HF_OFFICIAL + suffix))


HAMER_DIRECT_ASSETS = tuple(_hamer_direct_asset(asset) for asset in HAMER_MEMBERS.values())
