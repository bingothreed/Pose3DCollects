# GEM-X weights (placeholder)

Pose3D Collects uses this directory by default; the path is configurable in
**Settings & Resources → GEM-X → 权重目录**.  Model files are gitignored; the
expected layout is:

| File | Size | Source |
|---|---|---|
| `body_s240000.ckpt` | ~3.54 GB | Hugging Face `nvidia/GEM-X` |
| `body_config.json` | ~1 KB | Hugging Face `nvidia/GEM-X` |
| `yolox_humanart.onnx` | ~378 MB | OpenMMLab RTMPose YOLOX HumanArt archive (`end2end.onnx`) |
| `yolox_humanart_fp16.onnx` | ~189 MB | optional generated fp16 copy (not required) |

The app's **Auto Download Models** button (`workers/download_gemx_models.py`)
downloads the first three files here.  SOMA assets belong in
`../assets/soma/` (see that README).  Do not commit the weights.
