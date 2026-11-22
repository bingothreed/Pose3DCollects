# TDPT models (placeholder)

Pose3D Collects uses this directory by default; the path is configurable in
**Settings & Resources → TDPT → 模型目录**.  Model files are gitignored.

The two VNect models are interchangeable quality tiers — **either one is
enough to run the engine**, pick the one that suits your speed / accuracy
needs.  If the selected file is missing, the engine falls back to the other
one when it is available.

| File | Size | Tier | Source / note |
|---|---|---|---|
| `MobileNet3D3_6.onnx` | ~33 MB | Fast (single-frame 224) | public `digital-standard/ThreeDPoseTracker`; can be fetched with **Download public single-frame model** |
| `ThreeDPoseTracker_3frames.onnx` | ~112 MB | High precision (3-frame 448) | ThreeDPoseLibrary / SDK; **not** auto-downloaded, place it manually (non-commercial license) |

The engine reads the model selected in the TDPT settings card.  See
`../../tdpt/UPSTREAM.md` for the model conversion notes.
