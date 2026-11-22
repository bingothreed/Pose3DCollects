# TDPT models (placeholder)

Pose3D Collects uses this directory by default; the path is configurable in
**Settings & Resources → TDPT → 模型目录**.  Model files are gitignored.

| File | Size | Source / note |
|---|---|---|
| `MobileNet3D3_6.onnx` | ~33 MB | public `digital-standard/ThreeDPoseTracker`; can be fetched with **Auto Download Models** |
| `ThreeDPoseTracker_3frames.onnx` | ~112 MB | ThreeDPoseLibrary / SDK; **not** auto-downloaded, place it manually (non-commercial license) |

The engine reads the model selected in the TDPT settings card.  See
`../../tdpt/UPSTREAM.md` for the model conversion notes.
