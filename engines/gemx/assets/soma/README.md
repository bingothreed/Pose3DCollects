# SOMA 0.3.3 assets (placeholder)

Pose3D Collects uses this directory by default; the path is configurable in
**Settings & Resources → GEM-X → SOMA 资产目录**.  These files are gitignored.

| File | Size | Source |
|---|---|---|
| `SOMA_neutral.npz` | ~26 MB | Hugging Face `nvidia/SOMA-X`, revision `v0.3.3` |
| `SOMA_template_rig.usda` | ~27 MB | same |
| `SOMA_procedural_transforms.json` | ~20 KB | same |
| `correctives_model.pt` | ~68 MB | same |

The app's **Auto Download Models** button fetches them into this folder.  The
GEM-X checkpoint and detector live in `../weights/`.
