# FlowHMR source snapshot

- Upstream: FlowHMR Windows portable distribution, version `0.3.8`
  (`FlowHMR-Version.txt`)
- Embedded: 2026-11-22
- License: Apache-2.0 (`LICENSE`); bundled components are listed in
  `THIRD_PARTY_NOTICES.md` and each `third_party/*` directory.

Included: `flowhmr/`, `tools/`, `third_party/`, `app.py`, `configs/`, `docs/`,
runtime assets (`assets/person_reid`, example clip, motion stats), requirement
files and license/notice files.

Excluded (gitignored, prepared locally): `runtime/` (portable Python + torch),
`models/` (~20 GB), `.cache/`, `logs/`, `output/`, `data/rendering_data`.

Models are downloaded by `tools/windows/download_models.py` into
`engines/flowhmr/models/` by default.  The environment setup task creates
`.venv/` with Python 3.10 and installs `requirements-windows.txt`.

Local patch: `flowhmr/model_files.py` now lets the host set
`FLOWHMR_MODELS_DIR`; Pose3D Collects passes the user-configurable model
directory through that variable when starting the server and the download
task.  Re-apply this one-line patch when refreshing the FlowHMR snapshot.
