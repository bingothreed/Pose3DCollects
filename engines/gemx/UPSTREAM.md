# GEM-X source snapshot

- Upstream: NVIDIA GEM-X, package version `0.1.0` (`setup.cfg`)
- Embedded: 2026-11-22
- License: Apache-2.0 (`LICENSE`, `ATTRIBUTIONS.md`, `LICENSES/`).

Included: `gem/`, `pyproject.toml`, `setup.cfg`, `setup.py`, `MANIFEST.in`,
`requirements.txt`, README/changelog/security notes and license files.

Excluded (gitignored, prepared locally): `weights/`, `assets/soma/`, `.venv/`,
demo media, tests, docs and training/benchmark extras.

Models are downloaded by `workers/download_gemx_models.py` into
`engines/gemx/weights/` and `engines/gemx/assets/soma/`.
