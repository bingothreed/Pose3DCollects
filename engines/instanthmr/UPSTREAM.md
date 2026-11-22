# InstantHMR source snapshot

- Upstream: https://github.com/bingothreed/InstantHMR (fork used for this project)
- Commit: `2a02dc84a0cf62fb35042964e35669cef33d0cbb`
- Embedded: 2026-11-22
- License: Apache-2.0 (`LICENSE`); SAM components are covered by `LICENSE-SAM`.

Included: `instanthmr/`, `demo.py`, `requirements.txt`, license files.

Excluded (gitignored, prepared locally): `models/`, `.venv/`, `assets/`,
`exports/`, `pytorch/`, training and benchmark code, demo media.

Models are downloaded by `workers/download_instanthmr_models.py` into
`engines/instanthmr/models/`.
