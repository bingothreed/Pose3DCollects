# Embedded engine sources

Pose3D Collects keeps every engine runner inside this directory so a checkout is
self-contained.  Large artifacts stay out of git and are produced locally:

| Engine | Source | Models / environment (ignored) |
|---|---|---|
| InstantHMR | `engines/instanthmr/` | `models/`, `.venv/` |
| FlowHMR | `engines/flowhmr/` | `models/`, `runtime/`, `.venv/`, caches/logs |
| GEM-X | `engines/gemx/` | `weights/`, `assets/soma/`, `.venv/` |
| TDPT | `engines/tdpt/` | `models/`, `.venv/` |

Rules:

- Source snapshots are tracked; do not edit upstream code here unless the change
  belongs to this project.  Record updates in each engine's `UPSTREAM.md`.
- Model weights, virtual environments, caches and logs are gitignored and are
  prepared by the app's setup/download tasks (or copied in manually).
- Every model directory keeps a tracked `README.md` placeholder describing the
  expected files.  The default location is inside `engines/<id>/`, but the
  settings page can point the engine at another model directory.
- Keep the upstream `LICENSE` / `NOTICE` / third-party license files with each
  snapshot.  All four engines are Apache-2.0 at the top level; bundled
  third-party components keep their own terms.
