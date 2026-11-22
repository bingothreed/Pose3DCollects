# GeoCalib inference runtime

Unmodified inference-only files from https://github.com/cvg/GeoCalib at
`1dedb29f94fc11b896c5f572005978aeebccd67f`, under the accompanying Apache-2.0
LICENSE. Training, interactive demo and visualization helpers are excluded.
`manifest.json` records each included file's SHA-256 with the upstream LF line
endings enforced by `.gitattributes`. Original file headers and
third-party attribution are retained.

FlowHMR supplies this source in portable builds and code updates. Runtime
dependencies are installed through `tools/web/requirements.txt` and the Windows
lock file. Research checkouts and `optional_packages` are not runtime inputs.

The Models dialog downloads the author's v1.0 pinhole checkpoint to
`models/geocalib/pinhole.tar` and verifies its pinned size and SHA-256. A folder
containing the official `geocalib-pinhole.tar` can also be imported. FlowHMR
always passes this explicit local checkpoint to GeoCalib; inference never
downloads weights implicitly.
