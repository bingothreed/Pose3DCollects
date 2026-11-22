"""Install the standalone SAM 2.1 inference component (code only, like HaMeR)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flowhmr.model_downloads import download_asset  # noqa: E402
from flowhmr.model_sources import SAM2_CODE  # noqa: E402
from flowhmr.sam2_setup import install_code, runtime_installed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    if args.source_archive:
        install_code(args.source_archive)
    elif not runtime_installed():
        source = download_asset(SAM2_CODE, lambda *_: None, lambda: None)
        try:
            install_code(source)
        finally:
            source.unlink(missing_ok=True)
    print("SAM 2.1 inference component is ready.", flush=True)


if __name__ == "__main__":
    main()
