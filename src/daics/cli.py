from __future__ import annotations

import argparse
from pathlib import Path

from daics.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAICS-SWaT project CLI")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Repo root path")
    args = parser.parse_args()

    log.info("DAICS repo is set up ✅")
    log.info("Project root: %s", args.project_root.resolve())


if __name__ == "__main__":
    main()
