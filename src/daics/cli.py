from __future__ import annotations

import argparse

from daics.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="daics")
    ap.add_argument("--config", required=True, help="configs/base.yaml")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show-config", help="Print parsed config")

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "show-config":
        print(cfg)
        return


if __name__ == "__main__":
    main()
