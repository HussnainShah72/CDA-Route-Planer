"""Command-line pipeline for the CDA process mining project."""

from __future__ import annotations

import argparse
from pathlib import Path

from pms_project.config import ROUTES_CSV, XES_LOG
from pms_project.commute_case import write_hussnain_commute_case
from pms_project.extraction import build_routes_csv
from pms_project.xes_logging import create_xes_log


def main() -> None:
    """Run pipeline subcommands."""

    parser = argparse.ArgumentParser(description="CDA bus route process mining pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Build data/routes.csv")
    extract_parser.add_argument("--source-dir", type=Path, default=None)
    extract_parser.add_argument("--output", type=Path, default=ROUTES_CSV)
    extract_parser.add_argument("--limit", type=int, default=None)

    xes_parser = subparsers.add_parser("xes", help="Build data/cda_bus_routes.xes")
    xes_parser.add_argument("--routes-csv", type=Path, default=ROUTES_CSV)
    xes_parser.add_argument("--output", type=Path, default=XES_LOG)

    commute_parser = subparsers.add_parser(
        "commute",
        help="Build Hussnain's G-13/2 to FAST-NU bonus trace",
    )
    commute_parser.add_argument("--routes-csv", type=Path, default=ROUTES_CSV)
    commute_parser.add_argument(
        "--output",
        type=Path,
        default=ROUTES_CSV.parent / "hussnain_commute_case.csv",
    )

    all_parser = subparsers.add_parser("all", help="Run extraction and XES export")
    all_parser.add_argument("--source-dir", type=Path, default=None)
    all_parser.add_argument("--routes-csv", type=Path, default=ROUTES_CSV)
    all_parser.add_argument("--xes-output", type=Path, default=XES_LOG)
    all_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "extract":
        path = build_routes_csv(args.output, args.source_dir, args.limit)
        print(f"Wrote {path}")
    elif args.command == "xes":
        path = create_xes_log(args.routes_csv, args.output)
        print(f"Wrote {path}")
    elif args.command == "commute":
        path = write_hussnain_commute_case(args.routes_csv, args.output)
        print(f"Wrote {path}")
    elif args.command == "all":
        routes_path = build_routes_csv(args.routes_csv, args.source_dir, args.limit)
        print(f"Wrote {routes_path}")
        xes_path = create_xes_log(routes_path, args.xes_output)
        print(f"Wrote {xes_path}")
        commute_path = write_hussnain_commute_case(
            routes_path,
            routes_path.parent / "hussnain_commute_case.csv",
        )
        print(f"Wrote {commute_path}")


if __name__ == "__main__":
    main()
