"""Bonus personal route-map helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pms_project.analytics import format_seconds
from pms_project.process_mining import escape_dot
from pms_project.rag_agent import GroundedRouteAgent, PathOption


PERSONAL_COLUMNS = [
    "member_name",
    "member_id",
    "home_area",
    "origin_stop",
    "destination_stop",
]


def load_personal_routes(path: Path) -> pd.DataFrame:
    """Load the personal route-map template."""

    if not path.exists():
        return pd.DataFrame(columns=PERSONAL_COLUMNS)
    frame = pd.read_csv(path).fillna("")
    missing = set(PERSONAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[PERSONAL_COLUMNS]


def build_personal_options(
    agent: GroundedRouteAgent,
    personal_routes: pd.DataFrame,
) -> list[dict[str, object]]:
    """Compute grounded path options for filled personal-route rows."""

    options: list[dict[str, object]] = []

    for row in personal_routes.itertuples(index=False):
        origin = str(row.origin_stop).strip()
        destination = str(row.destination_stop).strip()
        if not origin or not destination:
            options.append(
                {
                    "member_name": row.member_name,
                    "member_id": row.member_id,
                    "status": "Add origin_stop and destination_stop",
                    "option": None,
                }
            )
            continue

        option = agent.shortest_path(origin, destination)
        options.append(
            {
                "member_name": row.member_name,
                "member_id": row.member_id,
                "home_area": row.home_area,
                "origin_stop": origin,
                "destination_stop": destination,
                "status": "Path found" if option else "No path found in routes.csv",
                "option": option,
            }
        )

    return options


def personal_option_table(options: list[dict[str, object]]) -> pd.DataFrame:
    """Return a report-ready table for personal route-map evidence."""

    rows: list[dict[str, object]] = []
    for item in options:
        option = item.get("option")
        rows.append(
            {
                "member_name": item.get("member_name", ""),
                "member_id": item.get("member_id", ""),
                "home_area": item.get("home_area", ""),
                "origin_stop": item.get("origin_stop", ""),
                "destination_stop": item.get("destination_stop", ""),
                "route_path": describe_option(option),
                "estimated_time": (
                    format_seconds(option.total_seconds)
                    if isinstance(option, PathOption)
                    else ""
                ),
                "status": item.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def build_personal_dot(option: PathOption) -> str:
    """Build a compact Graphviz route map for one member."""

    lines = [
        "digraph G {",
        "rankdir=TB;",
        'graph [bgcolor="#E8DDB4", pad="0.2"];',
        'node [shape=box, style="rounded,filled", color="#DAA464", '
        'fillcolor="#3D3D3D", fontname="Arial", fontsize=11, fontcolor="#E8DDB4"];',
        'edge [fontname="Arial", fontsize=10, fontcolor="black", color="black", penwidth=2.5];',
    ]
    for step in option.steps:
        label = f"{step.route_id} = {format_seconds(step.duration_seconds)}"
        lines.append(
            f'"{escape_dot(step.source)}" -> "{escape_dot(step.target)}" '
            f'[label="{escape_dot(label)}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def describe_option(option: object) -> str:
    """Describe a personal path option for tables."""

    if not isinstance(option, PathOption):
        return ""
    return " -> ".join(option.stops)

