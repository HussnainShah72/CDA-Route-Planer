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
    "nearest_stop",
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
    routes: pd.DataFrame,
    personal_routes: pd.DataFrame,
) -> list[dict[str, object]]:
    """Compute grounded path options for filled personal-route rows."""

    agent = GroundedRouteAgent(routes, use_llm=False)
    options: list[dict[str, object]] = []

    for row in personal_routes.itertuples(index=False):
        origin = str(row.origin_stop).strip()
        nearest = str(row.nearest_stop).strip()
        if not origin or not nearest:
            options.append(
                {
                    "member_name": row.member_name,
                    "member_id": row.member_id,
                    "status": "Add origin_stop and nearest_stop",
                    "option": None,
                }
            )
            continue

        option = agent.shortest_path(origin, nearest)
        options.append(
            {
                "member_name": row.member_name,
                "member_id": row.member_id,
                "home_area": row.home_area,
                "origin_stop": origin,
                "nearest_stop": nearest,
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
                "nearest_stop": item.get("nearest_stop", ""),
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
        "rankdir=LR;",
        'graph [bgcolor="transparent", pad="0.2"];',
        'node [shape=box, style="rounded,filled", color="#3A4750", '
        'fillcolor="#F7F9FB", fontname="Arial", fontsize=10];',
        'edge [fontname="Arial", fontsize=9, color="#2E7D32", penwidth=2];',
    ]
    for index, stop in enumerate(option.stops[:-1]):
        target = option.stops[index + 1]
        route = option.routes[index] if index < len(option.routes) else ""
        lines.append(
            f'"{escape_dot(stop)}" -> "{escape_dot(target)}" '
            f'[label="{escape_dot(route)}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def describe_option(option: object) -> str:
    """Describe a personal path option for tables."""

    if not isinstance(option, PathOption):
        return ""
    return " -> ".join(option.stops)

