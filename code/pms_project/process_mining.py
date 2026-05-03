"""Process discovery and graph rendering helpers."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from pms_project.analytics import transition_table
from pms_project.config import DEFAULT_SERVICE_DATE, DEFAULT_TIMEZONE
from pms_project.xes_logging import load_routes_for_xes


import streamlit as st

@st.cache_resource
def discover_process_model(
    routes_csv: Path,
    miner: str = "heuristic",
    service_date: str = DEFAULT_SERVICE_DATE,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> object:
    """Discover a process model from the merged event log with PM4Py."""

    import pm4py

    frame = load_routes_for_xes(routes_csv, service_date, timezone_name)
    frame = frame.rename(
        columns={
            "case_id": "case:concept:name",
            "stop_name": "concept:name",
            "arrival_timestamp": "time:timestamp",
        }
    )
    event_log = pm4py.convert_to_event_log(
        pm4py.format_dataframe(
            frame,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )
    )

    miner_key = miner.strip().casefold()
    if miner_key.startswith("inductive"):
        return pm4py.discover_petri_net_inductive(event_log)
    if miner_key.startswith("heur"):
        return pm4py.discover_heuristics_net(event_log)
    raise ValueError("miner must be 'inductive' or 'heuristic'")


def filter_routes(frame: pd.DataFrame, route_id: str) -> pd.DataFrame:
    """Filter route rows for the GUI route selector."""

    if route_id == "All Routes":
        return frame.copy()
    return frame[frame["route_id"] == route_id].copy()


def build_process_dot(
    frame: pd.DataFrame,
    threshold_seconds: float,
) -> str:
    """Build a Graphviz DOT directly-follows map with edge annotations."""

    transitions = transition_table(frame)
    nodes = sorted(set(transitions["source"]).union(set(transitions["target"])))
    lines = [
        "digraph G {",
        "rankdir=TB;",
        'graph [bgcolor="transparent", pad="0.5", nodesep="0.6", ranksep="0.9", splines="polyline"];',
        'node [shape=box, style="rounded,filled", color="#3A4750", '
        'fillcolor="#F7F9FB", fontname="Arial", fontsize=12, margin="0.25,0.15"];',
        'edge [fontname="Arial", fontsize=10, color="#4E6E81", arrowsize=0.8];',
    ]

    for node in nodes:
        lines.append(f'"{escape_dot(node)}";')

    for row in transitions.itertuples(index=False):
        color = "#C62828" if row.avg_duration_seconds > threshold_seconds else "#4E6E81"
        penwidth = "2.4" if row.avg_duration_seconds > threshold_seconds else "1.3"
        label = (
            f"{row.duration_label}\\n"
            f"cases: {row.case_frequency}"
        )
        lines.append(
            f'"{escape_dot(row.source)}" -> "{escape_dot(row.target)}" '
            f'[label="{escape_dot(label)}", color="{color}", '
            f'fontcolor="{color}", penwidth="{penwidth}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def escape_dot(value: object) -> str:
    """Escape labels for DOT output."""

    return html.escape(str(value), quote=True).replace("\n", "\\n")

