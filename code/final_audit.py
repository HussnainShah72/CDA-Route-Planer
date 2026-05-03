"""Final compliance audit for the SE4009 CDA process mining project."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from pms_project.analytics import throughput_summary, top_bottlenecks, transition_table
from pms_project.commute_case import build_hussnain_commute, hussnain_commute_trace
from pms_project.rag_agent import GroundedRouteAgent, SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES_CSV = PROJECT_ROOT / "data" / "routes.csv"
XES_LOG = PROJECT_ROOT / "data" / "cda_bus_routes.xes"


def main() -> None:
    """Run final project compliance checks."""

    routes = pd.read_csv(ROUTES_CSV)
    check_routes(routes)
    check_analytics(routes)
    check_xes(XES_LOG)
    check_agent(routes)
    check_hussnain_commute(routes)
    print("FINAL AUDIT PASSED")


def check_routes(routes: pd.DataFrame) -> None:
    """Validate forward-only 8-route dataset shape."""

    required = ["route_id", "stop_name", "arrival_time", "departure_time"]
    assert list(routes.columns) == required, routes.columns.tolist()
    assert routes["route_id"].nunique() == 8, sorted(routes["route_id"].unique())
    forbidden = routes["route_id"].str.contains(
        "back|reverse|return",
        case=False,
        na=False,
    ) | routes["stop_name"].str.contains(
        "backward|reverse|return",
        case=False,
        na=False,
    )
    assert int(forbidden.sum()) == 0, "Dataset contains non-forward rows"


def check_analytics(routes: pd.DataFrame) -> None:
    """Validate throughput and bottleneck calculations."""

    summary = throughput_summary(routes)
    assert summary.average_seconds > 0
    assert summary.minimum_seconds > 0
    assert summary.maximum_seconds >= summary.minimum_seconds

    transitions = transition_table(routes)
    assert {"duration_label", "case_frequency"}.issubset(transitions.columns)
    top_three = top_bottlenecks(transitions, threshold_seconds=0, limit=3)
    assert len(top_three) == 3


def check_xes(path: Path) -> None:
    """Validate strict XES metadata and Islamabad ISO timestamps."""

    root = ET.parse(path).getroot()
    assert root.attrib["xes.version"] == "1.0"
    assert root.attrib["xes.features"] == ""
    assert root.attrib["xes.author"] == "Hussnain Haider Group"

    namespace = "{http://www.xes-standard.org/}"
    timestamps = [
        date.attrib["value"]
        for date in root.findall(f".//{namespace}date")
        if date.attrib.get("key") == "time:timestamp"
    ]
    assert timestamps
    for value in timestamps:
        assert value.endswith("+05:00"), value
        datetime.fromisoformat(value)


def check_agent(routes: pd.DataFrame) -> None:
    """Validate strict no-hallucination guardrails."""

    expected = (
        "You are an assistant for CDA routes. Use ONLY the provided routes.csv. "
        "If a stop is not in the file, say 'Stop not found in current dataset'."
    )
    assert SYSTEM_PROMPT == expected
    agent = GroundedRouteAgent(routes, use_llm=False)
    assert agent.answer("Khanna Pul to Mars") == "Stop not found in current dataset"


def check_hussnain_commute(routes: pd.DataFrame) -> None:
    """Validate Hussnain's G-13/2 to H-11/4 FAST-NU case."""

    commute = build_hussnain_commute(routes)
    assert commute.home_area == "G-13/2"
    assert commute.destination_area == "H-11/4 (FAST-NU)"
    assert commute.home_nearest_stop == "NUST Metro Station"
    assert commute.destination_stop == "FAST University"
    assert commute.route_id == "FR-01"
    assert re.search(r"NUST Metro Station.*FAST University", " ".join(commute.path))

    trace = hussnain_commute_trace(routes)
    assert trace.iloc[0]["case_id"] == "HUSSNAIN-G13-FAST"
    assert trace.iloc[-1]["stop_name"] == "FAST University"
    assert float(trace.iloc[-1]["cumulative_seconds"]) > 0


if __name__ == "__main__":
    main()

