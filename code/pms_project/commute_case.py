"""Specific bonus commute case for Hussnain Haider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pms_project.analytics import format_seconds, transition_table
from pms_project.rag_agent import GroundedRouteAgent


HUSSNAIN_CASE_ID = "HUSSNAIN-G13-FAST"
HUSSNAIN_MEMBER_NAME = "Hussnain Haider"
HUSSNAIN_MEMBER_ID = "i23-0695"
HOME_AREA = "G-13/2"
HOME_NEAREST_STOP = "NUST Metro Station"
DESTINATION_AREA = "H-11/4 (FAST-NU)"
DESTINATION_STOP = "FAST University"
COMMUTE_ROUTE_ID = "FR-01"


@dataclass(frozen=True)
class HussnainCommute:
    """Report-ready commute summary."""

    case_id: str
    member_name: str
    member_id: str
    home_area: str
    home_nearest_stop: str
    destination_area: str
    destination_stop: str
    route_id: str
    path: list[str]
    estimated_seconds: float

    @property
    def estimated_time(self) -> str:
        """Human-readable transition-duration estimate."""

        return format_seconds(self.estimated_seconds)


def build_hussnain_commute(routes: pd.DataFrame) -> HussnainCommute:
    """Build Hussnain's G-13/2 to H-11/4 FAST-NU commute case."""

    required_stops = {HOME_NEAREST_STOP, DESTINATION_STOP}
    available_stops = set(routes["stop_name"].dropna())
    missing = required_stops - available_stops
    if missing:
        raise ValueError(
            "Stop not found in current dataset: "
            + ", ".join(sorted(missing))
        )

    agent = GroundedRouteAgent(routes, use_llm=False)
    option = agent.shortest_path(HOME_NEAREST_STOP, DESTINATION_STOP)
    if option is None:
        raise ValueError(
            f"No forward-pass path from {HOME_NEAREST_STOP} "
            f"to {DESTINATION_STOP} in routes.csv."
        )

    return HussnainCommute(
        case_id=HUSSNAIN_CASE_ID,
        member_name=HUSSNAIN_MEMBER_NAME,
        member_id=HUSSNAIN_MEMBER_ID,
        home_area=HOME_AREA,
        home_nearest_stop=HOME_NEAREST_STOP,
        destination_area=DESTINATION_AREA,
        destination_stop=DESTINATION_STOP,
        route_id=COMMUTE_ROUTE_ID,
        path=option.stops,
        estimated_seconds=option.total_seconds,
    )


def hussnain_commute_trace(routes: pd.DataFrame) -> pd.DataFrame:
    """Return a case-style trace with cumulative transition durations."""

    commute = build_hussnain_commute(routes)
    transitions = transition_table(
        routes[routes["route_id"] == commute.route_id].copy()
    )

    rows: list[dict[str, object]] = []
    cumulative_seconds = 0.0
    for order, stop in enumerate(commute.path, start=1):
        previous_stop = commute.path[order - 2] if order > 1 else ""
        transition_seconds = 0.0
        if previous_stop:
            match = transitions[
                (transitions["source"] == previous_stop)
                & (transitions["target"] == stop)
            ]
            if match.empty:
                raise ValueError(
                    f"Missing transition in event log: {previous_stop} -> {stop}"
                )
            transition_seconds = float(match.iloc[0]["avg_duration_seconds"])
            cumulative_seconds += transition_seconds

        rows.append(
            {
                "case_id": commute.case_id,
                "member_name": commute.member_name,
                "member_id": commute.member_id,
                "home_area": commute.home_area,
                "home_nearest_stop": commute.home_nearest_stop,
                "destination_area": commute.destination_area,
                "destination_stop": commute.destination_stop,
                "route_id": commute.route_id,
                "event_order": order,
                "stop_name": stop,
                "previous_stop": previous_stop,
                "transition_seconds_from_previous": transition_seconds,
                "cumulative_seconds": cumulative_seconds,
                "cumulative_time": format_seconds(cumulative_seconds),
            }
        )

    return pd.DataFrame(rows)


def write_hussnain_commute_case(
    routes_csv: Path,
    output_path: Path,
) -> Path:
    """Write Hussnain's personal commute trace for report evidence."""

    routes = pd.read_csv(routes_csv)
    trace = hussnain_commute_trace(routes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_path, index=False)
    return output_path

