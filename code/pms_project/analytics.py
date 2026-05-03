"""Timing, frequency, and bottleneck analytics for CDA routes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pms_project.case_inference import add_trip_case_ids


@dataclass(frozen=True)
class ThroughputSummary:
    """End-to-end case duration summary."""

    average_seconds: float
    minimum_seconds: float
    maximum_seconds: float


def prepare_routes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return route rows with parsed timedelta columns."""

    prepared = add_trip_case_ids(frame)
    prepared["arrival_delta"] = prepared["arrival_time"].map(parse_time_delta)
    prepared["departure_delta"] = prepared["departure_time"].map(parse_time_delta)
    prepared["event_index"] = prepared.groupby("case_id").cumcount()
    return prepared


def transition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute directly-follows transitions and edge-level statistics."""

    prepared = prepare_routes(frame)
    rows: list[dict[str, object]] = []
    for case_id, group in prepared.groupby("case_id", sort=False):
        ordered = group.sort_values("event_index").reset_index(drop=True)
        route_id = ordered.iloc[0]["route_id"]
        for index in range(len(ordered) - 1):
            current = ordered.iloc[index]
            next_row = ordered.iloc[index + 1]
            duration = next_row["arrival_delta"] - current["departure_delta"]
            if duration.total_seconds() < 0:
                duration += pd.Timedelta(days=1)
            rows.append(
                {
                    "route_id": route_id,
                    "case_id": case_id,
                    "source": current["stop_name"],
                    "target": next_row["stop_name"],
                    "duration_seconds": duration.total_seconds(),
                    "duration_label": format_seconds(duration.total_seconds()),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "avg_duration_seconds",
                "duration_label",
                "case_frequency",
                "routes",
            ]
        )

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["source", "target"], as_index=False)
        .agg(
            avg_duration_seconds=("duration_seconds", "mean"),
            case_frequency=("case_id", "nunique"),
            routes=("route_id", lambda values: ", ".join(sorted(set(values)))),
        )
        .sort_values("avg_duration_seconds", ascending=False)
    )
    grouped["duration_label"] = grouped["avg_duration_seconds"].map(format_seconds)
    return grouped


def throughput_summary(frame: pd.DataFrame) -> ThroughputSummary:
    """Compute average, minimum, and maximum trip throughput per route case."""

    prepared = prepare_routes(frame)
    totals: list[float] = []
    for _, group in prepared.groupby("case_id", sort=False):
        ordered = group.sort_values("event_index")
        first_arrival = ordered.iloc[0]["arrival_delta"]
        last_departure = ordered.iloc[-1]["departure_delta"]
        total = last_departure - first_arrival
        if total.total_seconds() < 0:
            total += pd.Timedelta(days=1)
        totals.append(total.total_seconds())

    if not totals:
        return ThroughputSummary(0.0, 0.0, 0.0)

    series = pd.Series(totals)
    return ThroughputSummary(
        average_seconds=float(series.mean()),
        minimum_seconds=float(series.min()),
        maximum_seconds=float(series.max()),
    )


def top_bottlenecks(
    transitions: pd.DataFrame,
    threshold_seconds: float,
    limit: int = 3,
) -> pd.DataFrame:
    """Return the slowest transitions above a user-defined threshold."""

    if transitions.empty:
        return transitions
    bottlenecks = transitions[
        transitions["avg_duration_seconds"] > threshold_seconds
    ].copy()
    return bottlenecks.sort_values("avg_duration_seconds", ascending=False).head(limit)


def parse_time_delta(value: str) -> pd.Timedelta:
    """Convert HH:MM or HH:MM:SS to a timedelta since midnight."""

    parts = [int(part) for part in str(value).strip().split(":")]
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    elif len(parts) == 3:
        hour, minute, second = parts
    else:
        raise ValueError(f"Invalid time value: {value}")
    return pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def format_seconds(seconds: float) -> str:
    """Format a duration as minutes and seconds."""

    seconds_int = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} hr {minutes} min {seconds_part} sec"
    if minutes:
        return f"{minutes} min {seconds_part} sec"
    return f"{seconds_part} sec"
