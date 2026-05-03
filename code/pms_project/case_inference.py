"""Infer trip-level cases from route schedule rows."""

from __future__ import annotations

import pandas as pd


def add_trip_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Add case_id values by splitting repeated stop sequences per route."""

    result = frame.copy()
    case_ids: list[str] = []
    trip_numbers: list[int] = []

    for route_id, group in result.groupby("route_id", sort=False):
        origin_stop = str(group.iloc[0]["stop_name"])
        trip_number = 1
        events_in_trip = 0
        previous_arrival: int | None = None

        for row in group.itertuples(index=False):
            arrival_seconds = time_to_seconds(str(row.arrival_time))
            stop_name = str(row.stop_name)
            starts_repeated_trip = events_in_trip > 0 and stop_name == origin_stop
            crosses_midnight = (
                previous_arrival is not None
                and arrival_seconds < previous_arrival
            )
            if starts_repeated_trip or crosses_midnight:
                trip_number += 1
                events_in_trip = 0

            case_ids.append(f"{route_id}-T{trip_number:03d}")
            trip_numbers.append(trip_number)
            events_in_trip += 1
            previous_arrival = arrival_seconds

    result["case_id"] = case_ids
    result["trip_number"] = trip_numbers
    return result


def time_to_seconds(value: str) -> int:
    """Convert HH:MM or HH:MM:SS to seconds since midnight."""

    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    elif len(parts) == 3:
        hour, minute, second = parts
    else:
        raise ValueError(f"Invalid time value: {value}")
    return hour * 3600 + minute * 60 + second

