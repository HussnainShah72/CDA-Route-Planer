"""Create a strict XES event log from routes.csv."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd

from pms_project.case_inference import add_trip_case_ids
from pms_project.config import DEFAULT_SERVICE_DATE, DEFAULT_TIMEZONE, REQUIRED_ROUTE_COLUMNS


def create_xes_log(
    routes_csv: Path,
    output_path: Path,
    service_date: str = DEFAULT_SERVICE_DATE,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> Path:
    """Convert routes.csv into a strict XES 1.0 event log.

    The writer is intentionally explicit instead of relying on PM4Py's exporter,
    because the assignment requires exact root metadata and Islamabad +05:00
    timestamps in the serialized file.
    """

    events = load_routes_for_xes(routes_csv, service_date, timezone_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_strict_xes(events, output_path)
    return output_path


def write_strict_xes(events: pd.DataFrame, output_path: Path) -> None:
    """Write XES XML with exact assignment-required metadata."""

    namespace = "http://www.xes-standard.org/"
    ET.register_namespace("", namespace)
    log = ET.Element(
        f"{{{namespace}}}log",
        {
            "xes.version": "1.0",
            "xes.features": "",
            "xes.author": "Hussnain Haider Group",
        },
    )
    ET.SubElement(log, f"{{{namespace}}}extension", {
        "name": "Concept",
        "prefix": "concept",
        "uri": "http://www.xes-standard.org/concept.xesext",
    })
    ET.SubElement(log, f"{{{namespace}}}extension", {
        "name": "Time",
        "prefix": "time",
        "uri": "http://www.xes-standard.org/time.xesext",
    })
    ET.SubElement(log, f"{{{namespace}}}extension", {
        "name": "Lifecycle",
        "prefix": "lifecycle",
        "uri": "http://www.xes-standard.org/lifecycle.xesext",
    })
    ET.SubElement(log, f"{{{namespace}}}global", {"scope": "trace"})
    ET.SubElement(log, f"{{{namespace}}}global", {"scope": "event"})
    ET.SubElement(log, f"{{{namespace}}}classifier", {
        "name": "Activity",
        "keys": "concept:name",
    })
    ET.SubElement(log, f"{{{namespace}}}string", {
        "key": "concept:name",
        "value": "CDA Bus Forward Routes",
    })
    ET.SubElement(log, f"{{{namespace}}}string", {
        "key": "source",
        "value": "cda.gov.pk/cdaTransitMap",
    })

    ordered = events.sort_values(["case_id", "event_index"])
    for case_id, group in ordered.groupby("case_id", sort=False):
        trace = ET.SubElement(log, f"{{{namespace}}}trace")
        ET.SubElement(trace, f"{{{namespace}}}string", {
            "key": "concept:name",
            "value": str(case_id),
        })
        ET.SubElement(trace, f"{{{namespace}}}string", {
            "key": "route_id",
            "value": str(group.iloc[0]["route_id"]),
        })
        for row in group.itertuples(index=False):
            event = ET.SubElement(trace, f"{{{namespace}}}event")
            ET.SubElement(event, f"{{{namespace}}}string", {
                "key": "concept:name",
                "value": str(row.stop_name),
            })
            ET.SubElement(event, f"{{{namespace}}}date", {
                "key": "time:timestamp",
                "value": row.arrival_timestamp.isoformat(),
            })
            ET.SubElement(event, f"{{{namespace}}}string", {
                "key": "lifecycle:transition",
                "value": "complete",
            })
            ET.SubElement(event, f"{{{namespace}}}string", {
                "key": "route_id",
                "value": str(row.route_id),
            })
            ET.SubElement(event, f"{{{namespace}}}string", {
                "key": "departure_time",
                "value": row.departure_timestamp.isoformat(),
            })

    rough_xml = ET.tostring(log, encoding="utf-8")
    pretty_xml = minidom.parseString(rough_xml).toprettyxml(
        indent="  ",
        encoding="utf-8",
    )
    output_path.write_bytes(pretty_xml)


def load_routes_for_xes(
    routes_csv: Path,
    service_date: str = DEFAULT_SERVICE_DATE,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Load routes.csv and add trace ids plus timezone-aware timestamps."""

    frame = pd.read_csv(routes_csv)
    missing = set(REQUIRED_ROUTE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{routes_csv} is missing columns: {sorted(missing)}")

    frame = add_trip_case_ids(frame)
    frame["event_index"] = frame.groupby("case_id").cumcount()

    service_day = date.fromisoformat(service_date)
    timezone = ZoneInfo(timezone_name)
    frame = add_monotonic_route_timestamps(frame, service_day, timezone)
    return frame


def add_monotonic_route_timestamps(
    frame: pd.DataFrame,
    service_day: date,
    timezone: ZoneInfo,
) -> pd.DataFrame:
    """Convert route time strings to ISO-ready datetimes per route."""

    frames: list[pd.DataFrame] = []
    for _, group in frame.groupby("case_id", sort=False):
        group = group.copy()
        day_offset = 0
        previous_arrival: datetime | None = None
        arrivals: list[datetime] = []
        departures: list[datetime] = []

        for row in group.itertuples(index=False):
            arrival = combine_time(service_day, row.arrival_time, timezone, day_offset)
            if previous_arrival and arrival < previous_arrival:
                day_offset += 1
                arrival = combine_time(service_day, row.arrival_time, timezone, day_offset)

            departure = combine_time(service_day, row.departure_time, timezone, day_offset)
            if departure < arrival:
                departure += timedelta(days=1)

            arrivals.append(arrival)
            departures.append(departure)
            previous_arrival = arrival

        group["arrival_timestamp"] = arrivals
        group["departure_timestamp"] = departures
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def combine_time(
    service_day: date,
    value: str,
    timezone: ZoneInfo,
    day_offset: int = 0,
) -> datetime:
    """Combine a schedule time value with a service date and timezone."""

    parsed = time.fromisoformat(ensure_seconds(value))
    return datetime.combine(
        service_day + timedelta(days=day_offset),
        parsed,
        timezone,
    )


def ensure_seconds(value: str) -> str:
    """Normalize HH:MM strings to HH:MM:SS."""

    parts = str(value).strip().split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    if len(parts) == 3:
        return str(value).strip()
    raise ValueError(f"Invalid time value: {value}")
