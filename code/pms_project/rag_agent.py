"""Grounded trip-planning assistant over routes.csv."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from heapq import heappop, heappush

import pandas as pd

from pms_project.analytics import format_seconds, prepare_routes, transition_table


SYSTEM_PROMPT = (
    "You are an assistant for CDA routes. Use ONLY the provided routes.csv. "
    "If a stop is not in the file, say 'Stop not found in current dataset'."
)


@dataclass(frozen=True)
class RouteEdge:
    """Single directed stop transition on one route."""

    source: str
    target: str
    route_id: str
    duration_seconds: float


@dataclass(frozen=True)
class RouteStep:
    """One step in a grounded route option."""

    source: str
    target: str
    route_id: str
    duration_seconds: float


@dataclass(frozen=True)
class PathOption:
    """A grounded route option between two stops."""

    stops: list[str]
    routes: list[str]
    steps: list[RouteStep]
    total_seconds: float


class GroundedRouteAgent:
    """Route planner that only answers from the supplied routes dataframe."""

    def __init__(self, routes: pd.DataFrame, use_llm: bool = True) -> None:
        self.routes = routes.copy()
        self.transitions = transition_table(self.routes)
        self.route_edges = route_edges(self.routes)
        self.stop_names = sorted(self.routes["stop_name"].dropna().unique())
        self.stop_lookup = {normalize(stop): stop for stop in self.stop_names}
        self.use_llm = use_llm

    def answer(self, query: str) -> str:
        """Answer a trip-planning query without inventing stops."""

        mentioned = self.find_mentioned_stops(query)
        lower_query = query.casefold()

        if "which route" in lower_query and mentioned:
            return self.routes_through_stop(mentioned[0])
        if "last bus" in lower_query and mentioned:
            return self.last_departure_from_stop(mentioned[0])
        if len(mentioned) >= 2:
            option = self.shortest_path(mentioned[0], mentioned[1])
            if option is None:
                return (
                    f"I could not find a route path from {mentioned[0]} to "
                    f"{mentioned[1]} in routes.csv."
                )
            grounded = self.path_response(option)
            return self.polish_with_llm(query, grounded)

        return "Stop not found in current dataset"

    def find_mentioned_stops(self, query: str) -> list[str]:
        """Find stop names that occur in a user query."""

        normalized_query = normalize(query)
        matches: list[tuple[int, int, str]] = []
        for key, stop in self.stop_lookup.items():
            if not key:
                continue
            position = normalized_query.find(key)
            if position >= 0:
                matches.append((position, -len(key), stop))

        matches.sort()
        return dedupe([match[2] for match in matches])

    def routes_through_stop(self, stop: str) -> str:
        """List routes that include a stop."""

        rows = self.routes[self.routes["stop_name"] == stop]
        route_ids = sorted(rows["route_id"].unique())
        if not route_ids:
            return f"{stop} was not found in routes.csv."
        return f"{stop} is served by: {', '.join(route_ids)}."

    def last_departure_from_stop(self, stop: str) -> str:
        """Return the latest scheduled departure for a stop."""

        rows = self.routes[self.routes["stop_name"] == stop]
        if rows.empty:
            return f"{stop} was not found in routes.csv."
        latest = rows.sort_values("departure_time").iloc[-1]
        return (
            f"The latest departure from {stop} is {latest['departure_time']} "
            f"on route {latest['route_id']}."
        )

    def shortest_path(self, origin: str, destination: str) -> PathOption | None:
        """Find a lowest-duration path through route-specific process edges."""

        graph: dict[str, list[RouteEdge]] = defaultdict(list)
        for edge in self.route_edges:
            graph[edge.source].append(edge)

        counter = count()
        queue: list[
            tuple[float, int, str, str, list[str], list[RouteStep]]
        ] = []
        heappush(queue, (0.0, next(counter), 0.0, origin, "", [origin], []))
        best: dict[tuple[str, str], float] = {(origin, ""): 0.0}

        while queue:
            priority, _, total_seconds, node, active_route, stops, steps = heappop(queue)
            if node == destination:
                routes = [step.route_id for step in steps]
                return PathOption(
                    stops=stops,
                    routes=routes,
                    steps=steps,
                    total_seconds=total_seconds,
                )

            if best.get((node, active_route), float("inf")) < priority:
                continue

            for edge in graph.get(node, []):
                transfer_penalty = 600.0 if active_route and edge.route_id != active_route else 0.0
                next_priority = priority + edge.duration_seconds + transfer_penalty
                next_total_seconds = total_seconds + edge.duration_seconds
                state = (edge.target, edge.route_id)
                if best.get(state, float("inf")) <= next_priority:
                    continue
                best[state] = next_priority
                heappush(
                    queue,
                    (
                        next_priority,
                        next(counter),
                        next_total_seconds,
                        edge.target,
                        edge.route_id,
                        stops + [edge.target],
                        steps + [
                            RouteStep(
                                source=edge.source,
                                target=edge.target,
                                route_id=edge.route_id,
                                duration_seconds=edge.duration_seconds,
                            )
                        ],
                    ),
                )
        return None

    def path_response(self, option: PathOption) -> str:
        """Build a factual route response from computed path data."""

        origin = option.stops[0]
        destination = option.stops[-1]
        route_summary = summarize_route_changes(option.routes)
        next_departure = self.next_departure(
            origin,
            option.routes[0] if option.routes else None,
        )
        path = " -> ".join(option.stops)
        transfers = max(0, len(dedupe_consecutive(option.routes)) - 1)

        return (
            f"Available path from {origin} to {destination}: {path}. "
            f"Route(s): {route_summary}. Transfers required: {transfers}. "
            f"Estimated travel time: {format_seconds(option.total_seconds)}. "
            f"Next available departure from {origin}: {next_departure}."
        )

    def next_departure(self, stop: str, route_id: str | None) -> str:
        """Return the first schedule departure for the selected route and stop."""

        rows = self.routes[self.routes["stop_name"] == stop]
        if route_id:
            rows = rows[rows["route_id"] == route_id]
        if rows.empty:
            return "not available in routes.csv"
        return str(rows.sort_values("departure_time").iloc[0]["departure_time"])

    def polish_with_llm(self, query: str, grounded_answer: str) -> str:
        """Optionally use OpenAI through LangChain while preserving grounding."""

        llm_config = resolve_llm_config()
        if not self.use_llm or llm_config is None:
            return grounded_answer

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return grounded_answer

        llm = ChatOpenAI(**llm_config, temperature=0)
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            "Rewrite the grounded "
            "answer clearly and briefly. Do not add stops, routes, times, or "
            "transfers that are not present in the grounded answer.\n\n"
            f"User query: {query}\n"
            f"Grounded answer: {grounded_answer}"
        )
        try:
            response = llm.invoke(prompt)
        except Exception:
            return grounded_answer
        return str(response.content)


def resolve_llm_config() -> dict[str, str] | None:
    """Resolve OpenAI-compatible LLM configuration from environment."""

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return {
            "api_key": groq_key,
            "base_url": os.getenv(
                "GROQ_BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {
            "api_key": openai_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        }

    return None


def normalize(value: str) -> str:
    """Normalize text for conservative stop matching."""

    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def route_edges(routes: pd.DataFrame) -> list[RouteEdge]:
    """Build route-specific edges from the discovered process map data."""

    prepared = prepare_routes(routes)
    edges: list[RouteEdge] = []
    for case_id, group in prepared.groupby("case_id", sort=False):
        ordered = group.sort_values("event_index").reset_index(drop=True)
        route_id = str(ordered.iloc[0]["route_id"])
        for index in range(len(ordered) - 1):
            current = ordered.iloc[index]
            next_row = ordered.iloc[index + 1]
            duration = next_row["arrival_delta"] - current["departure_delta"]
            if duration.total_seconds() < 0:
                duration += pd.Timedelta(days=1)
            edges.append(
                RouteEdge(
                    source=str(current["stop_name"]),
                    target=str(next_row["stop_name"]),
                    route_id=route_id,
                    duration_seconds=float(duration.total_seconds()),
                )
            )

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for edge in edges:
        grouped[(edge.source, edge.target, edge.route_id)].append(
            edge.duration_seconds
        )

    return [
        RouteEdge(
            source=source,
            target=target,
            route_id=route_id,
            duration_seconds=sum(values) / len(values),
        )
        for (source, target, route_id), values in grouped.items()
    ]


def dedupe(values: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def summarize_route_changes(routes: list[str]) -> str:
    """Summarize route sequence and transfers."""

    if not routes:
        return "not available"
    compressed = dedupe_consecutive(routes)
    if len(compressed) == 1:
        return compressed[0]
    return " -> ".join(compressed) + f" ({len(compressed) - 1} transfer(s))"


def dedupe_consecutive(values: list[str]) -> list[str]:
    """Remove repeated adjacent route ids."""

    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result
