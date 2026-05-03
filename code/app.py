"""Streamlit GUI for CDA route process mining and trip planning."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from pms_project.analytics import (  # noqa: E402
    format_seconds,
    throughput_summary,
    top_bottlenecks,
    transition_table,
)
from pms_project.config import ROUTES_CSV, XES_LOG  # noqa: E402
from pms_project.commute_case import (  # noqa: E402
    build_hussnain_commute,
    hussnain_commute_trace,
)
from pms_project.personal_maps import (  # noqa: E402
    build_personal_dot,
    build_personal_options,
    load_personal_routes,
    personal_option_table,
)
from pms_project.process_mining import (  # noqa: E402
    build_process_dot,
    discover_process_model,
    filter_routes,
)
from pms_project.rag_agent import GroundedRouteAgent  # noqa: E402
from pms_project.xes_logging import create_xes_log  # noqa: E402


def load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE pairs when python-dotenv is unavailable."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    else:
        load_dotenv(path)


load_local_env(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    """Render the Streamlit app."""

    st.set_page_config(page_title="CDA Process Mining", layout="wide")
    st.title("CDA Bus Route Process Mining")

    if not ROUTES_CSV.exists():
        st.error(
            "data/routes.csv was not found. Run "
            "`python code/run_pipeline.py extract --limit 8` first."
        )
        st.stop()

    routes = load_routes(ROUTES_CSV)
    route_options = ["All Routes"] + sorted(routes["route_id"].unique())

    with st.sidebar:
        st.header("Controls")
        selected_route = st.selectbox("Route filter", route_options)
        miner = st.selectbox("Discovery miner", ["Heuristic Miner", "Inductive Miner"])
        threshold_minutes = st.slider(
            "Bottleneck threshold (minutes)",
            min_value=1,
            max_value=60,
            value=8,
        )
        if st.button("Export XES log"):
            path = create_xes_log(ROUTES_CSV, XES_LOG)
            st.success(f"Wrote {path}")
        llm_key = st.text_input(
            "LLM API key",
            type="password",
            value=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        )
        if llm_key:
            if llm_key.startswith("gsk_"):
                os.environ["GROQ_API_KEY"] = llm_key
            else:
                os.environ["OPENAI_API_KEY"] = llm_key

    filtered = filter_routes(routes, selected_route)
    threshold_seconds = threshold_minutes * 60

    summary = throughput_summary(filtered)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg throughput", format_seconds(summary.average_seconds))
    col_b.metric("Min throughput", format_seconds(summary.minimum_seconds))
    col_c.metric("Max throughput", format_seconds(summary.maximum_seconds))

    map_col, analytics_col = st.columns([2, 1])
    with map_col:
        st.subheader("Process Map")
        st.graphviz_chart(build_process_dot(filtered, threshold_seconds), use_container_width=True)

    with analytics_col:
        st.subheader("Evidence")
        transitions = transition_table(filtered)
        st.caption(f"Miner selected: {miner}")
        validate_miner(ROUTES_CSV, miner)
        st.dataframe(
            transitions[
                [
                    "source",
                    "target",
                    "duration_label",
                    "case_frequency",
                    "routes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Top 3 Slowest Bottlenecks")
    bottlenecks = top_bottlenecks(transitions, threshold_seconds)
    if bottlenecks.empty:
        st.info("No transitions exceed the selected threshold.")
    else:
        st.dataframe(
            bottlenecks[
                [
                    "source",
                    "target",
                    "duration_label",
                    "case_frequency",
                    "routes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Grounded Trip Planner")
    render_chat(routes)

    st.subheader("Personal Route Map Bonus")
    render_personal_maps(routes)
    render_hussnain_commute(routes)


@st.cache_data
def load_routes(path: Path) -> pd.DataFrame:
    """Load routes.csv."""

    return pd.read_csv(path)


@st.cache_resource
def build_agent(_routes: pd.DataFrame) -> GroundedRouteAgent:
    """Cache the grounded route agent."""

    return GroundedRouteAgent(_routes)


def render_chat(routes: pd.DataFrame) -> None:
    """Render the trip-planning chat panel."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "I'm your CDA AI Agent. I have access to your Event Logs and Trip CSV. Ask me anything!"
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about routes, stops, departures, or travel time")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    agent = build_agent(routes)
    answer = agent.answer(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)


def render_personal_maps(routes: pd.DataFrame) -> None:
    """Render bonus personal route-map evidence."""

    personal_path = ROUTES_CSV.parent / "personal_routes.csv"
    personal_routes = load_personal_routes(personal_path)
    if personal_routes.empty:
        st.info("Add data/personal_routes.csv to generate personal route maps.")
        return

    st.caption(
        "Fill home_area and nearest_stop in data/personal_routes.csv, using exact "
        "stop names from routes.csv."
    )
    st.dataframe(personal_routes, use_container_width=True, hide_index=True)

    options = build_personal_options(routes, personal_routes)
    st.dataframe(personal_option_table(options), use_container_width=True, hide_index=True)
    for item in options:
        option = item.get("option")
        if option is None:
            continue
        st.markdown(f"**{item['member_name']}**")
        st.graphviz_chart(build_personal_dot(option), use_container_width=True)


def render_hussnain_commute(routes: pd.DataFrame) -> None:
    """Render Hussnain's explicit G-13/2 to FAST-NU bonus case."""

    commute = build_hussnain_commute(routes)
    trace = hussnain_commute_trace(routes)
    st.markdown("**Hussnain Haider: G-13/2 to H-11/4 (FAST-NU)**")
    st.caption(
        f"Home area {commute.home_area}; nearest CDA stop "
        f"{commute.home_nearest_stop}; destination {commute.destination_area}; "
        f"route {commute.route_id}; estimated transition time "
        f"{commute.estimated_time}."
    )
    st.dataframe(trace, use_container_width=True, hide_index=True)


def validate_miner(routes_csv: Path, miner: str) -> None:
    """Run selected PM4Py miner and show whether discovery succeeds."""

    try:
        discover_process_model(routes_csv, miner)
    except Exception as exc:
        st.warning(f"{miner} could not run yet: {exc}")
    else:
        st.success(f"{miner} discovery completed from the event log.")


if __name__ == "__main__":
    main()
