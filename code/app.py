"""Streamlit GUI for CDA route process mining and trip planning."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import graphviz

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

    # Inject custom CSS for the requested charcoal and gold palette
    st.markdown("""
        <style>
            /* Main app background */
            .stApp {
                background-color: #2D2D2D;
            }
            
            /* Sidebar background */
            [data-testid="stSidebar"] {
                background-color: #1E1E1E !important;
            }
            
            /* Global text styling with colors and shadows */
            h1, h2, h3, h4, h5, h6, .stMarkdown, p, span, label, .stMetric, .stDataFrame, .stTable, [data-testid="stCaption"] {
                color: #E8DDB4 !important;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            
            /* Header specific colors */
            h1 {
                color: #DAA464 !important;
                text-shadow: 2px 2px 4px rgba(0,0,0,1);
                text-align: center;
                padding-bottom: 20px;
            }
            
            h2, h3 {
                color: #DEC384 !important;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.9);
            }

            /* Metric labels and values */
            [data-testid="stMetricLabel"] p {
                color: #DEC384 !important;
                font-size: 1.1rem !important;
            }
            [data-testid="stMetricValue"] div {
                color: #DAA464 !important;
                font-weight: bold !important;
            }
            
            /* Sidebar header */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
                color: #DAA464 !important;
            }

            /* Buttons and interactive elements */
            .stButton>button {
                background-color: #DAA464 !important;
                color: #2D2D2D !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                border: none !important;
                text-shadow: none !important;
            }
            .stButton>button:hover {
                background-color: #DEC384 !important;
                border: 1px solid #E8DDB4 !important;
            }
            
            /* Fix for dataframes and tables: Light background with dark charcoal text */
            .stDataFrame [data-testid="stTable"], .stTable, [data-testid="stTable"] td, [data-testid="stTable"] th {
                background-color: #E8DDB4 !important;
                color: #2D2D2D !important;
                text-shadow: none !important;
            }
            .stDataFrame, div[data-testid="stTable"] {
                background-color: #E8DDB4 !important;
            }
            
            /* Specific container styling for the planner */
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.planner-marker) {
                background-color: #262626 !important;
                border: 2px solid #DAA464 !important;
                border-radius: 15px !important;
                padding: 10px !important;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.5) !important;
            }
            .planner-marker {
                display: none;
            }

            /* Custom card/div container for plain text info */
            .custom-card {
                background-color: #333333 !important;
                border: 1px solid #DAA464 !important;
                padding: 20px !important;
                border-radius: 12px !important;
                margin-bottom: 20px !important;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.4) !important;
            }

            /* Metrics as cards */
            [data-testid="stMetric"] {
                background-color: #333333 !important;
                border: 1px solid #DAA464 !important;
                padding: 15px !important;
                border-radius: 10px !important;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3) !important;
                text-align: center !important;
            }
            
            /* Dataframes as cards */
            .stDataFrame {
                border: 1px solid #DAA464 !important;
                border-radius: 10px !important;
                padding: 5px !important;
                background-color: #333333 !important;
            }

            /* Chat message and planner container styling */
            [data-testid="stChatMessage"] {
                background-color: #3D3D3D !important;
                border-radius: 15px !important;
                border: 1px solid #4D4D4D !important;
                padding: 15px !important;
                margin-bottom: 15px !important;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
            }
            [data-testid="stChatMessageContent"] p {
                color: #E8DDB4 !important;
            }
            .chat-container {
                background-color: #262626 !important;
                border: 1px solid #DAA464 !important;
                padding: 20px !important;
                border-radius: 15px !important;
                margin-top: 10px !important;
            }

            /* Divider color */
            hr {
                border-top: 1px solid #DAA464 !important;
                opacity: 0.3;
            }
        </style>
    """, unsafe_allow_html=True)

    # Banner at the top
    if Path("code/assets/banner.png").exists():
        st.image("code/assets/banner.png", use_container_width=True)
    
    st.title("CDA Bus Route Process Mining")
    st.divider()

    if not ROUTES_CSV.exists():
        st.error(
            "data/routes.csv was not found. Run "
            "`python code/run_pipeline.py extract --limit 8` first."
        )
        st.stop()

    routes = load_routes(ROUTES_CSV)
    route_options = ["All Routes"] + sorted(routes["route_id"].unique())

    with st.sidebar:
        if Path("code/assets/logo.png").exists():
            st.image("code/assets/logo.png", use_container_width=True)
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
        st.checkbox(
            "Polish route answers with LLM (slower; needs API key)",
            value=False,
            key="polish_llm",
            help="When off, trip plans use instant grounded text. When on, replies are rewritten via Groq/OpenAI.",
        )
        st.divider()
        st.markdown("### 🔍 Display Settings")
        graph_width = st.slider(
            "Graph Zoom (Width)",
            min_value=500,
            max_value=6000,
            value=1200,
            step=100,
            help="Increase to make the graph larger and text more visible."
        )

    filtered = filter_routes(routes, selected_route)
    threshold_seconds = threshold_minutes * 60
    transitions = transition_table(filtered)

    summary = throughput_summary(filtered)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg throughput", format_seconds(summary.average_seconds))
    col_b.metric("Min throughput", format_seconds(summary.minimum_seconds))
    col_c.metric("Max throughput", format_seconds(summary.maximum_seconds))
    st.divider()

    st.subheader("Process Map")
    dot_str = build_process_dot(filtered, threshold_seconds, transitions)
    # Render SVG to allow manual width scaling (zoom)
    svg_data = graphviz.Source(dot_str).pipe(format="svg").decode("utf-8")
    st.image(svg_data, width=graph_width)
    st.divider()

    st.subheader("Evidence")
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
    st.divider()

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
    st.divider()

    st.subheader("🤖 Grounded Trip Planner")
    with st.container(border=True):
        st.markdown('<div class="planner-marker"></div>', unsafe_allow_html=True)
        render_chat_fragment()

    with st.expander("Personal Route Map Bonus", expanded=True):
        render_personal_maps(routes, graph_width)


@st.cache_data
def load_routes(path: Path) -> pd.DataFrame:
    """Load routes.csv."""

    return pd.read_csv(path)


@st.cache_resource
def build_agent(_routes: pd.DataFrame, use_llm: bool) -> GroundedRouteAgent:
    """Cache the grounded route agent (separate cache entries per LLM mode)."""

    return GroundedRouteAgent(_routes, use_llm=use_llm)


@st.fragment
def render_chat_fragment() -> None:
    """Chat-only reruns with static greeting and inline input."""

    routes = load_routes(ROUTES_CSV)
    use_llm = bool(st.session_state.get("polish_llm", False))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 1. Static Greeting (Displayed once at the top)
    st.markdown("""
        <div style="color: #E8DDB4; background-color: #3D3D3D; padding: 18px; border-radius: 12px; border-left: 6px solid #DAA464; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
            👋 Hi! I’m your <b>CDA trip assistant</b>. I answer from your schedule data (routes.csv): 
            routes between stops, which line serves a stop, and departures. 
            Say hello, ask <b>what you can do</b>, or try <b>“from [stop A] to [stop B]”</b>.
        </div>
    """, unsafe_allow_html=True)

    # 2. Messages area (scrollable if needed)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Inline Input (Actually inside the section div)
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            prompt = st.text_input(
                "Chat Input", 
                placeholder="Ask about routes, stops...", 
                label_visibility="collapsed",
                key="chat_prompt"
            )
        with col_btn:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Rerun fragment to show user message immediately
        st.rerun()

    # Process the last message if it's from the user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_prompt = st.session_state.messages[-1]["content"]
        agent = build_agent(routes, use_llm)
        with st.spinner("Thinking…"):
            answer = agent.answer(last_prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def build_personal_commute_trace(item: dict[str, object]) -> pd.DataFrame:
    """Build a member-specific commute trace table from a route option."""

    option = item.get("option")
    if option is None:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    cumulative_seconds = 0.0
    for index, stop in enumerate(option.stops):
        previous_stop = option.stops[index - 1] if index > 0 else ""
        route_id = option.routes[index - 1] if index > 0 and index - 1 < len(option.routes) else ""
        duration = option.steps[index - 1].duration_seconds if index > 0 else 0.0
        cumulative_seconds += duration
        rows.append(
            {
                "member_name": item.get("member_name", ""),
                "member_id": item.get("member_id", ""),
                "home_area": item.get("home_area", ""),
                "origin_stop": item.get("origin_stop", ""),
                "destination_stop": item.get("destination_stop", ""),
                "status": item.get("status", ""),
                "event_order": index + 1,
                "stop_name": stop,
                "previous_stop": previous_stop,
                "route_id": route_id,
                "transition_seconds_from_previous": duration,
                "cumulative_seconds": cumulative_seconds,
                "cumulative_time": format_seconds(cumulative_seconds),
            }
        )

    return pd.DataFrame(rows)


def render_personal_maps(routes: pd.DataFrame, graph_width: int) -> None:
    """Render bonus personal route-map evidence."""

    personal_path = ROUTES_CSV.parent / "personal_routes.csv"
    personal_routes = load_personal_routes(personal_path)
    
    agent = build_agent(routes, False)
    options = build_personal_options(agent, personal_routes)
    
    # Team members + Custom option
    member_names = [item["member_name"] for item in options] + ["➕ Create Your Own Route"]
    
    selected_member = st.selectbox(
        "Select team member or build custom route:",
        member_names,
        index=0,
    )

    if selected_member == "➕ Create Your Own Route":
        all_stops = sorted(set(routes["stop_name"]))
        st.markdown("### 🛠️ Custom Route Builder")
        col1, col2 = st.columns(2)
        origin = col1.selectbox("Origin Stop", all_stops, key="custom_origin")
        destination = col2.selectbox("Destination Stop", all_stops, key="custom_dest")
        
        option = agent.shortest_path(origin, destination)
        selected = {
            "member_name": "Custom Explorer",
            "home_area": "Dynamic Area",
            "origin_stop": origin,
            "destination_stop": destination,
            "status": "Path discovered" if option else "No path found in current routes",
            "option": option
        }
    else:
        selected = next(
            (item for item in options if item["member_name"] == selected_member),
            None,
        )

    if selected is None:
        st.warning("Selected member data is not available.")
        return

    option = selected.get("option")
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: #DAA464; margin-top: 0; margin-bottom: 10px;">{selected.get('member_name')}</h4>
            <div style="line-height: 1.6; color: #E8DDB4;">
                <b>🏠 Home area:</b> {selected.get('home_area', '')}<br>
                <b>📍 Origin:</b> {selected.get('origin_stop', '')}<br>
                <b>🏁 Destination:</b> {selected.get('destination_stop', '')}<br>
                <b>ℹ️ Status:</b> {selected.get('status', '')}
            </div>
        </div>
    """, unsafe_allow_html=True)
    if option is None:
        return

    st.markdown(f"**Route path:** {' → '.join(option.stops)}")
    st.dataframe(
        build_personal_commute_trace(selected),
        use_container_width=True,
        hide_index=True,
    )
    svg_data = graphviz.Source(build_personal_dot(option)).pipe(format="svg").decode("utf-8")
    st.image(svg_data, width=graph_width)


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
