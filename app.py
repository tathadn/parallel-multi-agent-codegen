"""
Parallel Multi-Agent Code Generator — Streamlit UI

Features:
  - Real-time agent status dashboard with parallel worker tracking
  - Interactive DAG visualization showing task dependencies
  - Live log stream with timestamps
  - Code artifact viewer with syntax highlighting
  - Performance metrics and timing breakdown
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from graph.pipeline import build_graph
from models.state import AgentState, TaskStatus

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Parallel Multi-Agent Codegen",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished dashboard look
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at 20% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        position: relative;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
        position: relative;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
        position: relative;
    }

    /* Status cards */
    .status-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .status-card.active {
        border-color: #89b4fa;
        box-shadow: 0 0 15px rgba(137, 180, 250, 0.15);
    }
    .status-card.done {
        border-color: #a6e3a1;
    }
    .status-card.error {
        border-color: #f38ba8;
    }
    .status-card .agent-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #cdd6f4;
    }
    .status-card .agent-status {
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* DAG node */
    .dag-node {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        font-size: 0.85rem;
        color: #cdd6f4;
        font-family: 'JetBrains Mono', monospace;
    }
    .dag-node.idle { border-color: #585b70; }
    .dag-node.running { border-color: #89b4fa; background: rgba(137,180,250,0.1); }
    .dag-node.done { border-color: #a6e3a1; background: rgba(166,227,161,0.1); }
    .dag-node.error { border-color: #f38ba8; background: rgba(243,139,168,0.1); }

    /* Log area */
    .log-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #a6adc8;
        padding: 0.15rem 0;
        border-bottom: 1px solid rgba(49, 50, 68, 0.5);
    }

    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #cdd6f4;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #6c7086;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    max_iterations = st.slider(
        "Max revision iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum number of code → review → test revision loops",
    )

    tester_model = st.selectbox(
        "Tester model",
        options=[
            "claude-haiku-4-5-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ],
        index=1,
        format_func=lambda x: {
            "claude-haiku-4-5-20241022": "⚡ Haiku (fast)",
            "claude-sonnet-4-20250514": "🎯 Sonnet (balanced)",
            "claude-opus-4-20250514": "🧠 Opus (thorough)",
        }.get(x, x),
    )

    st.markdown("---")
    st.markdown("### 💰 Cost & Tokens")
    cost_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### 📊 Pipeline Architecture")
    st.markdown("""
    ```
    Orchestrator
        │
    Planner
        │
    DAG Decompose
        │
    ┌───┴───┐
    Worker₁ Worker₂  ← parallel
    └───┬───┘
    Integrator
        │
    Reviewer
        │
    Tester
        │
    ✅ or 🔄 revise
    ```
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#6c7086; font-size:0.75rem;'>"
        "Built with LangGraph + Claude<br>"
        "github.com/tathadn/multi-agent-codegen"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>⚡ Parallel Multi-Agent Codegen <span class="badge">v2.0</span></h1>
        <p>Describe what you want to build — an orchestrator decomposes it into parallel tasks,
        dispatches concurrent AI coding agents, then reviews and tests the merged output.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

user_request = st.text_area(
    "What do you want to build?",
    placeholder=(
        "Example: A Python FastAPI server with /health, /users CRUD endpoints, "
        "SQLAlchemy models, Pydantic schemas, and pytest tests"
    ),
    height=100,
    label_visibility="collapsed",
)

col_run, col_example = st.columns([1, 3])
with col_run:
    run_clicked = st.button("🚀 Generate", type="primary", use_container_width=True)
with col_example:
    example = st.selectbox(
        "Or try an example:",
        options=[
            "",
            "A Python CLI tool that converts CSV files to JSON with validation and error handling",
            "A FastAPI REST API with user authentication, JWT tokens, and SQLite database",
            "A Python library for graph algorithms: BFS, DFS, Dijkstra, with a CLI interface",
            "A task queue system with worker pool, retry logic, and dead letter queue",
        ],
        label_visibility="collapsed",
    )

if example and not user_request:
    user_request = example


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def render_dag(state: AgentState) -> None:
    """Render the task DAG as visual nodes."""
    if not state.task_dag:
        return

    st.markdown("#### 🔀 Task Dependency Graph")

    nodes = state.task_dag.topological_order()
    for node in nodes:
        status_class = node.status.value.lower()
        status_emoji = {
            "idle": "⏳",
            "running": "🔄",
            "done": "✅",
            "error": "❌",
        }.get(status_class, "⏳")

        deps = ""
        if node.depends_on:
            dep_names = []
            for did in node.depends_on:
                dn = state.task_dag.get_node(did)
                if dn:
                    dep_names.append(dn.name)
            deps = f" ← depends on: {', '.join(dep_names)}"

        elapsed = f" ({node.elapsed}s)" if node.elapsed else ""

        st.markdown(
            f'<div class="dag-node {status_class}">'
            f"{status_emoji} <strong>{node.name}</strong>"
            f'<span style="color:#6c7086">{deps}{elapsed}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_cost(state: AgentState, target) -> None:
    """Render cost dashboard: running total, cache hit rate, per-agent breakdown."""
    if not state.usage_log:
        target.markdown(
            "<div style='color:#6c7086; font-size:0.8rem;'>No LLM calls yet.</div>",
            unsafe_allow_html=True,
        )
        return

    total_cost = state.total_cost_usd()
    totals = state.total_tokens()
    hit_rate = state.cache_hit_rate()

    with target.container():
        c1, c2 = st.columns(2)
        c1.metric("Total cost", f"${total_cost:.4f}")
        c2.metric("Cache hit", f"{hit_rate * 100:.0f}%")

        st.caption(
            f"in: {totals['input']:,}  ·  cached: {totals['cached']:,}  ·  "
            f"out: {totals['output']:,}"
        )

        # Per-agent breakdown
        by_agent = state.cost_by_agent()
        if by_agent:
            rows = [
                {"agent": agent, "cost_usd": f"${cost:.4f}"}
                for agent, cost in sorted(by_agent.items(), key=lambda kv: kv[1], reverse=True)
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)

        with st.expander("Raw usage log", expanded=False):
            st.dataframe(
                [
                    {
                        "agent": u.agent,
                        "model": u.model.replace("claude-", "")
                        .replace("-20250514", "")
                        .replace("-20241022", ""),
                        "in": u.input_tokens,
                        "cached": u.cached_input_tokens,
                        "out": u.output_tokens,
                        "cost": f"${u.cost_usd:.4f}",
                        "latency_s": u.latency_s,
                    }
                    for u in state.usage_log
                ],
                hide_index=True,
                use_container_width=True,
            )


def render_metrics(state: AgentState) -> None:
    """Render performance metrics."""
    cols = st.columns(4)

    with cols[0]:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(state.artifacts)}</div>'
            f'<div class="metric-label">Files Generated</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with cols[1]:
        parallel_nodes = 0
        if state.task_dag:
            parallel_nodes = len(state.task_dag.nodes)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{parallel_nodes}</div>'
            f'<div class="metric-label">Parallel Tasks</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with cols[2]:
        score = state.review.score if state.review else "—"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{score}/10</div>'
            f'<div class="metric-label">Review Score</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with cols[3]:
        if state.test_result:
            tests_str = f"{state.test_result.passed_tests}/{state.test_result.total_tests}"
        else:
            tests_str = "—"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{tests_str}</div>'
            f'<div class="metric-label">Tests Passed</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


if run_clicked and user_request:
    # Build initial state
    initial_state = AgentState(
        user_request=user_request.strip(),
        max_iterations=max_iterations,
    )

    # Build graph
    graph = build_graph()

    # Create layout
    col_main, col_sidebar = st.columns([3, 1])

    with col_sidebar:
        status_placeholder = st.empty()
        dag_placeholder = st.empty()

    with col_main:
        metrics_placeholder = st.empty()
        log_placeholder = st.empty()
        results_placeholder = st.empty()

    # Stream execution
    agent_phases = {
        "orchestrator": ("🎯 Orchestrator", "Parsing request"),
        "planner": ("📋 Planner", "Building plan"),
        "orchestrator_dag": ("🔀 DAG Builder", "Decomposing into tasks"),
        "parallel_coders": ("⚡ Parallel Coders", "Coding concurrently"),
        "integrator": ("🔗 Integrator", "Merging outputs"),
        "reviewer": ("🔍 Reviewer", "Evaluating code"),
        "tester": ("🧪 Tester", "Running tests"),
    }

    current_phase = ""
    final_state = initial_state

    try:
        for event in graph.stream(initial_state, stream_mode="values"):
            if isinstance(event, AgentState):
                final_state = event
            elif isinstance(event, dict):
                final_state = AgentState(**event)

            # Update sidebar status
            with status_placeholder.container():
                st.markdown("#### 🔄 Agent Status")
                for key, (name, desc) in agent_phases.items():
                    is_active = final_state.status.value.lower() in key.lower()
                    card_class = "active" if is_active else ""

                    if final_state.status == TaskStatus.COMPLETED:
                        card_class = "done"
                    elif final_state.status == TaskStatus.FAILED:
                        card_class = "error"

                    st.markdown(
                        f'<div class="status-card {card_class}">'
                        f'<div class="agent-name">{name}</div>'
                        f'<div class="agent-status">{desc}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Update DAG visualization
            with dag_placeholder.container():
                render_dag(final_state)

            # Update metrics
            with metrics_placeholder.container():
                render_metrics(final_state)

            # Update cost dashboard
            render_cost(final_state, cost_placeholder)

            # Update logs
            with log_placeholder.container():
                st.markdown("#### 📜 Pipeline Log")
                for line in final_state.logs[-20:]:
                    st.markdown(
                        f'<div class="log-line">{line}</div>',
                        unsafe_allow_html=True,
                    )

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        final_state.log(f"❌ Fatal error: {e}")

    # Final results
    with results_placeholder.container():
        st.markdown("---")

        # Status banner
        if final_state.status == TaskStatus.COMPLETED:
            review_score = final_state.review.score if final_state.review else "N/A"
            passed = final_state.test_result.passed_tests if final_state.test_result else 0
            total = final_state.test_result.total_tests if final_state.test_result else 0
            st.success(
                f"✅ **Pipeline COMPLETED** — "
                f"{len(final_state.artifacts)} files generated, "
                f"review score {review_score}/10, "
                f"{passed}/{total} tests passed"
            )
        elif final_state.status == TaskStatus.FAILED:
            st.error("❌ **Pipeline FAILED** — see logs for details")
        else:
            st.warning(f"⚠️ Pipeline ended with status: {final_state.status.value}")

        # Timing breakdown
        if final_state.timings:
            st.markdown("#### ⏱️ Timing Breakdown")
            timing_cols = st.columns(len(final_state.timings))
            for i, (phase, duration) in enumerate(final_state.timings.items()):
                with timing_cols[i]:
                    st.metric(phase.replace("_", " ").title(), f"{duration}s")

        # Code artifacts
        if final_state.artifacts:
            st.markdown("#### 📁 Generated Files")
            for artifact in final_state.artifacts:
                with st.expander(f"📄 {artifact.filename}", expanded=False):
                    st.code(artifact.content, language=artifact.language)

        # Review details
        if final_state.review:
            with st.expander("🔍 Review Details", expanded=False):
                st.json(
                    {
                        "score": final_state.review.score,
                        "approved": final_state.review.approved,
                        "issues": final_state.review.issues,
                        "suggestions": final_state.review.suggestions,
                    }
                )

        # Test results
        if final_state.test_result:
            with st.expander("🧪 Test Results", expanded=False):
                st.json(
                    {
                        "passed": final_state.test_result.passed,
                        "total": final_state.test_result.total_tests,
                        "passed_count": final_state.test_result.passed_tests,
                        "failed_count": final_state.test_result.failed_tests,
                    }
                )
                if final_state.test_result.error_output:
                    st.code(final_state.test_result.error_output, language="text")

elif run_clicked and not user_request:
    st.warning("Please describe what you want to build.")
