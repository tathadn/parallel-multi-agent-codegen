# Parallel Multi-Agent Code Generator

A LangGraph-based system that decomposes coding tasks into a DAG and generates code concurrently using parallel AI agents. Built with Python, LangChain/LangGraph, Claude API, Pydantic v2, and Streamlit.

## Architecture

The pipeline flows: Orchestrator → Planner → Orchestrator (DAG decompose) → Parallel Coder Workers → Integrator → Reviewer → Tester → [revise or complete].

Key modules:
- `agents/orchestrator.py` — central coordinator, builds TaskDAG from Planner output. This is the core innovation.
- `agents/coder.py` — parallel workers using `asyncio` + `ThreadPoolExecutor` to run concurrent LLM calls.
- `agents/integrator.py` — merges parallel outputs, resolves import mismatches.
- `agents/planner.py`, `agents/reviewer.py`, `agents/tester.py` — standard pipeline agents.
- `agents/llm_utils.py` — shared `call_llm()` and `parse_json_response()` helpers wrapping `ChatAnthropic`.
- `models/state.py` — all Pydantic models: `AgentState`, `TaskDAG`, `TaskNode`, `Plan`, `CodeArtifact`, etc.
- `prompts/__init__.py` — all system prompts and templates. Agents return structured JSON, not prose.
- `graph/pipeline.py` — LangGraph `StateGraph` definition with conditional revision edge.
- `app.py` — Streamlit UI with real-time DAG visualization and agent status dashboard.

## Commands

```bash
pip install -e ".[dev]"          # Install with dev dependencies
streamlit run app.py             # Run the Streamlit app (localhost:8501)
python -m pytest -v              # Run tests
ruff check .                     # Lint
ruff format .                    # Format
mypy .                           # Type check
```

## Code conventions

- Python 3.10+. Use `from __future__ import annotations` in every module.
- Type hints on all function signatures. Use `Optional[X]` not `X | None` for Pydantic fields.
- Pydantic v2 `BaseModel` for all data structures. No dataclasses, no TypedDicts.
- All agent functions take `AgentState` and return `AgentState`. Never mutate state outside the agent function.
- LLM prompts live in `prompts/__init__.py`, not inline in agent code. Reference them by name.
- Agent LLM calls go through `agents/llm_utils.py` — use `call_llm()` and `parse_json_response()`. Never instantiate `ChatAnthropic` directly in agent modules.
- All agents log via `state.log("emoji message")` and record timing via `state.timings[key]`.
- JSON is the contract format between agents and LLMs. Agents must handle malformed JSON gracefully using `parse_json_response()` which strips markdown fences and extracts JSON.

## Critical patterns

- **TaskDAG is the core data structure.** `TaskNode.depends_on` contains IDs (strings), not references. Use `dag.get_node(id)` to resolve. `dag.ready_nodes()` returns nodes with all deps satisfied and status IDLE.
- **Parallel execution** uses `asyncio.new_event_loop()` in `coder_worker()` because LangGraph nodes are synchronous. The loop runs batches of `_run_node_async()` which dispatches to `ThreadPoolExecutor`. Do not use `asyncio.run()` — it conflicts with Streamlit's event loop.
- **Revision loop**: `should_continue()` in `graph/pipeline.py` routes to "complete", "fail", or "revise". On revise, `coder_worker()` resets all DAG nodes to IDLE and clears artifacts before re-running.
- **Model names**: Use `claude-sonnet-4-20250514` as default. The tester model is configurable from the Streamlit sidebar. Never hardcode `claude-3-*` model strings — this project uses Claude 4 family.

## Environment

- `ANTHROPIC_API_KEY` must be set in `.env` or environment. Loaded via `python-dotenv`.
- Optional: `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` for LangSmith tracing.

## When modifying agents

1. Keep the `AgentState` in → `AgentState` out contract. If you need new state fields, add them to `models/state.py` with defaults so existing graph nodes don't break.
2. If adding a new agent, register it in `agents/__init__.py`, add its node in `graph/pipeline.py`, and wire edges. Update the `agent_phases` dict in `app.py` for the dashboard.
3. If changing prompt structure, update both the system prompt and the corresponding parsing logic in the agent. Prompts instruct LLMs to return "ONLY JSON" — keep that contract.
4. Test any DAG changes by verifying `ready_nodes()` returns correct batches. A cycle in `depends_on` will hang the pipeline.

## Do not

- Do not add `__init__.py` files to the project root — it's not a package, it's an application.
- Do not use `print()` for logging — use `state.log()` so output appears in the Streamlit dashboard.
- Do not import between agent modules (e.g., coder importing from reviewer). Agents communicate only through `AgentState`.
- Do not put CSS or HTML in Python files outside of `app.py`. The Streamlit styling is consolidated there.
