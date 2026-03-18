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
- **Model tiering**: Not every agent needs Sonnet. Use the correct model per agent role (see Cost optimization below). The tester model is configurable from the Streamlit sidebar. Never hardcode `claude-3-*` model strings — this project uses Claude 4 family.

## Environment

- `ANTHROPIC_API_KEY` must be set in `.env` or environment. Loaded via `python-dotenv`.
- Optional: `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` for LangSmith tracing.

## When modifying agents

1. Keep the `AgentState` in → `AgentState` out contract. If you need new state fields, add them to `models/state.py` with defaults so existing graph nodes don't break.
2. If adding a new agent, register it in `agents/__init__.py`, add its node in `graph/pipeline.py`, and wire edges. Update the `agent_phases` dict in `app.py` for the dashboard.
3. If changing prompt structure, update both the system prompt and the corresponding parsing logic in the agent. Prompts instruct LLMs to return "ONLY JSON" — keep that contract.
4. Test any DAG changes by verifying `ready_nodes()` returns correct batches. A cycle in `depends_on` will hang the pipeline.

## Cost optimization

This project is a portfolio piece. Every API dollar counts. Follow these rules in priority order.

### 1. Prompt caching (highest impact — saves 60–90%)

All LLM calls in `agents/llm_utils.py` must use the Anthropic native SDK with `cache_control` enabled. System prompts are identical across runs — caching them cuts input costs to 10% after the first call.

```python
# In call_llm() — use anthropic SDK directly, not ChatAnthropic
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model=model_name,
    max_tokens=8192,
    cache_control={"type": "ephemeral"},  # ALWAYS include this
    system=system,
    messages=[{"role": "user", "content": prompt}],
)
```

Cache rules:
- Always use `cache_control={"type": "ephemeral"}` on every `messages.create()` call.
- Cache TTL is 5 minutes and refreshes on each hit — our full pipeline runs within this window.
- System prompts are cached automatically since they're the stable prefix of every request.
- Cache requires the Anthropic native Messages API. Do not use the OpenAI-compatible endpoint — it does not support caching.
- Minimum cacheable size is 1,024 tokens for Sonnet, 2,048 for Opus, and 4,096 for Haiku.

### 2. Model tiering (saves 30–50%)

Match model to task complexity. Each agent function accepts a `model_name` parameter — set the default per agent:

| Agent | Model | String | Why |
|---|---|---|---|
| Orchestrator (intake) | Haiku | `claude-haiku-4-5-20241022` | Simple validation, no reasoning |
| Planner | Sonnet | `claude-sonnet-4-20250514` | Needs strong reasoning for deps |
| Orchestrator (DAG) | Sonnet | `claude-sonnet-4-20250514` | Core parallelism logic |
| Coder Workers | Sonnet | `claude-sonnet-4-20250514` | Code quality matters |
| Integrator | Haiku | `claude-haiku-4-5-20241022` | Mechanical merge, not creative |
| Reviewer | Haiku | `claude-haiku-4-5-20241022` | Scoring + pattern matching |
| Tester | Haiku | `claude-haiku-4-5-20241022` | Test generation is formulaic |

When adding or modifying agents, always set the cheapest model that produces acceptable output. Only upgrade to Sonnet if Haiku output quality is noticeably worse for that agent's task.

### 3. Prompt compression

Keep prompts as short as possible without losing instruction quality:
- Do not include full JSON schema examples in system prompts. Describe fields tersely — the LLM knows JSON.
- Remove redundant guardrails like "no markdown fences, no prose" — `parse_json_response()` already strips fences.
- In Coder prompts, use compact list format for interface contracts, not full sentences.
- Target: system prompts under 500 tokens each, user prompts under 1,000 tokens each.

### 4. Surgical revisions (saves 20–40% on revision loops)

When the Reviewer or Tester flags issues, do NOT re-code the entire DAG. Only reset failing nodes:
- Parse `state.review.issues` and `state.test_result.error_output` to identify which files failed.
- Match failing filenames to their `task_node_id` in `state.artifacts`.
- Only reset those specific `TaskNode` objects to `WorkerStatus.IDLE`.
- Keep passing nodes as `DONE` and preserve their artifacts.

This avoids burning tokens regenerating code that already passed review.

### 5. Development workflow

- Use the simplest possible test prompt during development: "Python function that adds two numbers". Save complex prompts for final testing.
- Set `max_iterations=1` in the sidebar while debugging UI or agent logic.
- Test agents in isolation by calling `agent_function(mock_state)` directly — one LLM call instead of seven.
- Use the Batch API (50% discount on all tokens) for any non-interactive testing or CI runs. The Batch API is async and not suitable for the live Streamlit demo.

### Cost targets

| Scenario | Target per run |
|---|---|
| Optimized (caching + tiering + compression) | $0.06–$0.08 |
| Development (simple prompt, 1 iteration) | $0.02–$0.04 |
| Demo (full pipeline, complex prompt) | $0.10–$0.15 |
| Batch API (non-interactive) | 50% of above |

## Do not

- Do not add `__init__.py` files to the project root — it's not a package, it's an application.
- Do not use `print()` for logging — use `state.log()` so output appears in the Streamlit dashboard.
- Do not import between agent modules (e.g., coder importing from reviewer). Agents communicate only through `AgentState`.
- Do not put CSS or HTML in Python files outside of `app.py`. The Streamlit styling is consolidated there.
- Do not use `ChatAnthropic` from LangChain for LLM calls — use the `anthropic` SDK directly via `call_llm()` so prompt caching works. LangChain's wrapper does not pass `cache_control`.
- Do not use Sonnet for agents that only need Haiku (Orchestrator intake, Integrator, Reviewer, Tester). Check the model tiering table above before setting defaults.
- Do not send full JSON schema examples in system prompts. Keep prompts terse — every token costs money.
- Do not reset all DAG nodes on revision. Only reset nodes whose files were flagged by the Reviewer or Tester.
