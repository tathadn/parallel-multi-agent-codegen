# ⚡ Parallel Multi-Agent Code Generator

[![CI](https://github.com/tathadn/parallel-multi-agent-codegen/actions/workflows/ci.yml/badge.svg)](https://github.com/tathadn/parallel-multi-agent-codegen/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Describe what you want to build — an **orchestrator agent** decomposes it into a DAG of parallel tasks, dispatches concurrent AI coding agents, merges their outputs, reviews and tests the result, and loops back with **surgical revisions** when something fails. All while tracking per-call cost and cache hit rate on a live dashboard.

---

## 📚 Part of a Trilogy

This project is the **middle chapter** of a three-part exploration into multi-agent code generation. Each iteration attacks a different weakness of the previous one:

| # | Project | Core Innovation | What It Asks |
|---|---------|----------------|--------------|
| 1 | **[multi-agent-codegen](https://github.com/tathadn/multi-agent-codegen)** | Sequential agent pipeline (Orchestrator → Planner → Coder → Reviewer → Tester) with LangSmith tracing and Docker sandbox | *Can a team of specialized LLM agents produce working code end-to-end?* |
| 2 | **[parallel-multi-agent-codegen](https://github.com/tathadn/parallel-multi-agent-codegen)** *(this repo)* | **DAG-based task decomposition** + concurrent coder workers, prompt caching, model tiering, surgical revisions, live cost dashboard | *Can we go faster and cheaper by coding independent modules in parallel — the way a real dev team works?* |
| 3 | **[self-evolving-codegen](https://github.com/tathadn/self-evolving-codegen)** | **LLM-as-judge** evaluator + autonomous **prompt evolution** loop — the system rewrites its own prompts based on evaluation feedback | *Can the pipeline improve itself without human intervention?* |

**Progression:**
- **v1** established the baseline: "can agents even do this?"
- **v2 (this)** is about **engineering quality** — parallelism, cost, observability, resilience.
- **v3** is about **meta-learning** — closing the loop so the system improves itself.

If you're new, start here (v2) — it's the most production-shaped of the three.

---

## 🎯 What This Version Does

You type a request. The system:

1. **Validates** the request (Orchestrator intake)
2. **Plans** it into discrete steps with dependencies (Planner)
3. **Decomposes** the plan into a Task DAG of parallelizable units (Orchestrator DAG-builder)
4. **Dispatches** concurrent coder workers via `asyncio` — independent modules are generated in parallel
5. **Merges** their outputs, resolving interface mismatches (Integrator)
6. **Reviews** code quality 0–10 with structured feedback (Reviewer)
7. **Generates tests** and runs them in a Docker sandbox (Tester)
8. **Revises surgically** — only failing nodes reset; passing code is preserved — and loops up to *N* iterations
9. **Tracks every LLM call** — tokens, cache hits, cost, latency — and renders it live in a Streamlit dashboard

Everything routes through a single typed `AgentState` (Pydantic v2) that flows through a LangGraph `StateGraph` with a conditional revision edge.

---

## ✨ Features at a Glance

| Capability | How |
|---|---|
| **Parallel code generation** | DAG-based task decomposition + `asyncio` + `ThreadPoolExecutor` for concurrent worker dispatch |
| **Prompt caching** (60–90% input savings) | Native Anthropic SDK with `cache_control={"type": "ephemeral"}` on system prompts |
| **Model tiering** (30–50% savings) | Sonnet for reasoning-heavy agents (Planner, Coder, DAG builder), Haiku for mechanical ones (Integrator, Reviewer, Tester) |
| **Surgical revisions** (20–40% savings on retry loops) | Only nodes whose files were flagged by Reviewer/Tester reset to IDLE; passing artifacts persist |
| **Live cost dashboard** | Real-time USD total, cache hit rate, per-agent breakdown, raw usage log — all in the Streamlit sidebar |
| **Automatic retries** | `tenacity` exponential backoff (2–30s, 4 attempts) on rate limits, timeouts, and 5xx; `BadRequestError` never retried |
| **Structured error taxonomy** | Anthropic exceptions wrapped in domain types (`LLMRateLimitedError`, `LLMTimeoutError`, `LLMBadRequestError`, `ParseFailureError`) |
| **Real-time DAG visualization** | Streamlit dashboard shows worker status live (idle / running / done / error) |
| **Full LangSmith tracing** | `@traceable` decorator on every `call_llm()` — including parallel workers |
| **Docker sandbox execution** | Generated tests run in an isolated container (see `sandbox/Dockerfile`) |
| **102 offline tests** | Every code path mocked; zero real API calls in CI |
| **GitHub Actions CI** | Ruff + format + pytest on Python 3.10 / 3.11 / 3.12 |

---

## 🏛️ Architecture

```
User Request
     │
     ▼
┌──────────────────┐
│   Orchestrator    │  Validates request, sets pipeline status
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     Planner       │  Produces structured plan: steps, files, dependencies
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Orchestrator     │  Decomposes plan → Task DAG
│  (DAG Builder)    │  Identifies parallelizable modules
└────────┬─────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│   ← parallel coding via asyncio
└───┬────┘ └───┬────┘ └───┬────┘
    └─────┬────┴──────────┘
          │
          ▼
┌──────────────────┐
│    Integrator     │  Merges parallel outputs, resolves conflicts
└────────┬─────────┘
         │
         ▼
┌──────────────────┐ ◄─────────────────────────────┐
│     Reviewer      │  Scores code 0–10             │  surgical revision loop
└────────┬─────────┘                                │  (only failing nodes reset)
         │                                          │
         ▼                                          │
┌──────────────────┐  tests fail or review          │
│      Tester       │  not approved? ───────────────┘
└────────┬─────────┘
         │  all green
         ▼
    Final Output
    (code files + plan + review + test report + cost/usage log)
```

### How the Orchestrator builds the DAG

The Orchestrator is the architectural centerpiece. After the Planner produces a step-by-step implementation plan, the Orchestrator:

1. **Analyzes file dependencies** — which modules import from each other?
2. **Groups tightly coupled files** — e.g., a model and its schema become one DAG node
3. **Builds dependency edges** — if module A imports from module B, there's an edge `B → A`
4. **Maximizes parallelism** — nodes with no mutual dependencies are dispatched to concurrent workers

For a FastAPI app with user auth, the decomposition might look like:

```
         ┌──────────┐
         │  config   │  (no deps — starts immediately)
         └─────┬────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ models │ │ schemas│ │  auth  │   ← all 3 run in parallel
└───┬────┘ └───┬────┘ └───┬────┘
    └──────┬───┘          │
           ▼              │
      ┌────────┐          │
      │ routes │ ◄────────┘          ← waits for models, schemas, auth
      └───┬────┘
          ▼
      ┌────────┐
      │  main  │                      ← waits for routes
      └────────┘
```

Three modules code concurrently instead of five running sequentially — often a 40%+ end-to-end speedup.

---

## 🤖 Agent Roles

| Agent | Default Model | Role |
|---|---|---|
| **Orchestrator (intake)** | *(no LLM)* | Validates request, sets initial status |
| **Planner** | Sonnet | Generates structured plan: objective, steps, files, deps, complexity |
| **Orchestrator (DAG builder)** | Sonnet | Decomposes plan into parallel Task DAG |
| **Coder Workers** | Sonnet | Generate code for assigned DAG nodes (run concurrently) |
| **Integrator** | Haiku | Merges parallel outputs, resolves import mismatches |
| **Reviewer** | Haiku | Scores code quality (0–10), lists issues + suggestions |
| **Tester** | Haiku *(configurable)* | Generates pytest tests, runs them in sandbox, parses results |

Model tiering is intentional — Sonnet where reasoning quality matters, Haiku for mechanical tasks. The Tester model is configurable from the Streamlit sidebar for complex projects.

### Shared State

All agents read from and write to a Pydantic `AgentState`:

```python
class AgentState(BaseModel):
    user_request: str
    plan:         Optional[Plan]
    task_dag:     Optional[TaskDAG]       # parallel task graph
    artifacts:    list[CodeArtifact]
    review:       Optional[ReviewFeedback]
    test_result:  Optional[TestResult]

    # Pipeline control
    status:       TaskStatus
    iteration:    int
    max_iterations: int
    retry_budget: int                     # per-run retry limit

    # Observability
    logs:         list[str]               # timestamped pipeline log
    timings:      dict[str, float]        # per-phase timing
    usage_log:    list[LLMUsage]          # per-call cost/tokens/latency

    # Helpers
    def total_cost_usd(self) -> float: ...
    def cache_hit_rate(self) -> float: ...
    def cost_by_agent(self) -> dict[str, float]: ...
```

### TaskDAG — the core data structure

```python
class TaskNode(BaseModel):
    id:          str                   # unique short ID
    name:        str                   # module name
    description: str                   # what this module does
    files:       list[str]             # files to generate
    depends_on:  list[str]             # IDs of prerequisite nodes
    status:      WorkerStatus          # IDLE → RUNNING → DONE | ERROR
    worker_id:   Optional[str]
    started_at:  Optional[float]
    finished_at: Optional[float]

class TaskDAG(BaseModel):
    nodes: list[TaskNode]

    def ready_nodes(self) -> list[TaskNode]:
        """Nodes with all deps satisfied — ready for parallel dispatch."""
    def all_done(self) -> bool: ...
    def topological_order(self) -> list[TaskNode]: ...
```

---

## 💰 Cost Optimization

Three techniques stacked, all implemented in this repo:

### 1. Prompt caching (60–90% savings)

Every LLM call uses the native Anthropic SDK with `cache_control={"type": "ephemeral"}` on the system prompt. System prompts are identical across runs → cached after the first call → input cost drops to ~10%.

LangChain's `ChatAnthropic` wrapper does **not** pass `cache_control`, so this project bypasses it and calls the `anthropic` SDK directly through [agents/llm_utils.py](agents/llm_utils.py). LangSmith tracing is preserved via `@traceable`.

### 2. Model tiering (30–50% savings)

See the Agent Roles table above. Haiku for mechanical tasks, Sonnet for reasoning.

### 3. Surgical revisions (20–40% savings on revision loops)

When the Reviewer or Tester flags issues, [agents/coder.py](agents/coder.py) parses the issue strings and `test_result.error_output` to identify which files failed, then only resets the corresponding DAG nodes to `IDLE`. Passing code is preserved.

```
Reviewer flags "Missing error handling in routes.py"
  ↓
Only the "routes" DAG node resets → re-coded next iteration
config.py, models.py, auth.py → remain DONE, artifacts preserved
```

Falls back to a full reset when no filenames can be matched in the issue text.

### Cost dashboard

Everything above is verifiable in real time. Every `call_llm()` invocation records an `LLMUsage` entry:

```python
class LLMUsage(BaseModel):
    agent: str                  # "planner", "coder:abc123", "reviewer", ...
    model: str
    input_tokens: int
    cached_input_tokens: int    # from response.usage.cache_read_input_tokens
    cache_creation_tokens: int
    output_tokens: int
    cost_usd: float             # computed from agents/pricing.py
    latency_s: float
    timestamp: float
```

The Streamlit sidebar renders:
- **Running total** in USD
- **Cache hit rate** (cached / (cached + fresh) input tokens)
- **Per-agent breakdown** table sorted by cost
- **Raw usage log** expander with every call

### Cost targets

| Scenario | Target per run |
|---|---|
| Optimized (caching warm + tiering + surgical revisions) | $0.06–$0.08 |
| Development (simple prompt, 1 iteration) | $0.02–$0.04 |
| Demo (full pipeline, complex prompt, multi-iteration) | $0.10–$0.15 |

---

## 🛡️ Resilience

One 529 shouldn't kill a $0.10 pipeline.

| Error | Behavior |
|---|---|
| `RateLimitError` (429) | Retry 4× with exponential backoff (2s → 30s), then raise `LLMRateLimitedError` |
| `InternalServerError` (5xx) | Retry 4× with exponential backoff |
| `APIConnectionError` / `APITimeoutError` | Retry 4× with exponential backoff, then raise `LLMTimeoutError` |
| `BadRequestError` (400) | **No retry** — programmer bug, fail fast as `LLMBadRequestError` |
| JSON parse failure | `ParseFailureError` (also subclass of `ValueError` for backward compat) |

Implemented with `tenacity` in [agents/llm_utils.py](agents/llm_utils.py); domain error types live in [models/errors.py](models/errors.py).

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- Docker *(optional — only needed for sandbox test execution)*

### Install and run

```bash
# 1. Clone
git clone https://github.com/tathadn/parallel-multi-agent-codegen.git
cd parallel-multi-agent-codegen

# 2. Install (editable, with dev deps)
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Type a request, hit **Run**, and watch the DAG come alive on the right while the cost dashboard ticks up on the left.

---

## 📸 Example

**Input:**

```
A Python library for graph algorithms (BFS, DFS, Dijkstra) with a CLI interface
```

**Orchestrator decomposes into:**

```
┌──────────┐     ┌──────────┐
│  graph   │     │   cli    │   ← coded in parallel (no mutual deps)
│ library  │     │  module  │
└────┬─────┘     └────┬─────┘
     └──────┬─────────┘
            ▼
       ┌──────────┐
       │   main   │              ← waits for both
       └──────────┘
```

**Result:** 5 files generated, review 9/10, 12/12 tests passed, ~40% less wall time than sequential coding, full cost breakdown visible live in the sidebar.

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | `StateGraph` orchestration with conditional revision edges |
| [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) | Claude API with `cache_control` (prompt caching) |
| [LangSmith](https://smith.langchain.com/) | Full trace visibility via `@traceable` on every LLM call |
| [Pydantic v2](https://docs.pydantic.dev/) | Typed state, structured LLM outputs |
| [Streamlit](https://streamlit.io/) | Real-time dashboard (DAG viz, cost, logs) |
| [tenacity](https://github.com/jd/tenacity) | Retry policy with exponential backoff |
| [asyncio](https://docs.python.org/3/library/asyncio.html) + `ThreadPoolExecutor` | Concurrent worker execution |
| [pytest](https://pytest.org) + [ruff](https://github.com/astral-sh/ruff) | Testing, linting, formatting |

---

## 📁 Project Structure

```
parallel-multi-agent-codegen/
├── agents/
│   ├── __init__.py
│   ├── llm_utils.py         # Anthropic SDK + caching + retries + usage capture + @traceable
│   ├── pricing.py           # ★ Per-model cost table + compute_cost()
│   ├── orchestrator.py      # ★ Request parsing + DAG decomposition
│   ├── planner.py           # Planner: structured implementation plan
│   ├── coder.py             # ★ Parallel coder workers + surgical revisions
│   ├── integrator.py        # Merges parallel outputs, resolves conflicts
│   ├── reviewer.py          # Code review and scoring
│   └── tester.py            # Test generation + sandbox execution
├── graph/
│   └── pipeline.py          # LangGraph StateGraph with conditional revision edge
├── models/
│   ├── state.py             # ★ AgentState, TaskDAG, LLMUsage, all Pydantic models
│   └── errors.py            # ★ PipelineError taxonomy (retryable vs terminal)
├── prompts/
│   └── __init__.py          # All agent system prompts (compressed for token efficiency)
├── tests/                   # 102 tests — all mocked, zero API calls
│   ├── conftest.py          # Shared fixtures (sample_dag, state_with_dag, ...)
│   ├── test_models.py       # TaskDAG scheduling, AgentState validation (33 tests)
│   ├── test_pipeline.py     # should_continue() routing logic (12 tests)
│   ├── test_llm_utils.py    # parse_json_response, usage capture, retry, errors (20 tests)
│   ├── test_pricing.py      # Cost math for every model (8 tests)
│   └── test_agents.py       # Agent logic with mocked LLM + surgical revision (21 tests)
├── sandbox/
│   └── Dockerfile           # Isolated test execution environment
├── .github/workflows/
│   └── ci.yml               # Ruff + format + pytest on Py 3.10/3.11/3.12
├── app.py                   # Streamlit UI with real-time DAG viz + cost dashboard
├── pyproject.toml
├── .env.example
├── CLAUDE.md                # Project conventions and cost-optimization rules
└── README.md
```

---

## 🧪 Testing

**102 tests, all passing, zero real API calls** — every LLM call is mocked, so the suite runs in seconds and CI needs no secrets.

```bash
python -m pytest -v                     # all 102 tests
python -m pytest tests/test_models.py   # DAG scheduling (33 tests)
python -m pytest tests/test_pricing.py  # cost math (8 tests)
python -m pytest tests/test_llm_utils.py # parsing + usage + retry (20 tests)
python -m pytest tests/test_agents.py   # agent logic (21 tests)
python -m pytest tests/test_pipeline.py # routing logic (12 tests)

# Lint and format
ruff check .
ruff format --check .
```

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and PR to `main`:

- Python matrix: 3.10, 3.11, 3.12
- `ruff check .`
- `ruff format --check .`
- `python -m pytest -v`

No API key required — the mocked test suite is self-contained.

---

## 🔭 Tracing with LangSmith

Every LLM call across all agents (including parallel workers) is traced when LangSmith is enabled. Tracing uses `@traceable` directly on `call_llm()` — so it works even though the native Anthropic SDK is used instead of LangChain's wrapper.

```bash
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=parallel-multi-agent-codegen
```

Parallel worker calls appear side-by-side in the LangSmith timeline view.

---

## 🧭 Key Design Decisions

### Why DAG-based orchestration?

Real software projects have modular structure. A config module doesn't depend on the routes module. Database models don't depend on CLI argument parsing. Modeling these relationships as a DAG lets us code independent modules simultaneously — the same way a real dev team works.

### Why a separate Integration Agent?

When multiple LLMs code in parallel without seeing each other's output, interface mismatches are inevitable. Worker 1 exports `get_user()` while Worker 2 imports `fetch_user()`. The Integration Agent acts as a merge step — similar to resolving merge conflicts in Git — ensuring the final codebase is coherent.

### Why the native Anthropic SDK instead of LangChain's wrapper?

LangChain's `ChatAnthropic` does not pass `cache_control` to the API, so prompt caching is unavailable through it. The native `anthropic` SDK supports `cache_control={"type": "ephemeral"}` directly, enabling 60–90% cost savings on system prompts. The `@traceable` decorator from LangSmith bridges the gap — every `call_llm()` invocation is still fully traced.

### Why asyncio + ThreadPoolExecutor?

The native Anthropic SDK's `messages.create()` is synchronous. We wrap it in `asyncio.run_in_executor()` with a thread pool to achieve true concurrent API calls across multiple coder workers without needing a fully async client. LangGraph nodes are themselves synchronous, so the worker node spins up its own event loop per dispatch.

### Why surgical revisions instead of full re-runs?

A full re-run burns 100% of the original cost on every revision loop — even when only one file failed review. Surgical revision parses issue text and test errors to identify which artifacts failed, resets only those DAG nodes, and preserves the rest. This is 20–40% cheaper per revision iteration and preserves the work that already passed.

### Why wrap anthropic exceptions in domain types?

The graph layer shouldn't need to `import anthropic` just to route errors. `PipelineError` subclasses (`LLMRateLimitedError`, `LLMTimeoutError`, `LLMBadRequestError`, `ParseFailureError`, `DAGCycleError`, `RetryBudgetExhaustedError`) give the orchestrator clean, dependency-free exception types to match against, and make retry vs. terminal decisions explicit.

---

## 🔮 Future Work

- **Streaming tokens** — stream individual coder outputs as they generate into the Streamlit dashboard
- **Benchmark harness** — reproducible `benchmarks/run.py` over 8 canonical prompts with cost/latency/cache-hit regression tracking
- **Batch API mode** — 50% discount for non-interactive runs
- **Dynamic worker scaling** — adjust parallel workers based on DAG complexity
- **Multi-language support** — extend beyond Python to TypeScript, Go, Rust
- **GitHub integration** — push generated code to a new branch and open a PR
- **Cross-project learning** — feed the v3 [self-evolving-codegen](https://github.com/tathadn/self-evolving-codegen) prompt-evolution loop back into this v2 pipeline

---

## 🧬 How the Trilogy Fits Together

```
         ┌──────────────────────┐
         │  multi-agent-codegen  │  v1 — can agents do this at all?
         │     (sequential)      │
         └───────────┬──────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │ parallel-multi-agent  │  v2 — can we make it fast + cheap + observable?
         │       -codegen        │       ★ this repo
         └───────────┬──────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │ self-evolving-codegen │  v3 — can the pipeline improve itself?
         │                       │
         └──────────────────────┘
```

Each project is self-contained and runnable on its own. If you're reading code, the recommended order is **v1 → v2 → v3** — v2 reuses v1's agent vocabulary, and v3 reuses v2's LangGraph + evaluator skeleton.

---

## 🛠️ Tools Used

- **[Claude](https://claude.ai)** (Anthropic) — AI assistant used during development for architecture decisions, code generation, and debugging.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
