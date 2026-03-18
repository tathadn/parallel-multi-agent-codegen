# ⚡ Parallel Multi-Agent Code Generator

> Describe what you want to build — an **orchestrator agent** decomposes it into a DAG of parallel tasks, dispatches concurrent AI coding agents, merges their outputs, then reviews and tests the result automatically.

**This is v2 of [multi-agent-codegen](https://github.com/tathadn/multi-agent-codegen)**, rebuilt from the ground up with a parallel orchestration architecture. The original pipeline was sequential (Orchestrator → Planner → Coder → Reviewer → Tester). This version introduces **DAG-based task decomposition** and **concurrent code generation** — independent modules are coded simultaneously by parallel worker agents, dramatically reducing end-to-end latency for complex projects.

---

## What's New in v2

| Feature | v1 (Sequential) | v2 (Parallel) |
|---|---|---|
| **Coding model** | Single coder agent | Multiple parallel coder workers |
| **Task planning** | Linear step list | Directed Acyclic Graph (DAG) |
| **Orchestration** | Simple status tracking | DAG decomposition + dependency analysis |
| **Integration** | N/A (single coder) | Dedicated Integration Agent merges outputs |
| **Parallelism** | None | Independent modules coded concurrently |
| **Observability** | Basic status | Real-time DAG viz + timing breakdown |

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│   Orchestrator   │  Validates request, sets pipeline status
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Planner      │  Produces structured plan: steps, files, dependencies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orchestrator    │  Decomposes plan → Task DAG
│  (DAG Builder)   │  Identifies parallelizable modules
└────────┬────────┘
         │
    ┌────┴─────┬──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│   ← parallel coding via asyncio
└───┬────┘ └───┬────┘ └───┬────┘
    └────┬─────┴──────────┘
         │
         ▼
┌─────────────────┐
│   Integrator     │  Merges parallel outputs, resolves conflicts
└────────┬────────┘
         │
         ▼
┌─────────────────┐ ◄────────────────────────────┐
│    Reviewer      │  Scores code 0-10            │  revision loop
└────────┬────────┘                               │  (configurable)
         │                                        │
         ▼                                        │
┌─────────────────┐  tests fail or review         │
│     Tester       │  not approved? ──────────────┘
└────────┬────────┘
         │  all green
         ▼
    Final Output
    (code files + plan + review + test report)
```

### How the Orchestrator Builds the DAG

The Orchestrator is the central intelligence of the system. After the Planner produces a step-by-step implementation plan, the Orchestrator:

1. **Analyzes file dependencies** — identifies which modules import from each other
2. **Groups related files** — tightly coupled files (e.g., a model and its schema) become a single DAG node
3. **Builds dependency edges** — if module A imports from module B, there's an edge B → A
4. **Maximizes parallelism** — nodes with no mutual dependencies are dispatched to concurrent workers

For example, given a request for a FastAPI app with user auth:

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

Three modules code in parallel instead of five running sequentially — a significant speedup.

---

## Agent Roles

| Agent | Model | Role |
|---|---|---|
| **Orchestrator** | claude-sonnet-4-20250514 | Validates request, decomposes plan into parallel task DAG |
| **Planner** | claude-sonnet-4-20250514 | Generates structured plan: objective, steps, files, dependencies |
| **Coder Workers** | claude-sonnet-4-20250514 | Generate code for assigned DAG nodes (run concurrently) |
| **Integrator** | claude-sonnet-4-20250514 | Merges parallel outputs, resolves import conflicts |
| **Reviewer** | claude-sonnet-4-20250514 | Scores code quality (0-10), identifies issues |
| **Tester** | configurable (sidebar) | Generates pytest tests and runs them in sandbox |

### Shared State

All agents read from and write to a Pydantic `AgentState`:

```python
class AgentState(BaseModel):
    user_request: str
    plan:         Optional[Plan]
    task_dag:     Optional[TaskDAG]       # NEW: parallel task graph
    artifacts:    list[CodeArtifact]
    review:       Optional[ReviewFeedback]
    test_result:  Optional[TestResult]
    status:       TaskStatus
    iteration:    int
    max_iterations: int
    logs:         list[str]               # NEW: timestamped pipeline log
    timings:      dict[str, float]        # NEW: per-phase timing
```

### TaskDAG — The Core Data Structure

```python
class TaskNode(BaseModel):
    id:         str                    # unique short ID
    name:       str                    # module name
    description: str                   # what this module does
    files:      list[str]              # files to generate
    depends_on: list[str]              # IDs of prerequisite nodes
    status:     WorkerStatus           # IDLE → RUNNING → DONE
    worker_id:  Optional[str]          # assigned worker
    started_at: Optional[float]        # timing
    finished_at: Optional[float]

class TaskDAG(BaseModel):
    nodes: list[TaskNode]

    def ready_nodes(self) -> list[TaskNode]:
        """Nodes with all deps satisfied — ready for parallel dispatch."""
        ...

    def all_done(self) -> bool: ...
    def topological_order(self) -> list[TaskNode]: ...
```

---

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- Docker (optional — required only for sandbox test execution)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/tathadn/parallel-multi-agent-codegen.git
cd parallel-multi-agent-codegen

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Set your API key
cp .env.example .env
# edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Example

**Input:**

```
A Python library for graph algorithms (BFS, DFS, Dijkstra) with a CLI interface
```

**Orchestrator decomposes into DAG:**

```
┌─────────┐     ┌─────────┐
│  graph   │     │  cli    │   ← coded in parallel (no mutual deps)
│ library  │     │ module  │
└────┬────┘     └────┬────┘
     └─────┬────────┘
           ▼
      ┌─────────┐
      │  main   │              ← waits for both
      └─────────┘
```

**Result:** 5 files generated, review 9/10, 12/12 tests passed — in ~40% less time than sequential coding.

---

## Tech Stack

| Library | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent orchestration with conditional edges |
| [LangChain Anthropic](https://github.com/langchain-ai/langchain) | Claude API integration |
| [Pydantic v2](https://docs.pydantic.dev/) | Typed state, structured LLM outputs |
| [Streamlit](https://streamlit.io/) | Real-time web dashboard |
| [asyncio](https://docs.python.org/3/library/asyncio.html) | Concurrent worker execution |

---

## Project Structure

```
parallel-multi-agent-codegen/
├── agents/
│   ├── __init__.py
│   ├── llm_utils.py          # Shared LLM call + JSON parsing utilities
│   ├── orchestrator.py        # ⭐ Orchestrator: request parsing + DAG decomposition
│   ├── planner.py             # Planner: structured implementation plan
│   ├── coder.py               # ⭐ Parallel coder workers with asyncio
│   ├── integrator.py          # Integration agent: merges parallel outputs
│   ├── reviewer.py            # Code review and scoring
│   └── tester.py              # Test generation + sandbox execution
├── graph/
│   ├── __init__.py
│   └── pipeline.py            # LangGraph StateGraph with parallel nodes
├── models/
│   ├── __init__.py
│   └── state.py               # Pydantic models: AgentState, TaskDAG, etc.
├── prompts/
│   └── __init__.py            # All agent system prompts and templates
├── sandbox/
│   └── Dockerfile             # Isolated test execution environment
├── app.py                     # Streamlit UI with real-time DAG visualization
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Key Design Decisions

### Why DAG-based orchestration?

Real software projects have modular structure. A config module doesn't depend on the routes module. Database models don't depend on CLI argument parsing. By modeling these relationships as a DAG, we can code independent modules simultaneously — the same way a real dev team works.

### Why a separate Integration Agent?

When multiple LLMs code in parallel without seeing each other's output, interface mismatches are inevitable. Worker 1 might export `get_user()` while Worker 2 imports `fetch_user()`. The Integration Agent acts as a merge step — similar to resolving merge conflicts in Git — ensuring the final codebase is coherent.

### Why asyncio + ThreadPoolExecutor?

LangChain's `ChatAnthropic.invoke()` is synchronous. We wrap it in `asyncio.run_in_executor()` with a thread pool to achieve true concurrent API calls. This gives us the parallelism benefits without requiring a fully async LLM client.

---

## Tracing with LangSmith

Every LLM call across all agents (including parallel workers) is traced when LangSmith is enabled:

```bash
# Add to .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=parallel-multi-agent-codegen
```

You can see parallel worker calls running simultaneously in the LangSmith timeline view.

---

## Future Work

- **Streaming tokens** — stream individual coder outputs as they generate
- **Dynamic worker scaling** — adjust parallel workers based on DAG complexity
- **Multi-language support** — extend beyond Python to TypeScript, Go, Rust
- **GitHub integration** — push generated code to a new branch and open a PR
- **Agent memory** — learn from past runs to improve code quality over time
- **WebSocket dashboard** — replace polling with real-time push updates

---

## Evolution from v1

This project started as a [sequential multi-agent pipeline](https://github.com/tathadn/multi-agent-codegen) where agents ran one at a time. The key insight for v2 was that **code generation is inherently parallelizable** — most real projects consist of loosely-coupled modules that can be built simultaneously.

The Orchestrator Agent is the architectural centerpiece: it uses an LLM to analyze the Planner's output, identify true code-level dependencies (imports, shared types), and build a DAG that maximizes concurrent execution while respecting data flow constraints.

---

## License

MIT
