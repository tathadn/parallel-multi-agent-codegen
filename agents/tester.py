"""
Tester Agent — generates pytest files and runs them in an isolated sandbox.

In production this uses Docker for sandboxed execution.  For development
without Docker, it falls back to subprocess-based execution in a temp directory.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from models.state import AgentState, CodeArtifact, TaskStatus, TestResult
from prompts import TESTER_PROMPT, TESTER_SYSTEM

from .llm_utils import call_llm, parse_json_response


def tester_agent(state: AgentState, model_name: str = "claude-haiku-4-5-20241022") -> AgentState:
    """Generate test files and execute them."""
    t0 = time.time()
    state.status = TaskStatus.TESTING
    state.log("🧪 Tester: generating tests")

    # Generate test files via LLM
    artifacts_text = json.dumps(
        [{"filename": a.filename, "content": a.content} for a in state.artifacts],
        indent=2,
    )

    prompt = TESTER_PROMPT.format(
        user_request=state.user_request,
        artifacts=artifacts_text,
    )

    raw = call_llm(system=TESTER_SYSTEM, prompt=prompt, model_name=model_name)

    try:
        test_files_data = parse_json_response(raw)
    except ValueError as e:
        state.log(f"⚠️ Tester: failed to generate tests — {e}")
        state.test_result = TestResult(
            passed=False, error_output=f"Test generation failed: {e}"
        )
        state.timings["tester"] = round(time.time() - t0, 2)
        return state

    if isinstance(test_files_data, dict):
        test_files_data = [test_files_data]

    test_artifacts = [
        CodeArtifact(
            filename=f.get("filename", "test_unknown.py"),
            language=f.get("language", "python"),
            content=f.get("content", ""),
        )
        for f in test_files_data
    ]

    state.log(f"📝 Tester: generated {len(test_artifacts)} test file(s)")

    # Execute tests in sandbox
    state.test_result = _run_tests(state.artifacts, test_artifacts)

    emoji = "✅" if state.test_result.passed else "❌"
    state.log(
        f"{emoji} Tester: {state.test_result.passed_tests}/{state.test_result.total_tests} "
        f"tests passed"
    )
    state.timings["tester"] = round(time.time() - t0, 2)
    return state


def _run_tests(
    code_artifacts: list[CodeArtifact],
    test_artifacts: list[CodeArtifact],
) -> TestResult:
    """
    Run tests in a temporary directory.

    In production, this would use Docker (see sandbox/Dockerfile).
    For development, we use a subprocess in a temp dir.
    """
    with tempfile.TemporaryDirectory(prefix="codegen_test_") as tmpdir:
        # Write all code files
        for artifact in code_artifacts + test_artifacts:
            filepath = Path(tmpdir) / artifact.filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(artifact.content)

        # Run pytest
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-v", "--tb=short", tmpdir],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmpdir,
            )

            output = result.stdout + result.stderr

            # Parse pytest output for counts
            passed = failed = 0
            for line in output.splitlines():
                if " passed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        if part == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

            total = passed + failed
            return TestResult(
                passed=(result.returncode == 0),
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                error_output=output if result.returncode != 0 else "",
                test_files=[a.filename for a in test_artifacts],
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                error_output="Tests timed out after 60 seconds",
            )
        except FileNotFoundError:
            return TestResult(
                passed=False,
                error_output="pytest not found — install with: pip install pytest",
            )
