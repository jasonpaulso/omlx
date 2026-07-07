# SPDX-License-Identifier: Apache-2.0
"""Tests for the code-exec watchdog in livecodebench.py / humaneval.py.

Background: subprocess.run(timeout=...) already bounds the DIRECT child by
wall clock (verified: a sleep-forever / stdin-blocking program is killed at
the configured timeout, not left running). The real gap was that the timeout
handler only signals the direct child pid — a grandchild process spawned by
generated code survives as an orphan. These tests cover the fix:
start_new_session=True + os.killpg on timeout, and stdin=DEVNULL for
humaneval (which previously inherited the parent's stdin).
"""

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

import omlx.eval.humaneval as humaneval
import omlx.eval.livecodebench as livecodebench

SHORT_TIMEOUT = 2  # seconds — patched in place of the real 15s/30s constants


@pytest.fixture(autouse=True)
def _short_timeouts(monkeypatch):
    """Patch both modules' timeout constants so hang tests stay fast."""
    monkeypatch.setattr(livecodebench, "EXEC_TIMEOUT_SECONDS", SHORT_TIMEOUT)
    monkeypatch.setattr(humaneval, "EXEC_TIMEOUT_SECONDS", SHORT_TIMEOUT)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class TestLiveCodeBenchExecWatchdog:
    def test_sleep_forever_is_killed_at_timeout(self):
        start = time.time()
        stdout, success, error = livecodebench._execute_code(
            "import time\ntime.sleep(9999)\n", stdin_input=""
        )
        elapsed = time.time() - start

        assert success is False
        assert error == "Execution timed out"
        assert elapsed < SHORT_TIMEOUT + 5, "must not hang past the timeout"

    def test_input_call_resolves_instantly_not_by_hanging(self):
        """livecodebench always writes+closes stdin, so input() hits EOF
        immediately rather than blocking until the timeout — this is
        stronger than "killed at timeout" and confirms it stays that way.
        """
        start = time.time()
        stdout, success, error = livecodebench._execute_code(
            "x = input()\nprint(x)\n", stdin_input=""
        )
        elapsed = time.time() - start

        assert success is False  # EOFError, no input available
        assert elapsed < 1.0, "must resolve via EOF, not wait for the timeout"

    def test_grandchild_process_is_killed_with_parent(self, tmp_path):
        """A generated program that spawns its own subprocess must not leave
        an orphan running after the timeout fires (the bug this branch fixes).
        """
        marker = tmp_path / "grandchild.pid"
        grandchild_script = tmp_path / "grandchild.py"
        grandchild_script.write_text(
            "import os, time\n"
            f"open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        code = (
            "import subprocess, time\n"
            f"subprocess.Popen(['python3', {str(grandchild_script)!r}])\n"
            "time.sleep(9999)\n"
        )

        start = time.time()
        _, success, error = livecodebench._execute_code(code, stdin_input="")
        elapsed = time.time() - start

        assert success is False
        assert error == "Execution timed out"
        assert elapsed < SHORT_TIMEOUT + 5

        # Give the grandchild a moment to have written its marker before
        # asserting it's gone — it starts concurrently with the parent.
        for _ in range(20):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.exists(), "grandchild never started — test setup is broken"

        grandchild_pid = int(marker.read_text())
        assert not _pid_alive(
            grandchild_pid
        ), "grandchild survived the parent's timeout kill — orphan leaked"

    def test_normal_fast_program_passes(self):
        stdout, success, error = livecodebench._execute_code(
            "x = input()\nprint(int(x) * 2)\n", stdin_input="21"
        )
        assert success is True
        assert stdout.strip() == "42"

    def test_public_path_hang_marks_question_failed_and_run_continues(self):
        """Drive the full BaseBenchmark.run() path with a stub engine whose
        generated code hangs — the question must score as incorrect and the
        run must complete, not hang.
        """
        bench = livecodebench.LiveCodeBenchBenchmark()
        item = {
            "id": "q1",
            "title": "hang",
            "description": "n/a",
            "inputs": ["1\n"],
            "outputs": ["1"],
            "difficulty": "easy",
            "starter_code": "",
        }

        async def fake_chat(messages, **kwargs):
            return SimpleNamespace(text="```python\nimport time\ntime.sleep(9999)\n```")

        engine = SimpleNamespace(chat=fake_chat)

        start = time.time()
        result = asyncio.run(bench.run(engine, [item]))
        elapsed = time.time() - start

        assert elapsed < SHORT_TIMEOUT + 10, "run() must not hang on a stuck question"
        assert result.total_questions == 1
        assert result.correct_count == 0
        assert result.question_results[0].correct is False

    def test_public_path_normal_program_scores_correct(self):
        bench = livecodebench.LiveCodeBenchBenchmark()
        item = {
            "id": "q1",
            "title": "double",
            "description": "n/a",
            "inputs": ["21\n"],
            "outputs": ["42"],
            "difficulty": "easy",
            "starter_code": "",
        }

        async def fake_chat(messages, **kwargs):
            return SimpleNamespace(
                text="```python\nx = input()\nprint(int(x) * 2)\n```"
            )

        engine = SimpleNamespace(chat=fake_chat)

        result = asyncio.run(bench.run(engine, [item]))

        assert result.total_questions == 1
        assert result.correct_count == 1
        assert result.question_results[0].correct is True


class TestHumanEvalExecWatchdog:
    def test_sleep_forever_is_killed_at_timeout(self):
        start = time.time()
        success, error = humaneval._execute_with_tests(
            "def f():\n    import time\n    time.sleep(9999)\n    return 1\n",
            "def check(candidate):\n    assert candidate() == 1\n",
            "f",
        )
        elapsed = time.time() - start

        assert success is False
        assert error == "Execution timed out"
        assert elapsed < SHORT_TIMEOUT + 5

    def test_input_call_resolves_instantly_via_devnull(self):
        """humaneval previously inherited the parent's stdin (no input=/
        stdin= at all), so a blocking input() call could hang until the
        timeout depending on what the parent's stdin was. stdin=DEVNULL
        makes this resolve via immediate EOF instead — stronger than
        "bounded by the timeout".
        """
        start = time.time()
        success, error = humaneval._execute_with_tests(
            "def f():\n    return input()\n",
            "def check(candidate):\n    assert candidate() == ''\n",
            "f",
        )
        elapsed = time.time() - start

        assert success is False  # EOFError reading from /dev/null
        assert elapsed < 1.0, "must resolve via EOF, not wait for the timeout"

    def test_grandchild_process_is_killed_with_parent(self, tmp_path):
        marker = tmp_path / "grandchild.pid"
        grandchild_script = tmp_path / "grandchild.py"
        grandchild_script.write_text(
            "import os, time\n"
            f"open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        code = (
            "def f():\n"
            "    import subprocess, time\n"
            f"    subprocess.Popen(['python3', {str(grandchild_script)!r}])\n"
            "    time.sleep(9999)\n"
            "    return 1\n"
        )

        success, error = humaneval._execute_with_tests(
            code, "def check(candidate):\n    assert candidate() == 1\n", "f"
        )

        assert success is False
        assert error == "Execution timed out"

        for _ in range(20):
            if marker.exists():
                break
            time.sleep(0.1)
        assert marker.exists(), "grandchild never started — test setup is broken"

        grandchild_pid = int(marker.read_text())
        assert not _pid_alive(
            grandchild_pid
        ), "grandchild survived the parent's timeout kill — orphan leaked"

    def test_normal_fast_program_passes(self):
        success, error = humaneval._execute_with_tests(
            "def f(x):\n    return x * 2\n",
            "def check(candidate):\n    assert candidate(21) == 42\n",
            "f",
        )
        assert success is True
        assert error == ""

    def test_public_path_hang_marks_question_failed_and_run_continues(self):
        bench = humaneval.HumanEvalBenchmark()
        item = {
            "id": "HumanEval/0",
            "prompt": "def f(x):\n",
            "test": ("def check(candidate):\n" "    assert candidate(1) == 1\n"),
            "entry_point": "f",
            "question": "def f(x):\n",
        }

        async def fake_chat(messages, **kwargs):
            return SimpleNamespace(
                text="```python\ndef f(x):\n    import time\n    time.sleep(9999)\n```"
            )

        engine = SimpleNamespace(chat=fake_chat)

        start = time.time()
        result = asyncio.run(bench.run(engine, [item]))
        elapsed = time.time() - start

        assert elapsed < SHORT_TIMEOUT + 10, "run() must not hang on a stuck question"
        assert result.total_questions == 1
        assert result.correct_count == 0
        assert result.question_results[0].correct is False
