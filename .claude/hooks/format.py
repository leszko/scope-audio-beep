#!/usr/bin/env python3
"""Claude Code formatter hook — runs ruff on Python files."""

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True)
    except FileNotFoundError:
        pass


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return

    path_str = data.get("tool_input", {}).get("file_path")
    if not path_str:
        return

    path = Path(path_str).resolve()
    if not path.exists() or path.suffix != ".py":
        return

    repo_root = Path(__file__).parent.parent.parent
    run(["uv", "run", "ruff", "check", "--fix", str(path)], repo_root)
    run(["uv", "run", "ruff", "format", str(path)], repo_root)


if __name__ == "__main__":
    main()
