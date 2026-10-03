"""Git Integration Tool for Atlas Coding Agent."""

import subprocess

from langchain.tools import tool

from tools._workspace import _resolve_workspace_path, _truncate_text


def _cwd(cwd: str = ".") -> str:
    return str(_resolve_workspace_path(cwd))


@tool
def git_status(cwd: str = ".") -> str:
    """Return the current git status."""
    result = subprocess.run(["git", "status"], cwd=_cwd(cwd), capture_output=True, text=True)
    output = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    return _truncate_text(output)

@tool
def git_diff(cwd: str = ".") -> str:
    """Return the current git diff."""
    result = subprocess.run(["git", "diff"], cwd=_cwd(cwd), capture_output=True, text=True)
    output = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    return _truncate_text(output)

@tool
def git_commit(message: str, cwd: str = ".") -> str:
    """Commit all current changes with the given message."""
    effective_cwd = _cwd(cwd)
    subprocess.run(["git", "add", "."], cwd=effective_cwd, capture_output=True)
    result = subprocess.run(["git", "commit", "-m", message], cwd=effective_cwd, capture_output=True, text=True)
    output = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    return _truncate_text(output)

__all__ = ["git_status", "git_diff", "git_commit"]
