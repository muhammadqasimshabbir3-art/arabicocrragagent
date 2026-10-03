"""Terminal Tool for Atlas Coding Agent."""

import subprocess
import time
from typing import Any, Dict
import os

from langchain.tools import tool

from tools._workspace import _resolve_workspace_path, _truncate_text


@tool
def execute_terminal_command(command: str, cwd: str = ".", timeout: int = 60) -> Dict[str, Any]:
    """Execute a terminal command safely and capture output.
    
    Args:
        command: The shell command to run (e.g. 'pytest', 'uv add', 'python script.py')
        cwd: The working directory for the command.
        timeout: Maximum execution time in seconds.
        
    Returns:
        Dictionary containing stdout, stderr, exit code, and execution time.
    """
    start_time = time.time()
    try:
        effective_cwd = str(_resolve_workspace_path(cwd))
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "execution_time": time.time() - start_time,
            "cwd": os.getenv("PROJECT_DIR", "."),
            "error": str(e),
        }
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=effective_cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        execution_time = time.time() - start_time
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        stdout = _truncate_text(stdout)
        stderr = _truncate_text(stderr)
            
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "execution_time": execution_time,
            "cwd": effective_cwd,
            "error": None
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        
        stdout = _truncate_text(stdout)
        stderr = _truncate_text(stderr)
            
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": -1,
            "execution_time": time.time() - start_time,
            "cwd": effective_cwd,
            "error": "Command timed out"
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "execution_time": time.time() - start_time,
            "cwd": effective_cwd,
            "error": str(e)
        }

__all__ = ["execute_terminal_command"]
