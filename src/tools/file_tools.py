"""File Manager Tool for Atlas Coding Agent."""

import shutil
from pathlib import Path

from langchain.tools import tool

from tools._workspace import _resolve_workspace_path, _truncate_text

def _project_path(filepath: str) -> Path:
    return _resolve_workspace_path(filepath)


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    try:
        path = _project_path(filepath)
        if not path.exists():
            return f"Error reading file: File not found: {path}"
        
        file_size = path.stat().st_size
        MAX_SIZE = 12000
        with path.open(encoding='utf-8') as f:
            if file_size > MAX_SIZE:
                content = f.read(MAX_SIZE)
                return _truncate_text(content, MAX_SIZE)
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    try:
        path = _project_path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Backup before overwrite
        if path.exists():
            shutil.copy2(path, f"{path}.bak")
        with path.open('w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path} (backup saved as {path}.bak if it existed)."
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def append_file(filepath: str, content: str) -> str:
    """Append content to a file."""
    try:
        path = _project_path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully appended to {path}."
    except Exception as e:
        return f"Error appending to file: {e}"

@tool
def delete_file(filepath: str) -> str:
    """Delete a file."""
    try:
        path = _project_path(filepath)
        path.unlink()
        return f"Successfully deleted {path}."
    except Exception as e:
        return f"Error deleting file: {e}"

__all__ = ["read_file", "write_file", "append_file", "delete_file"]
