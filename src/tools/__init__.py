"""Tools package — LangChain @tool functions available to the agent."""
from __future__ import annotations

from tools.web_search import web_search_sync as web_search
from tools.file_tools import read_file, write_file, append_file, delete_file
from tools.file_search_tools import search_files
from tools.git_tools import git_status, git_diff, git_commit
from tools.terminal_tools import execute_terminal_command
from tools.pdf_generator import create_pdf
from tools.report_io import prepare_report_output_path


def get_agent_tools() -> list:
    """Return the safe subset of tools the main agent can bind to the LLM.

    Excludes destructive tools (delete_file, git_commit, browser_tools)
    from auto-binding.
    """
    return [
        web_search,
        read_file,
        write_file,
        create_pdf,
        execute_terminal_command,
        git_status,
        git_diff,
        search_files,
    ]


__all__ = [
    "get_agent_tools",
    "web_search",
    "read_file", "write_file", "append_file", "delete_file",
    "search_files",
    "git_status", "git_diff", "git_commit",
    "execute_terminal_command",
    "create_pdf",
    "prepare_report_output_path",
]
