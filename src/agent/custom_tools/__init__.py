"""Legacy compatibility exports for tools."""

from tools.web_search import extract_search_query, web_search, web_search_sync

__all__ = ["web_search", "web_search_sync", "extract_search_query"]
