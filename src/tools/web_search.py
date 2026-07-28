"""Web search via DuckDuckGo (no API key)."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from utils.async_utils import run_in_thread


def extract_search_query(text: str) -> str:
    """Pull a clean search query from natural-language input."""
    query = (text or "").strip()
    query = re.sub(
        r"(?i)^(please\s+)?(search the web for|search online for|search for|look up|find|browse)\s*",
        "",
        query,
    )
    query = re.sub(
        r"^(من فضلك\s+)?(ابحث في الويب عن|ابحث عن|ابحث|تصفح)\s*",
        "",
        query,
    )
    query = re.sub(r"(?i)^(what is|who is|when is|where is)\s+", "", query)
    return query.strip(" ?؟.") or (text or "").strip()


def wants_profile_screenshot(text: str) -> bool:
    """True when user asks for screenshot/snap/image of profile pages."""
    raw = (text or "").strip()
    lowered = raw.lower()
    markers = (
        "screenshot",
        "screen shot",
        "snapshot",
        "preview image",
        "show me image",
        "لقطة شاشة",
        "سكرين شوت",
        "صورة الصفحة",
        "صوره الصفحه",
        "صورة للحساب",
    )
    return any(marker in lowered or marker in raw for marker in markers)


def extract_links(search_blob: str) -> list[str]:
    """Extract unique http(s) links from formatted web search text."""
    links = re.findall(r"https?://[^\s)]+", search_blob or "")
    out: list[str] = []
    seen: set[str] = set()
    for link in links:
        cleaned = link.rstrip(".,;")
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def screenshot_markdown(links: list[str], *, max_items: int = 3) -> str:
    """Build markdown image links using thum.io screenshots."""
    if not links:
        return ""
    lines = ["", "### Page screenshots", ""]
    for link in links[:max_items]:
        shot = f"https://image.thum.io/get/width/1200/{link}"
        lines.append(f"- {link}")
        lines.append(f"  ![screenshot]({shot})")
    return "\n".join(lines).strip()


def web_search_sync(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return formatted results."""
    search_query = extract_search_query(query)
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        search_tool = DuckDuckGoSearchResults(
            num_results=max_results,
            output_format="list",
            keys_to_include=["title", "snippet", "link"],
        )
        raw = search_tool.invoke(search_query)

        if isinstance(raw, tuple):
            results, _ = raw
        else:
            results = raw

        if not results:
            return f"No web results found for '{search_query}'"

        if isinstance(results, str):
            return f"**Web Search:** {search_query}\n\n{results}"

        lines = [f"**Web Search:** {search_query}", ""]
        for index, item in enumerate(results[:max_results], start=1):
            if not isinstance(item, dict):
                lines.append(f"{index}. {item}")
                continue
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            lines.append(f"{index}. **{title}**")
            if snippet:
                lines.append(f"   {snippet}")
            if link:
                lines.append(f"   {link}")
            lines.append("")

        return "\n".join(lines).strip()
    except Exception as exc:  # noqa: BLE001
        return (
            f"Web search error: {exc}. "
            "Install duckduckgo-search: uv add duckduckgo-search"
        )


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for current information, news, and facts.

    Use when the user asks for live/online information that is not in the
    local knowledge database or uploaded documents.

    Args:
        query: The search query.
        max_results: Maximum number of results (default 5).
    """
    return await run_in_thread(web_search_sync, query, max_results)


__all__ = [
    "web_search",
    "web_search_sync",
    "extract_search_query",
    "wants_profile_screenshot",
    "extract_links",
    "screenshot_markdown",
]
