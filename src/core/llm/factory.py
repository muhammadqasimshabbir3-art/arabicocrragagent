"""Provider-agnostic chat model factory (Groq default)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from core.config.settings import get_settings
from core.utils.async_utils import run_in_thread

_model_instance = None


def _resolve_model_name(settings) -> str:
    """Prefer LLM_MODEL, then provider-specific env, then groq_model setting."""
    return (
        (os.getenv("LLM_MODEL") or "").strip()
        or settings.llm_model
        or settings.groq_model
    )


def _timeout_kwargs(settings) -> dict[str, Any]:
    timeout = max(1, int(settings.llm_timeout_seconds))
    return {"timeout": timeout, "request_timeout": timeout}


def get_chat_model(*, temperature: float | None = None):
    """Return the configured chat model (cached for default temperature)."""
    global _model_instance
    settings = get_settings()
    temp = settings.llm_temperature if temperature is None else temperature
    model_name = _resolve_model_name(settings)
    timeouts = _timeout_kwargs(settings)

    if temperature is None and _model_instance is not None:
        return _model_instance

    provider = settings.llm_provider.lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temp,
            "api_key": settings.groq_api_key,
        }
        # ChatGroq accepts request_timeout / timeout depending on version.
        for key, value in timeouts.items():
            kwargs[key] = value
        try:
            model = ChatGroq(**kwargs)
        except TypeError:
            kwargs.pop("timeout", None)
            kwargs.pop("request_timeout", None)
            model = ChatGroq(
                model=model_name,
                temperature=temp,
                api_key=settings.groq_api_key,
            )
        if temperature is None:
            _model_instance = model
        return model

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=openai requires langchain-openai. "
                "Install with: uv sync --extra openai"
            ) from exc
        try:
            model = ChatOpenAI(
                model=model_name,
                temperature=temp,
                request_timeout=timeouts["request_timeout"],
            )
        except TypeError:
            model = ChatOpenAI(model=model_name, temperature=temp)
        if temperature is None:
            _model_instance = model
        return model

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = ChatOllama(model=model_name, temperature=temp)
        if temperature is None:
            _model_instance = model
        return model

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=gemini requires langchain-google-genai. "
                "Install with: uv add langchain-google-genai"
            ) from exc
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temp,
                timeout=timeouts["timeout"],
            )
        except TypeError:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=temp)
        if temperature is None:
            _model_instance = model
        return model

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider}. Use groq, openai, ollama, or gemini."
    )


async def get_chat_model_async(*, temperature: float | None = None):
    """Return chat model while keeping blocking imports off event loop."""
    return await run_in_thread(get_chat_model, temperature=temperature)


def invoke_plain(messages: list[dict[str, str]]) -> str:
    """Invoke the chat model with role/content dicts; return text."""
    from langchain_core.messages import HumanMessage, SystemMessage

    settings = get_settings()
    timeout = max(1, int(settings.llm_timeout_seconds))

    lc_messages: list[Any] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    def _call() -> str:
        response = get_chat_model().invoke(lc_messages)
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"LLM invoke timed out after {timeout}s"
            ) from exc
