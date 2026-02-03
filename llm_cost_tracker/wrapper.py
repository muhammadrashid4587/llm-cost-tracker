"""
Transparent wrapper / proxy functions for OpenAI and Anthropic clients.

The wrappers intercept API responses, extract token usage from the response
objects, record costs via a :class:`CostTracker`, and return the original
response unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from .tracker import CostTracker


# ---------------------------------------------------------------------------
# Generic proxy that intercepts attribute access
# ---------------------------------------------------------------------------

class _Proxy:
    """
    A transparent proxy around an arbitrary object.  Attribute access is
    forwarded to the wrapped object unless overridden in a subclass.
    """

    def __init__(self, wrapped: Any) -> None:
        object.__setattr__(self, "_wrapped", wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_wrapped"))


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI
# ═══════════════════════════════════════════════════════════════════════════

class _OpenAICompletionsProxy(_Proxy):
    """Intercepts ``client.chat.completions.create(...)``."""

    def __init__(self, wrapped: Any, tracker: CostTracker) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, "_tracker", tracker)

    def create(self, **kwargs: Any) -> Any:
        wrapped = object.__getattribute__(self, "_wrapped")
        tracker: CostTracker = object.__getattribute__(self, "_tracker")

        response = wrapped.create(**kwargs)
        _record_openai_usage(response, tracker, kwargs.get("model", "unknown"))
        return response


class _OpenAIChatProxy(_Proxy):
    """Replaces ``client.chat`` so that ``.completions`` returns our proxy."""

    def __init__(self, wrapped: Any, tracker: CostTracker) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, "_tracker", tracker)

    @property
    def completions(self) -> _OpenAICompletionsProxy:
        tracker: CostTracker = object.__getattribute__(self, "_tracker")
        wrapped = object.__getattribute__(self, "_wrapped")
        return _OpenAICompletionsProxy(wrapped.completions, tracker)


class _OpenAIClientProxy(_Proxy):
    """Top-level proxy returned by :func:`track_openai`."""

    def __init__(self, wrapped: Any, tracker: CostTracker) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, "_tracker", tracker)

    @property
    def chat(self) -> _OpenAIChatProxy:
        tracker: CostTracker = object.__getattribute__(self, "_tracker")
        wrapped = object.__getattribute__(self, "_wrapped")
        return _OpenAIChatProxy(wrapped.chat, tracker)


def _record_openai_usage(
    response: Any,
    tracker: CostTracker,
    model_hint: str,
) -> None:
    """Extract token usage from an OpenAI ChatCompletion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    model = getattr(response, "model", model_hint) or model_hint
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    # OpenAI may include prompt_tokens_details with cached_tokens
    cached_input_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_input_tokens = getattr(details, "cached_tokens", 0) or 0

    tracker.record(
        model=model,
        provider="openai",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def track_openai(client: Any, tracker: CostTracker) -> Any:
    """
    Wrap an OpenAI client so every ``chat.completions.create`` call is
    automatically tracked.

    Parameters
    ----------
    client : openai.OpenAI
        An instantiated OpenAI client.
    tracker : CostTracker
        The tracker that will accumulate usage records.

    Returns
    -------
    A proxy object that behaves identically to *client* but records costs.
    """
    return _OpenAIClientProxy(client, tracker)


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════════════════════════

class _AnthropicMessagesProxy(_Proxy):
    """Intercepts ``client.messages.create(...)``."""

    def __init__(self, wrapped: Any, tracker: CostTracker) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, "_tracker", tracker)

    def create(self, **kwargs: Any) -> Any:
        wrapped = object.__getattribute__(self, "_wrapped")
        tracker: CostTracker = object.__getattribute__(self, "_tracker")

        response = wrapped.create(**kwargs)
        _record_anthropic_usage(response, tracker, kwargs.get("model", "unknown"))
        return response


class _AnthropicClientProxy(_Proxy):
    """Top-level proxy returned by :func:`track_anthropic`."""

    def __init__(self, wrapped: Any, tracker: CostTracker) -> None:
        super().__init__(wrapped)
        object.__setattr__(self, "_tracker", tracker)

    @property
    def messages(self) -> _AnthropicMessagesProxy:
        tracker: CostTracker = object.__getattribute__(self, "_tracker")
        wrapped = object.__getattribute__(self, "_wrapped")
        return _AnthropicMessagesProxy(wrapped.messages, tracker)


def _record_anthropic_usage(
    response: Any,
    tracker: CostTracker,
    model_hint: str,
) -> None:
    """Extract token usage from an Anthropic Message response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    model = getattr(response, "model", model_hint) or model_hint
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cached_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    tracker.record(
        model=model,
        provider="anthropic",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def track_anthropic(client: Any, tracker: CostTracker) -> Any:
    """
    Wrap an Anthropic client so every ``messages.create`` call is
    automatically tracked.

    Parameters
    ----------
    client : anthropic.Anthropic
        An instantiated Anthropic client.
    tracker : CostTracker
        The tracker that will accumulate usage records.

    Returns
    -------
    A proxy object that behaves identically to *client* but records costs.
    """
    return _AnthropicClientProxy(client, tracker)
