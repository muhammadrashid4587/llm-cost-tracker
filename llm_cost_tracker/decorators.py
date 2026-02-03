"""
Decorator for tracking costs of functions that call LLM APIs.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, cast

from .tracker import CostTracker

F = TypeVar("F", bound=Callable[..., Any])


def track_cost(
    tracker: CostTracker,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator that records LLM cost based on the return value of a function.

    The decorated function **must** return an object that carries a ``usage``
    attribute (like an OpenAI ``ChatCompletion`` or Anthropic ``Message``).
    The decorator inspects that attribute, computes cost, and stores a
    record in *tracker*.

    Parameters
    ----------
    tracker : CostTracker
        The tracker instance to record costs into.
    model : str, optional
        Override the model name.  If ``None``, it is read from the
        response's ``.model`` attribute.
    provider : str, optional
        Override the provider (``"openai"``, ``"anthropic"``). If ``None``,
        a best-effort guess is made from the response object.

    Example
    -------
    ::

        tracker = CostTracker()

        @track_cost(tracker)
        def ask(question: str):
            return openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
            )

        answer = ask("What is 2+2?")
        print(tracker.summary())
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            response = fn(*args, **kwargs)
            _auto_record(response, tracker, model_override=model, provider_override=provider)
            return response

        return cast(F, wrapper)

    return decorator


def _auto_record(
    response: Any,
    tracker: CostTracker,
    *,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
) -> None:
    """Inspect a response object and record its usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    resolved_model = model_override or getattr(response, "model", "unknown") or "unknown"
    resolved_provider = provider_override or _guess_provider(response)

    # Try OpenAI-style fields first, then Anthropic-style.
    # Use `is not None` checks so that a valid 0 value is not treated as falsy.
    _pt = getattr(usage, "prompt_tokens", None)
    _it = getattr(usage, "input_tokens", None)
    input_tokens = _pt if _pt is not None else (_it if _it is not None else 0)

    _ct = getattr(usage, "completion_tokens", None)
    _ot = getattr(usage, "output_tokens", None)
    output_tokens = _ct if _ct is not None else (_ot if _ot is not None else 0)

    cached_input_tokens = 0
    # OpenAI cached tokens
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_input_tokens = getattr(details, "cached_tokens", 0) or 0
    # Anthropic cached tokens
    if cached_input_tokens == 0:
        cached_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    tracker.record(
        model=resolved_model,
        provider=resolved_provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _guess_provider(response: Any) -> str:
    """Best-effort provider detection from the response class path."""
    cls_module = type(response).__module__ or ""
    if "openai" in cls_module:
        return "openai"
    if "anthropic" in cls_module:
        return "anthropic"
    return "unknown"
