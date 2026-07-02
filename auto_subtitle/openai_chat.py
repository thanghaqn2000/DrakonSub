"""Helpers for OpenAI Chat Completions across model families."""

from typing import Any, Optional


def supports_custom_temperature(model: str) -> bool:
    """Reasoning models (gpt-5.x, o-series) only accept the default temperature."""
    name = model.lower().split("/")[-1]
    return not (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


def create_chat_completion(
    client,
    model: str,
    messages: list,
    *,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs: Any,
):
    params = {"model": model, "messages": messages, **kwargs}
    if temperature is not None and supports_custom_temperature(model):
        params["temperature"] = temperature
    if seed is not None:
        params["seed"] = seed
    if top_p is not None:
        params["top_p"] = top_p
    return client.chat.completions.create(**params)
