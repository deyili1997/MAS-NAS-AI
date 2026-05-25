"""OpenRouter-backed LLM client with an Anthropic-shaped API surface.

Why this exists
---------------
The MAS-NAS controller + 3 LLM baselines were originally written against the
Anthropic Python SDK and all call sites use the pattern:

    response = client.messages.create(model=..., max_tokens=..., messages=...)
    text = response.content[0].text

To switch to OpenRouter (https://openrouter.ai) without touching agent/baseline
code, this module exposes a class with the SAME shape — `client.messages.create`
that returns an object exposing `.content[0].text` — but actually routes the
call through OpenRouter via the openai-compatible chat-completions endpoint.

Usage (drop-in replacement)
---------------------------
    # before:
    import anthropic
    client = anthropic.Anthropic()

    # after:
    from utils.llm_client import OpenRouterClient
    client = OpenRouterClient()

    # call site unchanged:
    response = client.messages.create(model="anthropic/claude-haiku-4.5",
                                       max_tokens=4096,
                                       messages=[{"role": "user", "content": prompt}])
    text = response.content[0].text   # ← still works

Env vars
--------
- `OPENROUTER_API_KEY` (required): your OpenRouter API key (sk-or-v1-...).
  Falls back to `OPENAI_API_KEY` for convenience.

Model identifiers
-----------------
OpenRouter uses `<provider>/<model-name>` format. Examples:
- `anthropic/claude-haiku-4.5`
- `anthropic/claude-sonnet-4.5`
- `openai/gpt-4o-mini`
- `google/gemini-flash-1.5`
- `meta-llama/llama-3.3-70b-instruct`
- `deepseek/deepseek-chat`

Browse models + live pricing at https://openrouter.ai/models.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# openai SDK is OpenAI-compatible and works with OpenRouter via base_url.
# OpenRouter ships an OpenAI-shaped /chat/completions endpoint.
from openai import OpenAI


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class _ContentBlock:
    """Mimics one element of anthropic response.content[]."""
    text: str


@dataclass
class _AnthropicShapedResponse:
    """Wraps an OpenAI ChatCompletion response so calling code can do
    `response.content[0].text` (Anthropic-style) without modification."""
    content: list[_ContentBlock]


class _MessagesAPI:
    """Mimics client.messages.create(...) → Anthropic-shaped response."""

    def __init__(self, openai_client: OpenAI):
        self._client = openai_client

    def create(self, model: str, max_tokens: int, messages: list[dict[str, Any]],
               **extra_kwargs) -> _AnthropicShapedResponse:
        """Call OpenRouter (openai-compatible) and reshape the response.

        Args:
            model: OpenRouter model identifier, e.g. "anthropic/claude-haiku-4.5".
            max_tokens: passed straight through to the chat endpoint.
            messages: list of {"role": ..., "content": ...} dicts (same schema
                as Anthropic + OpenAI use).
            **extra_kwargs: forwarded to openai .chat.completions.create
                (e.g. temperature, top_p, stop). Most agent code does not set
                these, but baselines occasionally do (see baseline3/4).

        Anthropic → OpenAI translation:
            - Anthropic puts the system instruction in a top-level `system=`
              keyword arg. OpenAI/OpenRouter expect a system message at the
              head of `messages`. baseline4 (and any Anthropic-shape caller)
              uses the former; we extract it here and prepend it to messages,
              so the rest of the call site stays Anthropic-shaped.
        """
        system_prompt = extra_kwargs.pop("system", None)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        completion = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            **extra_kwargs,
        )
        text = completion.choices[0].message.content or ""
        return _AnthropicShapedResponse(content=[_ContentBlock(text=text)])


class OpenRouterClient:
    """Drop-in stand-in for `anthropic.Anthropic()` that hits OpenRouter.

    Only the surfaces used by mas_search / 3 agents / 3 baselines are
    implemented (`.messages.create`). Other Anthropic-specific endpoints
    (e.g. `.models`, `.batches`) are NOT mirrored — add as needed.
    """

    def __init__(self, api_key: str | None = None, base_url: str = _OPENROUTER_BASE_URL):
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OpenRouterClient: set OPENROUTER_API_KEY in the env "
                "(get one at https://openrouter.ai/keys). "
                "Loading from a project .env? `set -a; source .env; set +a` first."
            )
        self._openai = OpenAI(api_key=key, base_url=base_url)
        # Anthropic-style accessor: `client.messages.create(...)`
        self.messages = _MessagesAPI(self._openai)


# Backward-compat alias for code that imports a function-style factory.
def make_client() -> OpenRouterClient:
    return OpenRouterClient()
