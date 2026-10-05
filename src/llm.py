"""LLM + embedding clients that meter every call (tokens, USD, seconds) for the Flat RAG vs GraphRAG benchmark.

Providers (pick with env vars, otherwise the first one in PROVIDER_ORDER that has an API key wins):

    LLM_PROVIDER        = openai | openrouter | gemini | anthropic    (chat)
    EMBEDDING_PROVIDER  = openai | openrouter | gemini                (Anthropic has no embedding API)
    <PROVIDER>_CHAT_MODEL / <PROVIDER>_EMBEDDING_MODEL override the default models below.

One run uses one provider for the whole benchmark — no mid-run failover, so cost/quality numbers stay comparable.
"""

from __future__ import annotations

import importlib
import os
import re
import time
from dataclasses import dataclass, fields
from typing import Any, Callable

PROVIDERS = {
    "openai": {"key": "OPENAI_API_KEY", "base_url": None,
               "chat": "gpt-4o-mini", "embed": "text-embedding-3-small"},
    "openrouter": {"key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
                   "chat": "openai/gpt-4o-mini", "embed": "openai/text-embedding-3-small"},
    "gemini": {"key": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "chat": "gemini-2.5-flash-lite", "embed": "gemini-embedding-001"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "base_url": None,
                  "chat": "claude-opus-5-5", "embed": None},
}
PROVIDER_ORDER = ["openai", "openrouter", "gemini", "anthropic"]

# USD per 1M tokens (input, output). Check each provider's pricing page before reporting real numbers.
PRICES_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    # Gemini embedding is free of charge on the Free tier (aistudio key, no card): the binding
    # constraint is the request quota, not the price, so embedding cost really is $0.00000.
    "gemini-embedding-001": (0.0, 0.0),
    "gemini-embedding-2": (0.0, 0.0),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    # Tokens counted by character heuristic instead of a real provider counter.
    # Gemini's OpenAI-compatible /embeddings returns usage=None, so without this the Flat RAG
    # indexing row would report 0 tokens and flatter GraphRAG in the comparison.
    est_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f.name) - getattr(other, f.name) for f in fields(self)))

def price(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    per_in, per_out = PRICES_PER_M.get(model.split("/")[-1], (0.0, 0.0))   # "openai/gpt-4o-mini" -> "gpt-4o-mini"
    return (input_tokens * per_in + output_tokens * per_out) / 1_000_000

# Rough chars-per-token for mixed Vietnamese legal/news text: Vietnamese syllables pack fewer
# characters per token than English, so the usual "4 chars/token" would undercount here.
CHARS_PER_TOKEN = 2.5


def estimate_tokens(texts: list[str]) -> int:
    """Fallback token count when a provider omits usage. Order of magnitude only, never a substitute
    for a real counter — every number derived from this is flagged as estimated in the report."""
    return round(sum(len(text) for text in texts) / CHARS_PER_TOKEN)

def pick_provider(env_var: str, need_embeddings: bool) -> str:
    """Explicit env choice, else the first provider (in PROVIDER_ORDER) whose API key is set."""
    usable = [p for p in PROVIDER_ORDER if not need_embeddings or PROVIDERS[p]["embed"]]
    chosen = os.getenv(env_var, "").strip().lower()
    if chosen:
        if chosen not in usable:
            raise RuntimeError(f"{env_var}={chosen} không hợp lệ; chọn một trong: {', '.join(usable)}")
        if not os.getenv(PROVIDERS[chosen]["key"]):
            raise RuntimeError(f"{env_var}={chosen} nhưng chưa có {PROVIDERS[chosen]['key']} trong .env")
        return chosen
    for provider in usable:
        if os.getenv(PROVIDERS[provider]["key"]):
            return provider
    keys = " / ".join(PROVIDERS[p]["key"] for p in usable)
    raise RuntimeError(f"Chưa có API key nào cho {'embedding' if need_embeddings else 'chat'}: cần một trong {keys}")

def _strip_fences(text: str) -> str:
    """Some providers wrap JSON in ```json fences even when asked for raw JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()

def _openai_client(provider: str):
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    return OpenAI(api_key=os.environ[cfg["key"]], base_url=cfg["base_url"])

# One text = one API call, which is what blows through Gemini's free-tier
# "100 embed_content requests/minute" cap when indexing a few hundred chunks.
# Batching keeps the call count under the cap and cuts indexing latency ~10x.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
RETRY_ATTEMPTS = 5          # provider SDK retries twice on its own; this is the outer, quota-aware layer
RETRY_MAX_SECONDS = 60.0    # a per-minute quota resets within a minute; longer than that is per-day quota


def _provider_delay(error: BaseException) -> float | None:
    """Seconds the provider asked us to wait, or None when it gave no usable hint."""
    match = re.search(r"retry (?:after|in)\s+([\d.]+)\s*s", str(error))
    return float(match.group(1)) if match else None


def _retry_delay(error: BaseException, attempt: int) -> float:
    """Prefer the provider's own hint ('retry in 38.9s'); fall back to exponential backoff."""
    hinted = _provider_delay(error)
    if hinted is not None:
        return min(hinted + 1.0, RETRY_MAX_SECONDS)
    return min(2.0 ** attempt, RETRY_MAX_SECONDS)


def _with_retry(label: str, fn: Callable[[], Any]) -> Any:
    """Run fn(); on 429 wait and retry so a per-minute quota blip does not kill a long run.

    Two quota kinds share the 429 status and need opposite treatment:
      - per-minute (embed_content 100/min): resets within seconds -> wait and retry;
      - per-day (generate_content 500/day): the provider asks for ~18h -> retrying is pointless,
        so fail fast with an actionable message instead of hanging the run.

    These print lines are unaccented ASCII on purpose: they can fire on a cp1252 Windows
    console, where an accented Vietnamese message would raise UnicodeEncodeError from inside
    the retry handler and hide the real quota error.
    """
    from openai import RateLimitError

    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except RateLimitError as error:
            hinted = _provider_delay(error)
            if hinted is not None and hinted > RETRY_MAX_SECONDS:
                print(f"[{label}] quota het theo NGAY (provider yeu cho {hinted / 3600:.1f} gio) "
                      f"- thu lai sau, hoac doi provider qua LLM_PROVIDER / EMBEDDING_PROVIDER "
                      f"(LAB_GUIDE.md muc 'Xu ly loi').", flush=True)
                raise
            if attempt == RETRY_ATTEMPTS:
                print(f"[{label}] still rate-limited after {RETRY_ATTEMPTS} retries: "
                      f"check quota/billing, or switch provider via LLM_PROVIDER / EMBEDDING_PROVIDER")
                raise
            delay = _retry_delay(error, attempt)
            print(f"[{label}] 429 rate limit - waiting {delay:.0f}s then retry "
                  f"({attempt + 1}/{RETRY_ATTEMPTS})", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")

class MeteredLLM:
    """`chat` and `embed` are drop-in `llm_fn` / `embedding_fn`; `usage` accumulates across calls."""

    def __init__(self, chat_provider: str | None = None, embed_provider: str | None = None) -> None:
        self.chat_provider = chat_provider or pick_provider("LLM_PROVIDER", need_embeddings=False)
        self.embed_provider = embed_provider or pick_provider("EMBEDDING_PROVIDER", need_embeddings=True)
        self.chat_model_id = os.getenv(f"{self.chat_provider.upper()}_CHAT_MODEL", PROVIDERS[self.chat_provider]["chat"])
        self.embed_model_id = os.getenv(f"{self.embed_provider.upper()}_EMBEDDING_MODEL",
                                        PROVIDERS[self.embed_provider]["embed"])
        self.chat_model = f"{self.chat_provider}:{self.chat_model_id}"
        self.embedding_model = f"{self.embed_provider}:{self.embed_model_id}"
        self._backend_name = self.embedding_model
        self.usage = Usage()
        self._chat_client: Any
        self._embed_client: Any
        if self.chat_provider == "anthropic":
            anthropic = importlib.import_module("anthropic")
            self._chat_client = anthropic.Anthropic(api_key=os.environ[PROVIDERS["anthropic"]["key"]])
        else:
            self._chat_client = _openai_client(self.chat_provider)
        self._embed_client = (self._chat_client if self.embed_provider == self.chat_provider
                              else _openai_client(self.embed_provider))

    def chat(self, prompt: str, json_mode: bool = False) -> str:
        start = time.perf_counter()
        if self.chat_provider == "anthropic":
            text, model, tokens_in, tokens_out = self._chat_anthropic(prompt)
        else:
            if json_mode and self.chat_provider != "gemini":
                response = _with_retry("chat", lambda: self._chat_client.chat.completions.create(
                    model=self.chat_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                ))
            else:
                response = _with_retry("chat", lambda: self._chat_client.chat.completions.create(
                    model=self.chat_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                ))
            text, model = response.choices[0].message.content or "", self.chat_model_id
            usage = response.usage
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
        self.usage += Usage(1, tokens_in, tokens_out, price(model, tokens_in, tokens_out), time.perf_counter() - start)
        return _strip_fences(text) if json_mode else text

    def _chat_anthropic(self, prompt: str) -> tuple[str, str, int, int]:
        # Claude Opus 5.5: thinking is always on and sampling params are removed; effort is the cost lever.
        # Server-side fallback re-runs a policy-declined request on another model inside the same call.
        response = _with_retry("chat", lambda: self._chat_client.beta.messages.create(
            model=self.chat_model_id,
            max_tokens=16000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            messages=[{"role": "user", "content": prompt}],
        ))
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.model, response.usage.input_tokens, response.usage.output_tokens

    def embed(self, text: str) -> list[float]:
        """Single text: still one API call, so it is metered as one call like before."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, EMBED_BATCH_SIZE per request, in the order given.

        Google bills/quota-counts one batch as a single embed_content request, so indexing
        176 chunks costs ~4 calls instead of 176 — the difference between finishing and
        hitting the free-tier 100-requests/minute cap.
        """
        if not texts:
            return []
        size = max(1, EMBED_BATCH_SIZE)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            window = texts[start:start + size]
            start_time = time.perf_counter()
            response = _with_retry(f"embed({self.embed_model_id})",
                                   lambda: self._embed_client.embeddings.create(
                                       model=self.embed_model_id, input=window))
            # Gemini's OpenAI-compatible /embeddings returns usage=None. Falling back to 0 there would
            # understate Flat RAG indexing cost and make GraphRAG look unfairly cheap, so estimate
            # instead and record how many tokens were guesses.
            tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            estimated = 0
            if not tokens:
                tokens = estimated = estimate_tokens(window)
            self.usage += Usage(1, tokens, 0, price(self.embed_model_id, tokens),
                                time.perf_counter() - start_time, estimated)
            # OpenAI returns index-ordered data; Gemini leaves `index=None` and relies on array order,
            # so fall back to the response order whenever the field is absent.
            data = list(response.data)
            order = sorted(range(len(data)),
                           key=lambda pos: (data[pos].index is None,
                                             pos if data[pos].index is None else data[pos].index))
            vectors.extend([float(value) for value in data[pos].embedding] for pos in order)
        return vectors

    __call__ = embed
