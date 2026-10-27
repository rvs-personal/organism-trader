from __future__ import annotations
import os
from .base import Backend


def make_backend(name: str | None = None, **kw) -> Backend:
    name = (name or os.environ.get("OTRADER_BACKEND", "stub")).lower()
    if name == "stub":
        from .stub_backend import StubBackend
        return StubBackend(**kw)
    if name == 'ollama':
        from .ollama_backend import OllamaBackend
        return OllamaBackend(**kw)
    if name == "openai_compat":
        from .openai_compat_backend import OpenAICompatBackend
        return OpenAICompatBackend(**kw)
    if name == "cursor":
        from .cursor_backend import CursorBackend
        return CursorBackend(**kw)
    raise ValueError(f"unknown backend {name}")
