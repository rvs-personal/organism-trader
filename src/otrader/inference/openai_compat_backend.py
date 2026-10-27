"""OpenAI-compatible chat-completions backend (DeepSeek, Gemini compat endpoint, OpenRouter, Groq...).
Owns the tool loop. Used as the control/fallback provider."""
from __future__ import annotations
import json
import os
import httpx
from .base import Backend, EpisodeResult, Usage, ToolFn, estimate_cost_inr


class OpenAICompatBackend(Backend):
    name = "openai_compat"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ["OPENAI_COMPAT_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["OPENAI_COMPAT_API_KEY"]
        self.client = httpx.Client(timeout=timeout)

    def _tools_payload(self, specs: dict[str, dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "description": s["description"],
                                                  "parameters": s["input_schema"]}} for n, s in specs.items()]

    def _chat(self, messages, tools, model_id) -> dict:
        body = {"model": model_id, "messages": messages}
        if tools:
            body["tools"] = tools
        r = self.client.post(f"{self.base_url}/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return r.json()

    def run_episode(self, system, user, tools: dict[str, ToolFn], tool_specs, model_id, max_tool_turns=8):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload_tools = self._tools_payload(tool_specs) if tools else None
        usage, trace = Usage(), []
        for turn in range(max_tool_turns + 1):
            resp = self._chat(messages, payload_tools if turn < max_tool_turns else None, model_id)
            u = resp.get("usage", {})
            usage.add(Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                            u.get("prompt_cache_hit_tokens", 0),
                            estimate_cost_inr(model_id, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))))
            msg = resp["choices"][0]["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return EpisodeResult(text=msg.get("content") or "", usage=usage, tool_calls=trace, model_id=model_id, raw=resp)
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = tools[name](args) if name in tools else {"error": f"unknown tool {name}"}
                trace.append({"tool": name, "args": args})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
        return EpisodeResult(text=messages[-1].get("content", "") if isinstance(messages[-1], dict) else "",
                             usage=usage, tool_calls=trace, model_id=model_id)
