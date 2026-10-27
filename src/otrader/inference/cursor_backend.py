"""Cursor SDK backend (Pro plan). Runs a LOCAL Cursor agent per episode with the built-in coding
toolset restricted to custom MCP tools and our perception tools injected as custom_tools.

Notes (from cursor.com/docs/sdk/python, Sept 2026):
- SDK runs bill under the same pools as the IDE; `agent.get_usage()` returns charged cost.
- custom_tools and tool allowlists are local-agent only.
- Model ids and params (effort) are account-specific: discover with `Cursor.models.list()`.
- Pin a fixed model id; do NOT use auto/auto-smart for a reproducible experiment.
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import asdict
from .base import Backend, EpisodeResult, Usage, ToolFn, estimate_cost_inr


class CursorBackend(Backend):
    name = "cursor"

    def __init__(self, workdir: str | None = None, effort: str | None = None,
                 included_only: bool = True, on_demand_disabled_confirmed: bool = False):
        from cursor_sdk import CursorClient  # noqa: F401  (import check)
        self.api_key = os.environ.get("CURSOR_API_KEY", "")
        if not self.api_key or self.api_key in {"crsr_xxx", "crsr_..."}:
            raise ValueError("Set a real CURSOR_API_KEY locally; the placeholder cannot run Cursor.")
        self.included_only = included_only
        if included_only and not on_demand_disabled_confirmed:
            raise ValueError("Disable on-demand billing in Cursor Spending, then set "
                             "inference.on_demand_disabled_confirmed=true. The SDK has no preflight billing cap.")
        self._workspace = tempfile.TemporaryDirectory(prefix="otrader-agent-")
        self.workdir = self._workspace.name  # no repo rules, mappings, credentials or real ticker context
        self._store = tempfile.TemporaryDirectory(prefix="otrader-sdk-store-")
        self._catalog = None
        self.effort = effort

    def list_models(self) -> list[dict]:
        from cursor_sdk import Cursor
        if self._catalog is None:
            self._catalog = {m.id: m for m in Cursor.models.list(api_key=self.api_key)}
        return [{"id": m.id, "parameters": [asdict(p) for p in m.parameters]}
                for m in self._catalog.values()]

    def validate_models(self, ids):
        for model_id in ids:
            self._model_selection(model_id)

    def _model_selection(self, model_id: str):
        from cursor_sdk import ModelSelection, ModelParameterValue
        if model_id in {"auto", "auto-smart"}:
            raise ValueError("Pin a fixed model for reproducible runs.")
        if self.included_only and model_id not in {"composer-2.5", "grok-4.5", "grok-4.6"}:
            raise ValueError(f"Model {model_id!r} is outside the configured Cursor Models allowlist.")
        self.list_models()
        if model_id not in self._catalog:
            raise ValueError(f"Configured model {model_id!r} is unavailable to this Cursor key.")
        definitions = {p.id: p for p in self._catalog[model_id].parameters}
        params = []
        if "fast" in definitions:
            params.append(ModelParameterValue(id="fast", value="false"))
        if self.effort:
            key = next((k for k in ("effort", "reasoning_effort") if k in definitions), None)
            if key is None:
                raise ValueError(f"{model_id} exposes no known effort parameter; leave daily_effort null.")
            allowed = {v.value for v in definitions[key].values}
            if allowed and self.effort not in allowed:
                raise ValueError(f"Unsupported effort {self.effort!r} for {model_id}; available: {sorted(allowed)}")
            params.append(ModelParameterValue(id=key, value=self.effort))
        return ModelSelection(id=model_id, params=params)

    def run_episode(self, system, user, tools: dict[str, ToolFn], tool_specs, model_id, max_tool_turns=8):
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, LocalAgentStoreConfig, CustomTool
        custom = {
            name: CustomTool(description=spec["description"], input_schema=spec["input_schema"],
                             execute=(lambda args, ctx, _fn=fn: _fn(dict(args))))
            for name, (fn, spec) in ((n, (tools[n], tool_specs[n])) for n in tools)
        }
        prompt = (f"{system}\n\nTool budget: at most {max_tool_turns} total tool calls. "
                  f"Use the returned evidence and finish within this budget.\n\n---\n\n{user}")
        trace: list[dict] = []
        with Agent.create(AgentOptions(
            model=self._model_selection(model_id),
            api_key=self.api_key,
            tools=["mcp"] if custom else [],             # custom tools use the MCP capability
            mcp_servers={},                              # no external MCP servers
            local=LocalAgentOptions(cwd=self.workdir, setting_sources=[], custom_tools=custom,
                                    store=LocalAgentStoreConfig(type="jsonl", root_dir=self._store.name)),
        )) as agent:
            run = agent.send(prompt)
            text_parts: list[str] = []
            for m in run.messages():
                if m.type == "tool_call" and m.status == "running":
                    payload = m.args if isinstance(m.args, dict) else {}
                    trace.append({"tool": payload.get("toolName", m.name),
                                  "args": payload.get("args", m.args)})
                    if len(trace) > max_tool_turns:
                        run.cancel()
                        raise RuntimeError("Cursor episode exceeded the configured tool-call budget.")
                elif m.type == "assistant":
                    for b in m.message.content:
                        if getattr(b, "type", "") == "text":
                            text_parts.append(b.text)
            res = run.wait()
            if res.status != "finished":
                raise RuntimeError(f"Cursor run {res.status}: {res.result}")
            u = res.usage
            usage = Usage(u.input_tokens if u else 0, u.output_tokens if u else 0, u.cache_read_tokens if u else 0, 0.0)
            try:
                billed = agent.get_usage()
                if billed and billed.cost is not None:
                    usage.cost_inr = round(billed.cost.charged_cents / 100 * 88.0, 4)  # 0 when plan-included
                else:
                    usage.cost_inr = None  # billing has not settled; never label an estimate as charged
            except Exception:
                usage.cost_inr = None  # billing has not settled; never label an estimate as charged
            if self.included_only and usage.cost_inr is not None and usage.cost_inr > 0:
                raise RuntimeError("Cursor reported a paid charge. Run stopped; verify account billing settings.")
            final_text = res.result or "".join(text_parts)
        return EpisodeResult(text=final_text, usage=usage, tool_calls=trace, model_id=model_id, raw=res)
