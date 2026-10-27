"""Step 1 with a real Cursor key: list models, then run ONE tiny custom-tool episode and print usage/cost.
    export CURSOR_API_KEY=crsr_...; python scripts/smoke_cursor.py
"""
import sys, json, os; sys.path.insert(0, "src")
from otrader.inference.cursor_backend import CursorBackend

from otrader.config import load_experiment
cfg = load_experiment()["inference"]
be = CursorBackend(effort=cfg.get("daily_effort"), included_only=cfg.get("included_only", True),
                   on_demand_disabled_confirmed=cfg.get("on_demand_disabled_confirmed", False))
models = be.list_models()
print("MODELS VISIBLE TO THIS KEY:")
for m in models:
    print("  ", m)
model_id = cfg["daily_model"]
be.validate_models([model_id, cfg["reflection_model"]])
print("\nusing", model_id)

executed_calls = []
def ping(args):
    executed_calls.append(args)
    return {"pong": args.get("n", 0) + 1}
specs = {"ping": {"description": "Returns n+1", "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}
res = be.run_episode(
    system="You are a test agent. Call the ping tool exactly once with n=41, then reply with ONLY JSON: {\"answer\": <pong>}",
    user="Go.", tools={"ping": ping}, tool_specs=specs, model_id=model_id, max_tool_turns=2)
print("\nTEXT:", res.text)
print("TOOL CALLS:", res.tool_calls)
print("USAGE:", res.usage.as_dict())
print("\nIf answer==42 and usage is populated, the runtime path works. Note the param id for effort in MODELS above and"
      " set inference.daily_model / daily_effort in config/experiment.yaml accordingly.")

from otrader.inference.base import extract_json
assert extract_json(res.text).get("answer") == 42, "Smoke response did not contain answer=42"
assert executed_calls == [{"n": 41}], "Smoke test did not execute ping exactly once"
print("SMOKE PASS: real tool execution and output verified.")
