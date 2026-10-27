# Model setup

Complete the offline example before you enable a model backend.

## Cursor

Install the optional adapter dependency:

```bash
python -m pip install -e '.[cursor]'
```

Set `CURSOR_API_KEY` through your local secret manager or shell environment. Do not put a key in a committed file.

Check the model names and effort settings in `config/experiment.yaml`. Use settings available to your account. The source includes a historical model choice; it is not a promise of current access.

The `included_only` setting checks available billing information after calls. It cannot prevent all charges before a call. Disable on-demand billing in your account if you do not want additional charges. Set `on_demand_disabled_confirmed` to true only after you check that account setting.

The following command makes a model call:

```bash
python scripts/smoke_cursor.py
```

After the check passes, start a separate run:

```bash
python -m otrader.runner --backend cursor --run-id model-pilot --days 1
```

## Other backends

The `openai_compat` adapter reads `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY` and `OPENAI_COMPAT_MODEL`. Use the model and service that you have chosen.

The `ollama` adapter uses a local inference service. Local model installation and performance tests are separate setup steps.

## Resume and retry limits

The runner preserves committed records. Some temporary call failures receive a limited number of retries. Authentication, billing and unknown failures can stop the run.

A process failure during an unfinished day can require new model calls. Do not assume that an interrupted call has no usage cost.

Cloud checkpoint modules are included in the source. The private notebook, tunnel settings and deployment scripts are not supplied as a ready-to-run cloud service.
