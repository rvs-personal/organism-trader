"""Thin CLI: otrader fetch | anonymize | run | dashboard | models"""
from __future__ import annotations
import subprocess
import sys
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def fetch():
    from .data.fetch import fetch_all
    import json; print(json.dumps(fetch_all(), indent=2))


@app.command()
def anonymize():
    from .data.anonymize import anonymize as _a
    print(_a())


@app.command()
def run(days: int = 5, backend: str = None, baselines_only: bool = False):
    from .runner import main
    argv = ["--days", str(days)] + (["--backend", backend] if backend else []) + (["--baselines-only"] if baselines_only else [])
    raise SystemExit(main(argv))


@app.command()
def dashboard():
    from pathlib import Path
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


@app.command()
def models():
    """List model ids visible to your Cursor API key (verify daily_model / effort param ids)."""
    from .inference.cursor_backend import CursorBackend
    for m in CursorBackend().list_models():
        print(m)
