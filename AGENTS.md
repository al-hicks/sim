# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python CLI application (`flight-replay`) that replays ArduPilot SITL flights and compares simulated vs real telemetry. The project lives in the `flight-replay/` subdirectory; all commands must run from there.

### Development commands

Dependencies are pre-installed into the system Python via `pip install -e ".[dev]"` (run from `flight-replay/`). No virtualenv activation or `uv run` prefix is needed.

| Action | Command | Working dir |
|---|---|---|
| Run tests | `python3 -m pytest -q` | `flight-replay/` |
| CLI help | `flight-replay --help` | anywhere |
| Reinstall deps | `pip install -e ".[dev]"` | `flight-replay/` |

See `flight-replay/README.md` for full CLI usage examples.

### Key gotchas

- **Working directory**: The `flight-replay/` subdirectory is the Python project root. `pip install` and `pytest` must be executed from there, not the repo root.
- **No lint tooling**: The project does not include a linter (ruff, flake8, mypy, etc.) in its dev dependencies. Pytest is the only dev tool.
- **ArduPilot SITL binary is optional**: The codebase includes a fixture-backed stub (`SubprocessSitlRunner`) that generates synthetic telemetry. All tests and development work without a real SITL binary.
- **BIN fixture warnings**: Running `extract-log` or `extract-mission` on `tests/fixtures/anonymized_60s.BIN` emits many `bad header` warnings from pymavlink. This is expected — the fixture is a synthetic placeholder.
- **No external services**: DuckDB is embedded (no server), no database, no Docker required.
- **Data directories**: CLI commands expect `data/` subdirectories to exist. Create them with `mkdir -p data/{raw_logs,params,missions,processed,replays,reports}` inside `flight-replay/`.
