# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python CLI application (`flight-replay`) that replays ArduPilot SITL flights and compares simulated vs real telemetry. The project lives in the `flight-replay/` subdirectory; all commands must run from there.

### Development commands

All commands use `uv run` from the `flight-replay/` directory:

| Action | Command |
|---|---|
| Install deps | `uv sync --extra dev` |
| Run tests | `uv run pytest -v` |
| CLI help | `uv run flight-replay --help` |

See `flight-replay/README.md` for full CLI usage examples.

### Key gotchas

- **Working directory**: The `flight-replay/` subdirectory is the Python project root. `uv sync` and `uv run` must be executed from there, not the repo root.
- **No lint tooling**: The project does not include a linter (ruff, flake8, mypy, etc.) in its dev dependencies. Pytest is the only dev tool.
- **ArduPilot SITL binary is optional**: The codebase includes a fixture-backed stub (`SubprocessSitlRunner`) that generates synthetic telemetry. All tests and development work without a real SITL binary.
- **BIN fixture warnings**: Running `extract-log` or `extract-mission` on `tests/fixtures/anonymized_60s.BIN` emits many `bad header` warnings from pymavlink. This is expected — the fixture is a synthetic placeholder.
- **No external services**: DuckDB is embedded (no server), no database, no Docker required.
- **Data directories**: CLI commands expect `data/` subdirectories to exist. Create them with `mkdir -p data/{raw_logs,params,missions,processed,replays,reports}` inside `flight-replay/`.
