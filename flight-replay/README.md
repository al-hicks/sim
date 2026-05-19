# ArduPilot SITL Flight Replay & Validation Framework

This project replays historical flight intent in ArduPilot SITL and compares
simulated telemetry against real flight telemetry. The goal is not to improve
the simulator itself; it is to provide honest, repeatable parameter comparison.

## Features in this MVP

- Flight log extraction from DataFlash logs (`.BIN`) into Parquet
- Mission extraction into normalized `mission.json`
- Alignment with explicit mission anchors and fallback policy
- Per-phase and overall comparator metrics
- Offline Plotly HTML report generation
- Batch aggregation over many flights via DuckDB
- Structured JSON-lines run logs and run manifests for provenance
- Swappable `SitlRunner` interface with a fixture-backed stub implementation

## Repository layout

```text
flight-replay/
  src/
  data/
    raw_logs/
    params/
    missions/
    processed/
    replays/
    reports/
  tests/
```

## Setup (Linux/macOS)

### Option A: uv

```bash
cd flight-replay
uv sync
```

### Option B: pip editable install

```bash
cd flight-replay
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Getting an ArduPilot SITL binary

Python dependencies alone are not enough. You need an ArduPilot SITL binary.

- Docker image: `ardupilot/ardupilot-sitl`
  - https://hub.docker.com/r/ardupilot/ardupilot-sitl
- Source build instructions:
  - https://ardupilot.org/dev/docs/building-the-code.html
  - https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html

For MVP development you can use the included fixture-backed SITL stub while
you wire in your real SITL runtime.

## Example commands

Extract one real log:

```bash
flight-replay extract-log --input data/raw_logs/flight.BIN --output data/processed/flight.parquet
```

Extract mission:

```bash
flight-replay extract-mission --input data/raw_logs/flight.BIN --output data/missions/flight.mission.json
```

Run one SITL replay (stub backend by default):

```bash
flight-replay run-sitl --mission data/missions/flight.mission.json --params data/params/baseline.param --output data/replays/flight_replay.parquet
```

Compare one real vs SITL replay:

```bash
flight-replay compare --real data/processed/flight.parquet --sitl data/replays/flight_replay.parquet --mission data/missions/flight.mission.json --output-dir data/reports
```

Run a batch:

```bash
flight-replay batch --real-folder data/processed --sitl-folder data/replays --mission-folder data/missions --output-dir data/reports/batch
```
