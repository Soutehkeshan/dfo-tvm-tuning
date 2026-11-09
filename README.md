# dfo-tvm-tuning

Dispersive Flies Optimisation as a search strategy for Apache TVM's MetaSchedule, benchmarked against its default evolutionary search on Arm edge devices.

## Dependencies
- Apache TVM `v0.27.0.post1` (commit `7b5acb32bd271342a24130f4ef439ff6c2e52c6c`), built from source with LLVM

## Build environment
- LLVM 23.1.4 (Homebrew)
- libomp (Homebrew), required for TVM's parallel runtime on macOS
- Python 3.12.11
- macOS 27.0.1, arm64
- Smoke test: `.venv/bin/python scripts/smoke_test.py`

## DFO communication
Between communication intervals, each fly group moves toward the better of its own
current best and the last globally shared best. A group's local improvement is shared
with other groups only at the next communication interval.
