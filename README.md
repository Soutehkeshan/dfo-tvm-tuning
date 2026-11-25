# dfo-tvm-tuning

Dispersive Flies Optimisation (DFO) as a search strategy for Apache TVM's MetaSchedule,
compared with its default evolutionary search on a Jetson Nano.

## How DFO searches
- Each fly holds 11 whole numbers: base-2 exponents of the tile sizes of loops i, j and k,
  plus an index into the unroll limits. A repair step keeps every position a valid program.
- There is one group of flies per sketch (program structure); a fly's fitness is the
  cost model's prediction.
- 256 flies are shared between the groups (at least 16 each). After every round, flies move
  towards the groups whose predictions most often reach the round's top 10%.
- Each round makes 10 moves, or 2,560 predictions, the same as evolutionary search. The
  16 best unmeasured candidates are then timed on the device.
- Disturbance threshold δ = 0.05. Every 5 moves the best fly is shared between groups.

## Layout
- `dfo_tvm/`: the strategy (`strategy.py`), the DFO update (`swarm.py`) and the
  position encoding (`encoding.py`)
- `scripts/benchmark.py`: one tuning run, timed on the Nano over RPC
- `scripts/run_queue.py`: a queue of runs, alternating strategies within each seed
- `scripts/retime.py`: times the best schedule of every run again in one session
- `scripts/smoke_test.py`: checks that TVM compiles and runs a kernel

Runs write to `logs/raw/<experiment>/` and `results/<experiment>/`; both are ignored by git.
Data from each experiment is attached to its release.

## Running
The Nano runs a TVM RPC tracker (port 9190) and server (port 9090) on its localhost;
forward both, then start a queue and re-time it once it has finished:

```sh
export JETSON_SSH=user@host
ssh -N -L 9190:127.0.0.1:9190 -L 9090:127.0.0.1:9090 "$JETSON_SSH" &
.venv/bin/python scripts/run_queue.py --seeds 0-9
.venv/bin/python scripts/retime.py
```

`JETSON_TVM_PYTHON` sets the Nano's TVM Python if it is not `~/.local/envs/tvm-rpc/bin/python`.

## Releases
- `v1.0`: first study. Fixed groups of 64 flies per sketch, several values of δ, against
  evolutionary search and random sampling (68 runs).
- `v2.0`: second study. Shared flies, equal screening, δ = 0.05, against evolutionary
  search (20 runs).

## Dependencies
- Apache TVM `v0.27.0.post1` (commit `7b5acb32bd271342a24130f4ef439ff6c2e52c6c`), built from source with LLVM
- LLVM 23.1.4 and libomp (Homebrew), Python 3.12.11, macOS 27.0.1 (arm64)
- Jetson Nano: TVM RPC runtime at the same version, MAXN power mode with `jetson_clocks`
