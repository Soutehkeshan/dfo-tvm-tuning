## 2026-11-02

- **TVM build (Claude Code):** Used Claude Code to build TVM v0.27.0.post1 from source (CMake/LLVM config), diagnose a Ninja/Make mismatch, and fix library discovery via a .pth file. I checked the result by importing TVM and confirming the version and runtime library path.
- **Dependency check (Claude):** Claude suggested confirming the tvm-ffi wheel matches TVM's pinned submodule; I ran the check and both were v0.1.14.post0.
- **Smoke test (Claude):** Used Claude to write a setup smoke test (scripts/smoke_test.py) that compiles and runs a vector-add kernel via LLVM; I ran it and it passed.
- **README (Claude):** Claude suggested recording the build environment (LLVM, Python, macOS) in the README for reproducibility; I generated the details from my machine.

## 2026-11-03

- **make_matmul (Claude Code):** Claude explained the te API (placeholder, reduce_axis, compute, create_prim_func) with a vector-add example and reviewed my matmul; I fixed the lambda arguments, the assignment inside the lambda and the missing reduce axis myself.
- **time_module (Claude Code):** Claude reviewed my timing function and pointed out 1-D input arrays, the time_evaluator arguments (number vs repeat, seconds), float32 tolerance and unseeded inputs; I made the fixes, including seeding with np.random.default_rng(SEED).

## 2026-11-04

- **time_untuned (Claude Code):** Claude ran my code and found two TVM v0.27 API changes: targets must use the dict form, and time_evaluator needs Executable.jit(). It also showed from the source that tvm.build is deprecated in favour of tvm.compile. I applied the fixes.
- **tune (Claude Code):** Claude checked my tune_tir arguments against the v0.27 source and explained which settings must stay identical across methods (space, cost model, measure callbacks). It diagnosed a missing libomp dependency for XGBoost, which I installed.
- **End-to-end check (Claude Code):** Claude ran a small check of my benchmark (N=64, 16 trials) to confirm the pipeline works: about 31x speedup over untuned, and the result passed the correctness check.
- **save_results and main (Claude Code):** Claude reviewed my results-saving and main block. It caught that Target.export() is not JSON-serialisable (its own earlier suggestion, so I switched to json.loads(str(TARGET))) and that WORK_DIR overwrote results across seeds. It suggested recording tuning wall-clock time. I wrote the changes.

## 2026-11-05

- **Extra seeds (Claude Code):** At my request, Claude ran my benchmark script for seeds 1–2 of evolutionary and replay-trace (64 trials), changing only STRATEGY and SEED, then tabulated the raw numbers and committed the results files; the analysis is my own.
- **Budget check (Claude Code):** Claude helped verify, by reading TVM’s C++ source, that the trial budget only controls stopping; I used this to justify reading all budgets from single 512-trial runs.

## 2026-11-06

- **DFO package skeleton (Claude):** Claude provided a skeleton for the DFO package (module layout, function signatures, sizing to match evolutionary search’s screening budget); I wrote the implementation.
- **Cost-model warm-up (Claude):** Claude helped trace in TVM’s source that XGBoost returns random predictions for the first 100 samples; I confirmed it through the call chain and used it to explain my 64-trial results.
- **Sphere test (Claude Code):** Claude ran a throwaway shifted-sphere sanity test of my swarm.py (5 seeds): converged to ~1e-8 vs random ~0.6, best-so-far monotone, diversity didn't collapse.
- **Rastrigin test (Claude Code):** Claude ran a throwaway Rastrigin test of my swarm.py sweeping δ (10 seeds): δ=0 stalls in local minima with zero diversity; δ≈0.01–0.05 reached the global basin in 10/10.

## 2026-11-09

- **derived_object check (Claude Code):** Claude confirmed @derived_object is still required in TVM v0.27 (a plain PySearchStrategy subclass raises 'Unknown SearchStrategy') and that it now lives in tvm.ir.utils.
- **eps_greedy check (Claude):** Claude helped me trace in TVM’s C++ source that EvolutionarySearch’s eps_greedy truncates to 0 random picks at 16 trials per round; I matched DFO to it.
