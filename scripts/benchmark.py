"""Tune and benchmark one matmul with TVM MetaSchedule, then save the results.

Search, XGBoost and building run on the Mac; every timing runs on the Jetson Nano over
RPC. The RPC tracker and server listen on the Nano's localhost; forward them first:
    ssh -N -L 9190:127.0.0.1:9190 -L 9090:127.0.0.1:9090 "$JETSON_SSH"

Environment:
    JETSON_SSH         SSH destination of the Nano (user@host), used for device-state checks
    JETSON_TVM_PYTHON  Python of the Nano's TVM environment (default ~/.local/envs/tvm-rpc/bin/python)
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import tvm
from tvm import te
import tvm._version
from tvm.s_tir import meta_schedule as ms

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
from dfo_tvm.encoding import describe_space  # noqa: E402
from dfo_tvm import strategy as dfo  # noqa: E402

# macOS tar adds AppleDouble ._* members that break linking the archive on the Nano.
# Set before any builder worker starts so the workers inherit it.
os.environ["COPYFILE_DISABLE"] = "1"

# ---------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------
N = 512                                                      # matrix size (N x N)
HOST = "jetson-nano"
EXPERIMENT = "v2"                                            # results/<experiment>, logs/raw/<experiment>
TARGET = tvm.target.Target(
    {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mcpu": "cortex-a57", "num-cores": 4}
)
STRATEGIES = ("evolutionary", "dfo")
STRATEGY = "evolutionary"                                    # default; override with --strategy
MAX_TRIALS = 64                                              # default trial budget; override with --trials
TRIALS_PER_ITER = 16                                         # 16 trials per round of search
SEED = 0                                                     # default tuning seed; override with --seed
DATA_SEED = 0                                                # matmul inputs for final timings, fixed across runs
DFO_DELTA = dfo.DELTA                                        # DFO disturbance threshold; override with --delta
REPEATS = 20                                                 # number of timing samples; result uses their median
NUMBER = 10                                                  # runs averaged within each timing sample

# Space: the target's default rules. This target has the arm_cpu key and asimd, so
# "from-target" resolves to the ARM NEON rules.
SPACE_SETTINGS = {"sch_rules": "from-target", "postprocs": "from-target", "mutator_probs": "from-target"}

# Tuning measurements: MetaSchedule's EvaluatorConfig defaults, identical for every strategy.
EVALUATOR_CONFIG = ms.runner.EvaluatorConfig(
    number=3, repeat=1, min_repeat_ms=100, enable_cpu_cache_flush=False
)
RPC_CONFIG = ms.runner.RPCConfig(
    tracker_host="127.0.0.1",                                # forwarded to the Nano over SSH
    tracker_port=9190,
    tracker_key="jetson-nano",
    session_priority=1,
    session_timeout_sec=60,
)
TIMING_SESSION_TIMEOUT_SEC = 1800                            # time_module sessions: untuned runs are slow
BUILDER_TIMEOUT_SEC = 120                                    # per-candidate build limit on the Mac
NANO_TVM_PYTHON = os.environ.get("JETSON_TVM_PYTHON", "~/.local/envs/tvm-rpc/bin/python")
MAX_KHZ = 1479000                                            # MAXN + jetson_clocks: 4 cores pinned here
NUM_CORES = 4


def make_matmul(n):
    """Return a TIR PrimFunc computing C = A @ B for n x n float32 matrices."""

    A = te.placeholder((n, n), dtype="float32", name="A")
    B = te.placeholder((n, n), dtype="float32", name="B")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return te.create_prim_func([A, B, C])


def make_space():
    return ms.space_generator.PostOrderApply(**SPACE_SETTINGS)


def check_design_spaces(func):
    """Print the resolved rules and every sketch's sampling instructions; stop unless the
    layout is the expected one. Returns the resolved rule lists for the result JSON."""

    context = ms.TuneContext(
        mod=tvm.IRModule({"main": func}),
        target=TARGET,
        space_generator=make_space(),
        task_name="main",
    )
    generator = context.space_generator
    resolved = {
        "sch_rules": [str(rule) for rule in generator.sch_rules],
        "postprocs": [str(postproc) for postproc in generator.postprocs],
        "mutator_probs": [
            {"mutator": str(mutator), "prob": float(prob)}
            for mutator, prob in generator.mutator_probs.items()
        ],
    }
    for kind in ("sch_rules", "postprocs"):
        print(f"{kind}:")
        for item in resolved[kind]:
            print(f"  {item}")
    print("mutator_probs:")
    for item in resolved["mutator_probs"]:
        print(f"  {item['prob']:.2f} {item['mutator']}")

    spaces = context.generate_design_space()
    print(f"design spaces: {len(spaces)}")
    problems = []
    for index, space in enumerate(spaces):
        sampling = [inst for inst in space.trace.insts if inst.kind.name.startswith("Sample")]
        print(f"  sketch {index}: {len(sampling)} sampling instructions")
        for inst in sampling:
            print(f"    {inst.kind.name} attrs={list(inst.attrs)} decision={space.trace.get_decision(inst)}")
        try:
            layout = describe_space(space)
        except ValueError as error:
            problems.append(f"sketch {index}: {error}")
            continue
        tiles = [
            (tile["n"], tile["extent"], tile["max_inner_factor"], len(tile["valid_exponents"]))
            for tile in layout["tiles"]
        ]
        categoricals = [len(cat["candidates"]) for cat in layout["categoricals"]]
        print(f"    tiles (n, extent, max_innermost, valid tilings): {tiles}; categoricals: {categoricals}")
        if tiles != [(4, N, 32, 200), (4, N, 32, 200), (2, N, 32, 6)] or categoricals != [4]:
            problems.append(f"sketch {index}: tiles {tiles}, categoricals {categoricals}")
    if len(spaces) != 3:
        problems.append(f"expected 3 sketches, got {len(spaces)}")
    if problems:
        raise SystemExit("Design space differs from the expected layout:\n  " + "\n  ".join(problems))
    print("design spaces match: 3 sketches; i and j 200 valid tilings each (innermost <= 32), "
          "k 6, one 4-way unroll")
    return resolved


def time_module(lib, n):
    """Return the median runtime in milliseconds of a compiled matmul module, timed on the Nano."""

    session = RPC_CONFIG.connect_tracker().request(
        key=RPC_CONFIG.tracker_key,
        priority=RPC_CONFIG.session_priority,
        session_timeout=TIMING_SESSION_TIMEOUT_SEC,
    )
    dev = session.cpu()
    with tempfile.TemporaryDirectory(prefix="benchmark-rpc-") as temp_dir:
        archive = Path(temp_dir) / "benchmark_module.tar"
        lib.export_library(str(archive))                     # .o files in a tar; the Nano links them
        session.upload(str(archive))
        mod = session.load_module(archive.name)

    to_dev = tvm.runtime.tensor
    rng = np.random.default_rng(DATA_SEED)
    a_np = rng.random((n, n), dtype=np.float32)
    b_np = rng.random((n, n), dtype=np.float32)
    a, b = to_dev(a_np, dev), to_dev(b_np, dev)
    c = to_dev(np.zeros((n, n), dtype="float32"), dev)

    # Run correctness once before timing; this also warms the entry point.
    mod["main"](a, b, c)
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-4, atol=1e-4)

    result = mod.time_evaluator("main", dev, repeat=REPEATS, number=NUMBER)(a, b, c)
    session.remove(archive.name)
    session.remove(archive.name + ".so")
    return result.median * 1000.0


def time_untuned(func):
    """Compile with no tuning and return its runtime (ms), the baseline for speedups."""

    lib = tvm.compile(func, target=TARGET)
    return time_module(lib, N)


def make_strategy(strategy, delta, work_dir):
    """Return the strategy for tune_tir; DFO writes its round logs into the run directory."""

    if strategy == "dfo":
        return dfo.DFOSearch(delta=delta, log_path=work_dir / "dfo_rounds.json")
    return strategy


def run_label(strategy, delta):
    if strategy == "dfo" and delta != DFO_DELTA:
        return f"dfo-d{delta:g}"
    return strategy


def tune(func, work_dir, max_trials, strategy, seed, delta):
    """Run MetaSchedule with the given search strategy and return the database."""

    return ms.tune_tir(
        mod=func,
        target=TARGET,
        work_dir=str(work_dir),
        max_trials_global=max_trials,
        num_trials_per_iter=TRIALS_PER_ITER,
        runner=ms.runner.RPCRunner(
            rpc_config=RPC_CONFIG, evaluator_config=EVALUATOR_CONFIG, max_workers=1
        ),
        seed=seed,
        space=make_space(),
        cost_model="xgb",
        strategy=make_strategy(strategy, delta, work_dir),
        builder=ms.builder.LocalBuilder(timeout_sec=BUILDER_TIMEOUT_SEC),  # cross-compiles, exports a .tar
        database="json",
    )


def time_tuned(database, func, work_dir):
    """Compile and return the runtime (ms) of the best schedule found."""

    sch = ms.tir_integration.compile_tir(database, func, TARGET)
    if sch is None:
        raise RuntimeError(f"Tuning found no valid schedule; check the logs in {work_dir}")
    lib = tvm.compile(sch.mod, target=TARGET)
    return time_module(lib, N)


# Read-only probe, run on the Nano's system Python 3.6 between (never during) timings.
_PROBE = r'''
import glob, json, os, subprocess
def read(p):
    try:
        return open(p).read().strip()
    except (IOError, OSError):
        return None
def run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=10).stdout
    except Exception as error:
        return str(error)
cpus = {}
for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")):
    f = d + "/cpufreq/"
    cpus[os.path.basename(d)] = {
        "online": read(d + "/online") or "1", "governor": read(f + "scaling_governor"),
        "min_khz": read(f + "scaling_min_freq"), "cur_khz": read(f + "scaling_cur_freq"),
        "max_khz": read(f + "scaling_max_freq")}
residency = {}
for d in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
    table = read(d + "/stats/time_in_state") or ""
    residency[os.path.basename(d)] = {l.split()[0]: int(l.split()[1]) for l in table.splitlines() if len(l.split()) == 2}
print(json.dumps({
    "cpus": cpus, "time_in_state_10ms": residency,
    "nvpmodel_q": run(["nvpmodel", "-q"]).strip(),
    "lscpu": [l for l in run(["lscpu"]).splitlines() if l.startswith(("CPU(s)", "On-line", "Off-line", "CPU max MHz", "CPU min MHz"))],
    "tegrastats": run(["timeout", "2", "tegrastats", "--interval", "1000"]).splitlines()[:1],
    "loadavg": read("/proc/loadavg"),
}))
'''


def ssh_target():
    target = os.environ.get("JETSON_SSH")
    if not target:
        raise SystemExit("Set JETSON_SSH to the Nano's SSH destination (user@host).")
    return target


def device_state():
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_target(), "python3", "-"],
        input=_PROBE, capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads([line for line in completed.stdout.splitlines() if line.startswith("{")][-1])


def clocks_ok(state):
    """True if 4 cores are online and pinned at the MAXN maximum."""

    online = [cpu for cpu in state["cpus"].values() if cpu["online"] == "1"]
    pinned = all(int(cpu[key]) == MAX_KHZ for cpu in online for key in ("min_khz", "cur_khz", "max_khz"))
    return len(online) == NUM_CORES and pinned and "MAXN" in state["nvpmodel_q"]


def check_device(state, when):
    """Stop unless the clocks are pinned."""

    if not clocks_ok(state):
        raise SystemExit(f"Device state check failed {when}: {json.dumps(state, indent=1)}")


def rpc_ok():
    """Return (True, None) if the tracker hands out a working Nano session, else (False, error)."""

    try:
        remote = RPC_CONFIG.connect_tracker().request(
            RPC_CONFIG.tracker_key, priority=1, session_timeout=60
        )
        exists = remote.cpu(0).exist
        del remote
        return bool(exists), None
    except Exception as error:  # noqa: BLE001
        return False, repr(error)


def checked_state(when):
    """Check RPC and clocks between runs; re-apply jetson_clocks once, stop if still wrong."""

    ok, error = rpc_ok()
    if not ok:
        raise SystemExit(f"RPC unreachable {when}: {error}")
    state = device_state()
    if not clocks_ok(state):
        print(f"clocks or cores wrong {when}, re-applying jetson_clocks once", flush=True)
        fix = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_target(), "sudo", "-n", "jetson_clocks"],
                             capture_output=True, text=True, timeout=60)
        state = device_state()
        if fix.returncode != 0 or not clocks_ok(state):
            raise SystemExit(f"clocks or cores still wrong {when}: {fix.returncode} {fix.stderr.strip()}")
    return state


def frequency_residency(before, after):
    """Seconds spent at each CPU frequency between two snapshots, per cpufreq policy."""

    result = {}
    for policy, start in before["time_in_state_10ms"].items():
        end = after["time_in_state_10ms"].get(policy, {})
        seconds = {freq: (end.get(freq, 0) - ticks) / 100 for freq, ticks in start.items()}
        total = sum(seconds.values())
        result[policy] = {
            "seconds_by_khz": {freq: value for freq, value in seconds.items() if value},
            "fraction_at_max": seconds.get(str(MAX_KHZ), 0.0) / total if total else None,
        }
    return result


def provenance():
    """TVM commits on both machines and the repo's commit and worktree state."""

    nano_tvm = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh_target(), NANO_TVM_PYTHON, "-c",
         "'import tvm, tvm._version as v; print(tvm.__version__, v.__commit_id__)'"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout.split()

    def git(*args, cwd=REPO_DIR):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    tvm_src = Path(tvm.__file__).resolve().parents[2]
    porcelain = git("status", "--porcelain")
    return {
        "tvm_mac": {"version": tvm.__version__, "commit_id": tvm._version.__commit_id__,
                    "source_commit": git("rev-parse", "HEAD", cwd=tvm_src)},
        "tvm_jetson": {"version": nano_tvm[0], "commit_id": nano_tvm[1]},
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(porcelain),
        "repo_status_porcelain": porcelain.splitlines(),
    }


def save_results(results, results_path, max_trials, strategy, seed, delta):
    """Write settings and measured runtimes to the tracked results directory.

    ``results`` should contain the untuned_ms, tuned_ms, and speedup values.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "settings": {
            "N": N,
            "HOST": HOST,
            "TARGET": json.loads(str(TARGET)),
            "TARGET_STR": str(TARGET),
            "STRATEGY": strategy,
            "TVM_VERSION": tvm.__version__,
            "MAX_TRIALS": max_trials,
            "TRIALS_PER_ITER": TRIALS_PER_ITER,
            "SEED": seed,
            "DATA_SEED": DATA_SEED,
            "REPEATS": REPEATS,
            "NUMBER": NUMBER,
            "SPACE": {"kind": "post-order-apply", **SPACE_SETTINGS},
            "EVALUATOR_CONFIG": EVALUATOR_CONFIG._asdict(),
            "RPC_CONFIG": RPC_CONFIG._asdict(),
            "TIMING_SESSION_TIMEOUT_SEC": TIMING_SESSION_TIMEOUT_SEC,
            "BUILDER_TIMEOUT_SEC": BUILDER_TIMEOUT_SEC,
        },
        **results,
    }
    if strategy == "dfo":
        data["settings"]["DFO"] = {
            "delta": delta,
            "n_flies": dfo.N_FLIES,
            "min_flies_per_group": dfo.MIN_FLIES_PER_GROUP,
            "top_fraction": dfo.TOP_FRACTION,
            "moves": dfo.MOVES_PER_ROUND,
            "comm_interval": dfo.COMM_INTERVAL,
            "eps_greedy": dfo.EPS_GREEDY,
        }
    with results_path.open("w", encoding="utf-8") as result_file:
        json.dump(data, result_file, indent=2)
        result_file.write("\n")
    return results_path


if __name__ == "__main__":  # required on macOS: the builder spawns processes
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES, default=STRATEGY)
    parser.add_argument("--trials", type=int, default=MAX_TRIALS, help="total trial budget")
    parser.add_argument("--seed", type=int, default=SEED, help="tuning seed")
    parser.add_argument("--delta", type=float, default=DFO_DELTA, help="DFO disturbance threshold")
    parser.add_argument("--experiment", default=EXPERIMENT, help="folder name for this set of runs")
    args = parser.parse_args()
    if args.strategy != "dfo" and args.delta != DFO_DELTA:
        raise SystemExit("--delta applies only to --strategy dfo")
    max_trials = args.trials
    run_name = f"{run_label(args.strategy, args.delta)}_n{N}_t{max_trials}_s{args.seed}"
    work_dir = REPO_DIR / "logs/raw" / args.experiment / run_name
    results_path = REPO_DIR / "results" / args.experiment / f"{run_name}.json"
    if work_dir.exists() or results_path.exists():
        raise SystemExit(
            f"Run already exists: {work_dir} or {results_path}. "
            "Delete it or change the settings before running again."
        )

    func = make_matmul(N)
    space_resolved = check_design_spaces(func)
    run_provenance = provenance()
    state_before = device_state()
    check_device(state_before, "before the run")

    untuned_ms = time_untuned(func)
    tuning_start = time.perf_counter()
    database = tune(func, work_dir, max_trials, args.strategy, args.seed, args.delta)
    tuning_wall_time_seconds = time.perf_counter() - tuning_start
    tuned_ms = time_tuned(database, func, work_dir)
    speedup = untuned_ms / tuned_ms

    state_after = device_state()
    residency = frequency_residency(state_before, state_after)
    print(f"untuned {untuned_ms:.3f} ms -> tuned {tuned_ms:.3f} ms ({speedup:.2f}x speedup)")
    results_path = save_results({
        "untuned_ms": untuned_ms,
        "tuned_ms": tuned_ms,
        "speedup": speedup,
        "tuning_wall_time_seconds": tuning_wall_time_seconds,
        "space_resolved": space_resolved,
        "provenance": run_provenance,
        "device": {
            "power_mode": "nvpmodel 0 (MAXN)",
            "clocks": f"jetson_clocks: {NUM_CORES} cores pinned at {MAX_KHZ} kHz",
            "before": state_before,
            "after": state_after,
            "frequency_residency": residency,
        },
    }, results_path, max_trials, args.strategy, args.seed, args.delta)
    print(f"Results saved to {results_path}")
    check_device(state_after, "after the run")
    if any(policy["fraction_at_max"] != 1.0 for policy in residency.values()):
        raise SystemExit(f"CPU clock left {MAX_KHZ} kHz during the run: {residency}")
