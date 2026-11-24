"""Run a queue of Jetson tuning runs one at a time, with per-run device checks.

Strategies alternate within each seed, so slow drift on the device hits both alike:
    python scripts/run_queue.py --seeds 0-9 --delta 0.05
Finished runs are skipped, so the same command resumes a stopped queue.
"""
import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import benchmark as bm  # noqa: E402

PY = sys.executable
UNTUNED_LO, UNTUNED_HI = 815 * 0.95, 840 * 1.05


def parse_seeds(text):
    """Accept "0-9", "3,5,7" or a mix such as "0-4,10"."""

    seeds = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        seeds += range(int(low), int(high or low) + 1)
    return seeds


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def run_name(strategy, seed, delta, trials):
    return f"{bm.run_label(strategy, delta)}_n{bm.N}_t{trials}_s{seed}"


def battery():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+);", out)
    return {"raw": " ".join(out.split()), "percent": int(m.group(1)) if m else None,
            "state": m.group(2).strip() if m else None, "ac": "AC Power" in out}


def temperature(state):
    line = (state.get("tegrastats") or [""])[0]
    return dict(re.findall(r"(\w+)@([\d.]+)C", line))


def run_stats(work_dir):
    stats = {"build_failures": None, "run_failures": None, "slowest_build_seconds": None,
             "db_best_ms": {}}
    db = work_dir / "database_tuning_record.json"
    if db.exists():
        costs = []
        for line in open(db):
            secs = json.loads(line)[1][1]
            costs.append(statistics.mean(secs) * 1000 if secs and max(secs) < 1e9 else None)
        for k in (112, 512, 1024):
            valid = [c for c in costs[:k] if c is not None]
            if len(costs) >= k and valid:
                stats["db_best_ms"][str(k)] = min(valid)
        stats["records"] = len(costs)
    main_logs = list((work_dir / "logs").glob("*task_0_main.log"))
    if main_logs:
        text = main_logs[0].read_text(errors="replace")
        stats["build_failures"] = text.count("Error in building:")
        stats["run_failures"] = text.count("Error in running:")
    sched_logs = list((work_dir / "logs").glob("*task_scheduler.log"))
    if sched_logs:
        stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*task_scheduler\.cc:\d+\] (.*)$")
        builds, t_builder = [], None
        for line in open(sched_logs[0]):
            m = stamp.match(line.strip())
            if not m:
                continue
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if "to builder" in m.group(2):
                t_builder = t
            elif "to runner" in m.group(2) and t_builder:
                builds.append((t - t_builder).total_seconds())
                t_builder = None
        stats["slowest_build_seconds"] = max(builds) if builds else None
        stats["rounds"] = len(builds)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", choices=bm.STRATEGIES, default=list(bm.STRATEGIES))
    parser.add_argument("--seeds", type=parse_seeds, required=True, help='e.g. "0-9" or "0,2,5"')
    parser.add_argument("--delta", type=float, default=bm.DFO_DELTA, help="DFO disturbance threshold")
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--experiment", default=bm.EXPERIMENT, help="folder name for this set of runs")
    args = parser.parse_args()
    raw = REPO / "logs/raw" / args.experiment
    queue_log = raw / "queue_log.jsonl"
    stdout_dir = raw / "queue_stdout"
    queue = [(strategy, seed, args.delta if strategy == "dfo" else bm.DFO_DELTA)
             for seed in args.seeds for strategy in args.strategies]

    stdout_dir.mkdir(parents=True, exist_ok=True)
    skipped, stop_reason, prev_batt = [], None, None
    log(f"queue of {len(queue)} runs")
    for index, (strategy, seed, delta) in enumerate(queue):
        name = run_name(strategy, seed, delta, args.trials)
        work_dir = raw / name
        result_path = REPO / "results" / args.experiment / f"{name}.json"
        entry = {"run": name, "index": index}

        if work_dir.exists() or result_path.exists():
            log(f"{name}: already exists, not overwriting")
            entry.update(status="skipped-exists", start=now(), end=now())
            skipped.append((name, "already exists"))
            with queue_log.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            continue

        batt = battery()
        entry["battery_start"] = batt
        falling = batt["state"] == "discharging" or not batt["ac"] or (
            prev_batt is not None and batt["percent"] is not None and batt["percent"] < prev_batt)
        if batt["percent"] is not None and batt["percent"] < 20 and falling:
            stop_reason = f"battery {batt['percent']}% and falling before {name}: {batt['raw']}"
            break
        prev_batt = batt["percent"]

        try:
            state = bm.checked_state(f"before {name}")
        except (SystemExit, Exception) as error:  # noqa: BLE001
            stop_reason = str(error)
            break
        entry["temperature_start_c"] = temperature(state)

        cmd = [PY, str(REPO / "scripts/benchmark.py"), "--strategy", strategy,
               "--trials", str(args.trials), "--seed", str(seed), "--experiment", args.experiment]
        if strategy == "dfo":
            cmd += ["--delta", str(delta)]
        entry["start"] = now()
        log(f"{name}: start (battery {batt['percent']}% {batt['state']}, temps {entry['temperature_start_c']})")
        t0 = time.monotonic()
        with open(stdout_dir / f"{name}.log", "w") as out:
            proc = subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
        entry["end"] = now()
        entry["wall_seconds"] = round(time.monotonic() - t0, 1)
        entry["returncode"] = proc.returncode

        try:
            entry.update(run_stats(work_dir))
        except Exception:  # noqa: BLE001
            entry["stats_error"] = traceback.format_exc(limit=2)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            entry["untuned_ms"] = result["untuned_ms"]
            entry["tuned_ms"] = result["tuned_ms"]
            entry["tuning_wall_seconds"] = result["tuning_wall_time_seconds"]
            entry["untuned_drift_flag"] = not (UNTUNED_LO <= result["untuned_ms"] <= UNTUNED_HI)
        entry["db_best_1024_ms"] = entry.get("db_best_ms", {}).get("1024")
        if proc.returncode == 0:
            entry["status"] = "ok"
        else:
            tail = (stdout_dir / f"{name}.log").read_text(errors="replace").splitlines()[-15:]
            entry["status"] = "failed"
            entry["error_tail"] = tail
            skipped.append((name, tail[-1] if tail else f"exit {proc.returncode}"))
        with queue_log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"{name}: {entry['status']} in {entry['wall_seconds']} s, untuned {entry.get('untuned_ms')}, "
            f"best@1024 {entry['db_best_1024_ms']}, build failures {entry.get('build_failures')}, "
            f"slowest build {entry.get('slowest_build_seconds')} s")

        if index < 2 and entry.get("build_failures") != 0:
            stop_reason = f"build failures in early run {name}: {entry.get('build_failures')}"
            break

    if stop_reason:
        log(f"QUEUE STOPPED: {stop_reason}")
        with queue_log.open("a") as f:
            f.write(json.dumps({"queue_stopped": stop_reason, "time": now()}) + "\n")
    else:
        log("queue finished")
    log(f"skipped runs: {skipped if skipped else 'none'}")


if __name__ == "__main__":
    main()
