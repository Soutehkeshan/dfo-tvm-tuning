"""Run the Jetson tuning queue one run at a time, with per-run device checks.

The queue lists every run in results/jetson. Finished runs are skipped, so the
same command resumes a stopped queue.
"""
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
sys.path.insert(0, str(REPO))
import benchmark as bm  # noqa: E402
from tvm import rpc  # noqa: E402

PY = sys.executable
RAW = REPO / "logs/raw/jetson"
QUEUE_LOG = RAW / "queue_log.jsonl"
STDOUT_DIR = RAW / "queue_stdout"
TRIALS = 1024
UNTUNED_LO, UNTUNED_HI = 815 * 0.95, 840 * 1.05

queue = [("evolutionary", 0, None), ("dfo", 0, None)]
for s in range(1, 10):
    queue += [("evolutionary", s, None), ("dfo", s, None)]
queue += [("replay-trace", s, None) for s in range(5)]
queue += [("dfo", s, d) for d in (0.01, 0.05) for s in range(3)]
for s in range(10, 20):
    queue += [("evolutionary", s, None), ("dfo", s, None)]
queue += [("dfo", s, d) for d in (0.01, 0.05) for s in range(3, 10)]
queue += [("dfo", s, 0.1) for s in range(3)]


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def run_name(strategy, seed, delta):
    return f"{bm.run_label(strategy, bm.DFO_DELTA if delta is None else delta)}_n{bm.N}_t{TRIALS}_s{seed}"


def battery():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+);", out)
    return {"raw": " ".join(out.split()), "percent": int(m.group(1)) if m else None,
            "state": m.group(2).strip() if m else None, "ac": "AC Power" in out}


def clocks_ok(state):
    online = [c for c in state["cpus"].values() if c["online"] == "1"]
    return len(online) == bm.NUM_CORES and all(
        int(c[k]) == bm.MAX_KHZ for c in online for k in ("min_khz", "cur_khz", "max_khz")
    ) and "MAXN" in state["nvpmodel_q"]


def temperature(state):
    line = (state.get("tegrastats") or [""])[0]
    return dict(re.findall(r"(\w+)@([\d.]+)C", line))


def rpc_ok():
    try:
        tracker = rpc.connect_tracker(bm.RPC_CONFIG.tracker_host, bm.RPC_CONFIG.tracker_port)
        remote = tracker.request(bm.RPC_CONFIG.tracker_key, priority=1, session_timeout=60)
        exists = remote.cpu(0).exist
        del remote
        return bool(exists), None
    except Exception as error:  # noqa: BLE001
        return False, repr(error)


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
    STDOUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped, stop_reason, prev_batt = [], None, None
    log(f"queue of {len(queue)} runs")
    for index, (strategy, seed, delta) in enumerate(queue):
        name = run_name(strategy, seed, delta)
        work_dir = RAW / name
        result_path = REPO / "results/jetson" / f"{name}.json"
        entry = {"run": name, "index": index}

        if work_dir.exists() or result_path.exists():
            log(f"{name}: already exists, not overwriting")
            entry.update(status="skipped-exists", start=now(), end=now())
            skipped.append((name, "already exists"))
            with QUEUE_LOG.open("a") as f:
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

        ok, error = rpc_ok()
        if not ok:
            stop_reason = f"RPC unreachable before {name}: {error}"
            break
        try:
            state = bm.device_state()
        except Exception as error:  # noqa: BLE001
            stop_reason = f"device probe failed before {name}: {error!r}"
            break
        entry["temperature_start_c"] = temperature(state)
        if not clocks_ok(state):
            log(f"{name}: clocks or cores wrong, re-applying jetson_clocks once")
            fix = subprocess.run(["ssh", "-o", "BatchMode=yes", bm.ssh_target(), "sudo", "-n", "jetson_clocks"],
                                 capture_output=True, text=True, timeout=60)
            state = bm.device_state()
            entry["jetson_clocks_reapplied"] = {"returncode": fix.returncode, "stderr": fix.stderr.strip()}
            if fix.returncode != 0 or not clocks_ok(state):
                stop_reason = f"clocks or cores still wrong before {name}: {entry['jetson_clocks_reapplied']}"
                break

        cmd = [PY, str(REPO / "scripts/benchmark.py"), "--strategy", strategy,
               "--trials", str(TRIALS), "--seed", str(seed)]
        if delta is not None:
            cmd += ["--delta", str(delta)]
        entry["start"] = now()
        log(f"{name}: start (battery {batt['percent']}% {batt['state']}, temps {entry['temperature_start_c']})")
        t0 = time.monotonic()
        with open(STDOUT_DIR / f"{name}.log", "w") as out:
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
            tail = (STDOUT_DIR / f"{name}.log").read_text(errors="replace").splitlines()[-15:]
            entry["status"] = "failed"
            entry["error_tail"] = tail
            skipped.append((name, tail[-1] if tail else f"exit {proc.returncode}"))
        with QUEUE_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"{name}: {entry['status']} in {entry['wall_seconds']} s, untuned {entry.get('untuned_ms')}, "
            f"best@1024 {entry['db_best_1024_ms']}, build failures {entry.get('build_failures')}, "
            f"slowest build {entry.get('slowest_build_seconds')} s")

        if index < 2 and entry.get("build_failures") != 0:
            stop_reason = f"build failures in early run {name}: {entry.get('build_failures')}"
            break

    if stop_reason:
        log(f"QUEUE STOPPED: {stop_reason}")
        with QUEUE_LOG.open("a") as f:
            f.write(json.dumps({"queue_stopped": stop_reason, "time": now()}) + "\n")
    else:
        log("queue finished")
    log(f"skipped runs: {skipped if skipped else 'none'}")


if __name__ == "__main__":
    main()
