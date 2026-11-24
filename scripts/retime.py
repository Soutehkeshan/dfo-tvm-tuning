"""Re-time the best schedule of each finished run at fixed trial checkpoints, on the Jetson.

For every results/jetson/*_n512_t1024_s*.json run, take the best valid database record among
the first k trials (k = 112, 256, 512, 768, 1024) and time it again with benchmark.py's final
timing (REPEATS x NUMBER, median). The untuned baseline is timed several times at the start
and end. Output goes to results/jetson/retime/; existing files are never overwritten, so a
stopped re-timing can be restarted with the same command.
"""
import datetime as dt
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import benchmark as bm  # noqa: E402
from tvm.s_tir import meta_schedule as ms  # noqa: E402

CHECKPOINTS = (112, 256, 512, 768, 1024)
UNTUNED_START, UNTUNED_END = 3, 2
RESULTS = REPO / "results" / bm.HOST_TAG
RAW = REPO / "logs/raw" / bm.HOST_TAG
OUT = RESULTS / "retime"


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def write_new(path, data):
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(data, indent=2) + "\n")


def checkpoint_records(work_dir):
    """Return {k: (trial_index, db_ms, record_json)} for the best valid record in the first k trials."""

    best, out = None, {}
    lines = (work_dir / "database_tuning_record.json").read_text().splitlines()
    for index, line in enumerate(lines, start=1):
        record = json.loads(line)
        secs = record[1][1]
        if secs and max(secs) < 1e9:
            ms_value = statistics.mean(secs) * 1000
            if best is None or ms_value < best[1]:
                best = (index, ms_value, record)
        if index in CHECKPOINTS and best is not None:
            out[index] = best
    return out


def build_record(func, work_dir, record):
    """Compile one record exactly as benchmark.time_tuned does, via a one-record database."""

    workload_line = (work_dir / "database_workload.json").read_text().splitlines()[record[0]]
    workload = ms.database.Workload.from_json(json.loads(workload_line))
    database = ms.database.MemoryDatabase()
    database.commit_workload(workload.mod)
    database.commit_tuning_record(ms.database.TuningRecord.from_json(record[1], workload))
    sch = ms.tir_integration.compile_tir(database, func, bm.TARGET)
    if sch is None:
        raise RuntimeError("record did not produce a schedule")
    return bm.tvm.compile(sch.mod, target=bm.TARGET)


def time_untuned_block(func, label, count):
    path = OUT / f"untuned_{label}.json"
    if path.exists():
        log(f"untuned {label}: exists, skipping")
        return
    before = bm.checked_state(f"before untuned {label}")
    values = []
    for i in range(count):
        values.append(bm.time_untuned(func))
        log(f"untuned {label} {i + 1}/{count}: {values[-1]:.1f} ms")
    after = bm.checked_state(f"after untuned {label}")
    write_new(path, {"untuned_ms": values, "start_or_end": label, "time": now(),
                     "device": {"before": before, "after": after,
                                "frequency_residency": bm.frequency_residency(before, after)}})


def run_order():
    runs = []
    for path in sorted(RESULTS.glob("*_n512_t1024_s*.json")):
        name = path.stem
        runs.append((int(name.rsplit("_s", 1)[1]), name))
    return [name for _, name in sorted(runs)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    func = bm.make_matmul(bm.N)
    bm.check_design_spaces(func)
    provenance = bm.provenance()
    write_new(OUT / f"provenance_{dt.datetime.now():%Y%m%dT%H%M%S}.json",
              {"provenance": provenance, "REPEATS": bm.REPEATS, "NUMBER": bm.NUMBER,
               "DATA_SEED": bm.DATA_SEED, "CHECKPOINTS": CHECKPOINTS})

    time_untuned_block(func, "start", UNTUNED_START)
    runs = run_order()
    log(f"{len(runs)} runs to re-time")
    failed = []
    for n, name in enumerate(runs, start=1):
        path = OUT / f"{name}.json"
        if path.exists():
            log(f"{name}: exists, skipping")
            continue
        work_dir = RAW / name
        original = json.loads((RESULTS / f"{name}.json").read_text())
        before = bm.checked_state(f"before {name}")
        t0 = time.monotonic()
        retimed, cache = {}, {}
        for k, (trial, db_ms, record) in checkpoint_records(work_dir).items():
            entry = {"trial_index": trial, "db_ms": db_ms}
            if trial not in cache:
                try:
                    cache[trial] = {"retimed_ms": bm.time_module(build_record(func, work_dir, record), bm.N)}
                except Exception as error:  # noqa: BLE001
                    ok, rpc_error = bm.rpc_ok()
                    if not ok:
                        raise SystemExit(f"RPC unreachable during {name}: {rpc_error}") from error
                    cache[trial] = {"error": traceback.format_exc(limit=3)}
                    failed.append((name, k))
            entry.update(cache[trial])
            retimed[str(k)] = entry
        after = bm.checked_state(f"after {name}")
        residency = bm.frequency_residency(before, after)
        write_new(path, {
            "run": name, "time": now(), "wall_seconds": round(time.monotonic() - t0, 1),
            "checkpoints": retimed,
            "original": {"untuned_ms": original["untuned_ms"], "tuned_ms": original["tuned_ms"]},
            "device": {"before": before, "after": after, "frequency_residency": residency},
            "clocks_at_max": all(p["fraction_at_max"] == 1.0 for p in residency.values()),
        })
        summary = ", ".join(f"{k}: {v.get('retimed_ms', float('nan')):.3f}" for k, v in retimed.items())
        log(f"{n}/{len(runs)} {name}: {summary} ms ({len(cache)} schedules)")

    time_untuned_block(func, "end", UNTUNED_END)
    log(f"re-timing finished; failed schedules: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
