"""Dispersive Flies Optimisation as a TVM MetaSchedule search strategy.

Sizing: 256 flies x 10 moves = 2560 cost-model predictions per round, the same as
EvolutionarySearch, which scores its population of 512 five times per round.
The flies are shared between the sketches, one group each. After every round, flies
move towards the groups whose predictions most often reach the round's top 10%; each
group keeps at least 16 flies.
At 16 trials per round, int(16 * 0.05) truncates to zero random picks, matching
EvolutionarySearch. Random positions only top up the round if DFO finds too few new
candidates.
Search effort per round is fixed; max_trials only limits the final number of returned
candidates and stops the strategy.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import tvm_ffi
from tvm.s_tir import Schedule
from tvm.s_tir.meta_schedule.arg_info import ArgInfo
from tvm.ir.utils import derived_object
from tvm.s_tir.meta_schedule.search_strategy import MeasureCandidate, PySearchStrategy
from . import encoding, swarm

N_FLIES = 256               # total over all groups
MIN_FLIES_PER_GROUP = 16
TOP_FRACTION = 0.1          # share of a round's predictions that counts as a hit
MOVES_PER_ROUND = 10        # cost-model-scored DFO steps between real measurements
DELTA = 0.05                # disturbance threshold
COMM_INTERVAL = 5           # moves between group communication (fixed, budget-free)
EPS_GREEDY = 0.05


def replay_with_postprocs(mod, trace, postprocs, seed=None, failure_stats=None):
    """Replay a decision trace and apply the TuneContext's postprocessors.

    This mirrors MetaSchedule's ThreadedTraceApply: replay the non-postproc part,
    enter postprocessing mode, then run the configured postprocessors in order.
    Invalid postprocessor candidates return None and should be skipped.
    """
    trace = getattr(trace, "trace", trace)
    def record_failure(message):
        if failure_stats is not None:
            failure_stats["count"] += 1
            if failure_stats["first_error"] is None:
                failure_stats["first_error"] = str(message)

    try:
        sch = Schedule(mod, seed=seed, debug_mask="none", error_render_level="none")
        trace.apply_to_schedule(sch, remove_postproc=True)
        sch.enter_postproc()
        for postproc in postprocs:
            if not postproc.apply(sch):
                record_failure(f"{type(postproc).__name__}.apply returned False")
                return None
        return sch
    except Exception as err:  # MetaSchedule skips candidates that fail during replay/postprocessing.
        record_failure(err)
        return None


@derived_object
class DFOSearch(PySearchStrategy):
    """Dispersive Flies Optimisation over MetaSchedule decisions, one fly group per sketch."""

    def __init__(self, n_flies=N_FLIES, moves=MOVES_PER_ROUND,
                 delta=DELTA, comm_interval=COMM_INTERVAL, eps_greedy=EPS_GREEDY,
                 min_flies=MIN_FLIES_PER_GROUP, top_fraction=TOP_FRACTION, log_path=None):
        """Store DFO settings; MetaSchedule resources are attached later."""
        if min_flies < 3 or n_flies < 3 * min_flies:
            raise ValueError(f"need min_flies >= 3 and n_flies >= 3 * min_flies, got {n_flies}, {min_flies}")
        if not 0 < top_fraction < 1:
            raise ValueError(f"top_fraction must be between 0 and 1, got {top_fraction}")
        if moves < 1:
            raise ValueError(f"moves must be positive, got {moves}")
        if comm_interval < 1:
            raise ValueError(f"comm_interval must be positive, got {comm_interval}")
        if not 0 <= delta <= 1:
            raise ValueError(f"delta must be between 0 and 1, got {delta}")
        if not 0 <= eps_greedy <= 1:
            raise ValueError(f"eps_greedy must be between 0 and 1, got {eps_greedy}")
        self.n_flies = n_flies
        self.moves = moves
        self.delta = delta
        self.comm_interval = comm_interval
        self.eps_greedy = eps_greedy
        self.min_flies = min_flies
        self.top_fraction = top_fraction
        self.log_path = None if log_path is None else str(log_path)

    def _initialize_with_tune_context(self, context):
        """Keep the context and postprocessors used by MetaSchedule."""
        self.context = context
        self.postprocs = context.space_generator.postprocs

    def _schedule_from_position(self, position, sketch_id):
        """Convert a fly to a postprocessed schedule for its sketch."""
        trace = encoding.position_to_decisions(
            position,
            self.design_spaces[sketch_id],
            self.layouts[sketch_id],
        )
        return replay_with_postprocs(
            self.context.mod,
            trace,
            self.postprocs,
            seed=int(self.rng.integers(0, 2**31 - 1)),
            failure_stats=self._round_failure_stats,
        )

    def _measure_candidate_from_position(self, position, sketch_id):
        """Build a runner candidate only after trace replay and postprocessing."""
        sch = self._schedule_from_position(position, sketch_id)
        if sch is None:
            return None
        args_info = ArgInfo.from_entry_func(sch.mod, remove_preproc=True)
        return MeasureCandidate(sch, args_info)

    def _module_hash(self, mod):
        return int(tvm_ffi.structural_hash(mod))

    def _module_is_known(self, mod):
        mod_hash = self._module_hash(mod)
        candidates = self._measured_modules.get(mod_hash, [])
        return any(tvm_ffi.structural_equal(mod, candidate) for candidate in candidates)

    def _remember_module(self, mod):
        if self._module_is_known(mod):
            return False
        self._measured_modules.setdefault(self._module_hash(mod), []).append(mod)
        return True

    def _load_measured_modules(self):
        """Seed duplicate filtering with records already in this workload's database."""
        for record in self.database.get_all_tuning_records():
            if not tvm_ffi.structural_equal(record.workload.mod, self.context.mod):
                continue
            try:
                candidate = record.as_measure_candidate()
            except Exception:
                continue
            self._remember_module(candidate.sch.mod)

    def _score_groups(self):
        """Score every fly across all sketches in one cost-model batch."""
        candidates = []
        owners = []
        for sketch_id, group in enumerate(self.groups):
            for fly_idx, position in enumerate(group.positions):
                candidate = self._measure_candidate_from_position(position, sketch_id)
                if candidate is None:
                    group.fitness[fly_idx] = np.inf
                    continue
                candidates.append(candidate)
                owners.append((sketch_id, fly_idx, position.copy()))

        if not candidates:
            return 0
        self._round_predictions += len(candidates)
        scores = np.asarray(self.cost_model.predict(self.context, candidates), dtype=float).reshape(-1)
        if scores.size != len(candidates):
            raise RuntimeError(
                f"Cost model returned {scores.size} scores for {len(candidates)} candidates"
            )

        valid_count = 0
        for candidate, owner, score in zip(candidates, owners, scores):
            sketch_id, fly_idx, position = owner
            self._round_scores.append((float(score), sketch_id))
            if np.isfinite(score):
                self.groups[sketch_id].fitness[fly_idx] = -float(score)
                self._add_round_candidate(float(score), candidate, position, sketch_id)
                valid_count += 1
            else:
                self.groups[sketch_id].fitness[fly_idx] = np.inf
        return valid_count

    def _add_round_candidate(self, score, candidate, position, sketch_id):
        """Keep the highest-scoring unique candidates that have not been measured."""
        mod = candidate.sch.mod
        if self._module_is_known(mod):
            return
        new_entry = {
            "score": score,
            "candidate": candidate,
            "position": position,
            "sketch_id": sketch_id,
            "mod": mod,
        }
        mod_hash = self._module_hash(mod)
        bucket = self._round_candidate_modules.setdefault(mod_hash, [])
        if any(tvm_ffi.structural_equal(mod, other) for other in bucket):
            return
        bucket.append(mod)
        self._round_candidate_pool.append(new_entry)

    def _step_groups(self):
        """Move each scored group, reinitializing any group with no valid scores."""
        for sketch_id, group in enumerate(self.groups):
            fitness = np.where(np.isnan(group.fitness), np.inf, group.fitness)
            if not np.isfinite(fitness).any():
                layout = self.layouts[sketch_id]
                group.initialise(np.asarray(
                    [encoding.random_position(self.rng, layout) for _ in range(group.n_flies)],
                    dtype=float,
                ))
                continue
            swarm_best = swarm.best_known_position(group, self.shared_best)
            group.step(
                swarm_best,
                repair_fn=lambda position, layout=self.layouts[sketch_id]: encoding.repair(
                    position, layout
                ),
            )

    def pre_tuning(self, max_trials, num_trials_per_iter, design_spaces, database, cost_model):
        """Prepare per-sketch fly groups, cached layouts, and duplicate tracking."""
        if max_trials < 0 or num_trials_per_iter < 1:
            raise ValueError("max_trials must be non-negative and num_trials_per_iter positive")
        if database is None:
            raise ValueError("DFO requires a database to track measured workloads")
        if cost_model is None:
            raise ValueError("DFO requires a cost model to score fly positions")
        self.max_trials = int(max_trials)
        self.num_trials_per_iter = int(num_trials_per_iter)
        self.trials_done = 0
        self.database = database
        self.cost_model = cost_model
        self.design_spaces = design_spaces
        self.layouts = [encoding.describe_space(space) for space in design_spaces]
        if len(self.layouts) != 3:
            raise ValueError(f"DFO expects 3 sketch layouts, got {len(self.layouts)}")

        def structure(layout):
            tile_structure = tuple((tile["n"], tile["extent"]) for tile in layout["tiles"])
            categorical_structure = tuple(
                (category["index"], category["candidates"])
                for category in layout["categoricals"]
            )
            return tile_structure, categorical_structure, layout["dim"]

        expected = structure(self.layouts[0])
        if any(structure(layout) != expected for layout in self.layouts[1:]):
            raise ValueError(
                "DFO communication requires matching tile (n, extent) order and "
                "categorical slots across all sketch layouts"
            )

        # TVM uses a negative rand_state as its unset sentinel. Zero is still a
        # valid seed and must remain reproducible.
        context_seed = int(self.context.rand_state)
        seed_sequence = np.random.SeedSequence(context_seed if context_seed >= 0 else None)
        rng_sequences = seed_sequence.spawn(len(self.layouts) + 1)
        self.rng = np.random.default_rng(rng_sequences[0])
        self.groups = []
        sizes = [self.n_flies // len(self.layouts)] * len(self.layouts)
        sizes[0] += self.n_flies - sum(sizes)
        for sketch_id, (layout, rng_sequence) in enumerate(zip(self.layouts, rng_sequences[1:])):
            group_rng = np.random.default_rng(rng_sequence)
            low, high = encoding.position_bounds(layout)
            initial_positions = np.asarray(
                [encoding.random_position(group_rng, layout) for _ in range(sizes[sketch_id])],
                dtype=float,
            )
            group = swarm.FlyGroup(
                sketch_id=sketch_id,
                n_flies=sizes[sketch_id],
                dim=layout["dim"],
                delta=self.delta,
                rng=group_rng,
                low=low,
                high=high,
            )
            group.initialise(initial_positions)
            self.groups.append(group)
        self.shared_best = None
        self.global_move_count = 0
        self.round_logs = []
        self.best_latency_seconds = np.inf
        self._measured_modules = {}
        self._load_measured_modules()

    def generate_measure_candidates(self):
        """Score DFO moves and return the best unique candidates for measurement."""
        if self.trials_done >= self.max_trials:
            return None

        self._round_start_time = time.perf_counter()
        target_count = min(self.num_trials_per_iter, self.max_trials - self.trials_done)
        random_count = int(target_count * self.eps_greedy)
        best_count = target_count - random_count
        self._round_failure_stats = {"count": 0, "first_error": None}
        self._round_candidate_pool = []
        self._round_candidate_modules = {}
        self._round_predictions = 0
        self._round_scores = []
        flies_by_group = [group.n_flies for group in self.groups]
        self._score_groups()
        moves_left = self.moves - 1

        for _ in range(max(0, moves_left)):
            self._step_groups()
            self._score_groups()
            self.global_move_count += 1
            shared_best = swarm.communicate(
                self.groups, self.global_move_count, self.comm_interval
            )
            if shared_best is not None:
                self.shared_best = shared_best

        expected = self.n_flies * self.moves
        if self._round_predictions != expected:
            raise RuntimeError(
                f"Round {len(self.round_logs) + 1} made {self._round_predictions} cost-model "
                f"predictions instead of {expected}"
            )
        hit_rates = self._hit_rates()

        self._round_best_candidates = sorted(
            self._round_candidate_pool,
            key=lambda entry: entry["score"],
            reverse=True,
        )[:best_count]
        selected = []
        for entry in self._round_best_candidates:
            if self._remember_module(entry["mod"]):
                selected.append(entry["candidate"])
        swarm_chosen = len(selected)

        attempts = 0
        max_attempts = max(100, (target_count - len(selected)) * 100)
        while len(selected) < target_count and attempts < max_attempts:
            attempts += 1
            sketch_id = int(self.rng.integers(len(self.groups)))
            position = encoding.random_position(self.rng, self.layouts[sketch_id])
            candidate = self._measure_candidate_from_position(position, sketch_id)
            if candidate is None or not self._remember_module(candidate.sch.mod):
                continue
            selected.append(candidate)

        best_score = max(
            (entry["score"] for entry in self._round_best_candidates), default=None
        )
        random_filled = len(selected) - swarm_chosen
        self.round_logs.append({
            "round": len(self.round_logs) + 1,
            "trials_completed_before_round": self.trials_done,
            "trials_submitted": len(selected),
            "swarm_chosen": swarm_chosen,
            "random_filled": random_filled,
            "underfilled": max(0, target_count - len(selected)),
            "best_predicted_score": best_score,
            "diversity_by_group": [group.diversity() for group in self.groups],
            "predictions": self._round_predictions,
            "flies_by_group": flies_by_group,
            "hit_rate_by_group": hit_rates,
            "replay_failures": self._round_failure_stats["count"],
            "first_replay_error": self._round_failure_stats["first_error"],
            "candidate_generation_seconds": time.perf_counter() - self._round_start_time,
        })
        if not selected:
            raise RuntimeError(
                f"DFO selected no candidates in round {len(self.round_logs)} "
                f"with {self.trials_done}/{self.max_trials} trials done"
            )
        self._reallocate(hit_rates)
        return selected

    def _hit_rates(self):
        """Per group: share of this round's predictions that reached the round's top fraction."""
        scores = np.array([score for score, _ in self._round_scores])
        owners = np.array([sketch_id for _, sketch_id in self._round_scores])
        finite = np.isfinite(scores)
        if not finite.any():
            return [0.0] * len(self.groups)
        threshold = np.quantile(scores[finite], 1 - self.top_fraction)
        rates = []
        for sketch_id in range(len(self.groups)):
            mine = owners == sketch_id
            hits = np.count_nonzero(mine & finite & (scores >= threshold))
            rates.append(hits / max(1, np.count_nonzero(mine)))
        return rates

    def _reallocate(self, hit_rates):
        """Move flies between groups for the next round; new flies start at random positions."""
        current = [group.n_flies for group in self.groups]
        sizes = swarm.allocate(current, hit_rates, self.n_flies, self.min_flies)
        for sketch_id, (group, size) in enumerate(zip(self.groups, sizes)):
            new_positions = [
                encoding.random_position(self.rng, self.layouts[sketch_id])
                for _ in range(max(0, int(size) - group.n_flies))
            ]
            group.resize(int(size), new_positions)

    def notify_runner_results(self, measure_candidates, results):
        """Count completed trials and record measured modules.

        Fly fitness remains the negated cost-model prediction; actual timings are
        consumed by MetaSchedule's measure callbacks and cost model.
        """
        if len(measure_candidates) != len(results):
            raise ValueError(
                f"Received {len(results)} runner results for {len(measure_candidates)} candidates"
            )
        self.trials_done += len(results)
        round_best = np.inf
        for candidate, result in zip(measure_candidates, results):
            self._remember_module(candidate.sch.mod)
            if result.error_msg is None and result.run_secs:
                run_secs = np.asarray(result.run_secs, dtype=float)
                run_secs = run_secs[np.isfinite(run_secs) & (run_secs > 0)]
                if run_secs.size:
                    round_best = min(round_best, float(np.median(run_secs)))
        if round_best < np.inf:
            self.best_latency_seconds = min(self.best_latency_seconds, round_best)
        if self.round_logs:
            round_log = self.round_logs[-1]
            round_log["best_round_latency_ms"] = (
                None if round_best == np.inf else round_best * 1000
            )
            round_log["best_latency_so_far_ms"] = (
                None if self.best_latency_seconds == np.inf else self.best_latency_seconds * 1000
            )
            round_log["round_wall_time_seconds"] = time.perf_counter() - self._round_start_time

    def post_tuning(self):
        """Write per-round summaries to the configured file and MetaSchedule logs."""
        if self.round_logs:
            summaries = json.dumps(self.round_logs, sort_keys=True)
            logging.getLogger("tvm.s_tir.meta_schedule").info("DFO round summaries: %s", summaries)
            if self.log_path is not None:
                output_path = Path(self.log_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(self.round_logs, indent=2) + "\n", encoding="utf-8"
                )

    def clone(self):
        """Return a fresh strategy instance with the same search parameters."""
        return DFOSearch(
            n_flies=self.n_flies,
            moves=self.moves,
            delta=self.delta,
            comm_interval=self.comm_interval,
            eps_greedy=self.eps_greedy,
            min_flies=self.min_flies,
            top_fraction=self.top_fraction,
            log_path=self.log_path,
        )
