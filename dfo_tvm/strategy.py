"""DFO as a MetaSchedule search strategy. Skeleton only: fill in each TODO yourself.

Sizing (to match EvolutionarySearch's screening effort, ~512 x 4 = 2048 cost-model
predictions per round):
    3 groups x 64 flies x 10 moves/round = 1920 predictions/round
Output per round: TRIALS_PER_ITER (16) NEW, unmeasured candidates; ~5% random
(matches EvolutionarySearch's eps_greedy = 0.05).
Nothing here may depend on max_trials, or reading all budgets from one 512-trial
run stops being valid.

Before writing: read the base class and how EvolutionarySearch uses its inputs:
    grep -n "class PySearchStrategy" -A 120 ../tvm/python/tvm/s_tir/meta_schedule/search_strategy/search_strategy.py
"""
import numpy as np
from tvm.s_tir.meta_schedule.search_strategy import PySearchStrategy  # check exact import path
# from tvm.s_tir.meta_schedule.utils import derived_object            # check whether v0.27 needs this decorator

from . import encoding, swarm

N_FLIES_PER_GROUP = 64
MOVES_PER_ROUND = 10        # cost-model-scored DFO steps between real measurements
DELTA = 0.001               # disturbance threshold: a parameter to tune and report
COMM_INTERVAL = 5           # moves between group communication (fixed, budget-free)
EPS_GREEDY = 0.05


# @derived_object             # if required in v0.27
class DFOSearch(PySearchStrategy):
    """Dispersive Flies Optimisation over MetaSchedule decisions, one fly group per sketch."""

    def __init__(self, n_flies=N_FLIES_PER_GROUP, moves=MOVES_PER_ROUND,
                 delta=DELTA, comm_interval=COMM_INTERVAL, eps_greedy=EPS_GREEDY):
        """TODO: store parameters only. No TVM work here."""

    def _initialize_with_tune_context(self, context):
        """Called once. TODO: keep context (rand state, target, logger)."""

    def pre_tuning(self, max_trials, num_trials_per_iter, design_spaces, database, cost_model):
        """Called before the first round.

        TODO:
        - Store num_trials_per_iter, database, cost_model, design_spaces (expect 3).
        - IGNORE max_trials except for stopping (keep the budget-independence property).
        - encoding.describe_space(...) per sketch; create one FlyGroup per sketch.
        - Initialise flies with random valid positions.
        - Set up a set of already-measured candidates to avoid duplicates.
        - Set up per-round logs: best predicted, diversity per group.
        """

    def generate_measure_candidates(self):
        """Called each round. Return a list of MeasureCandidate, or None to stop.

        TODO:
        1. Stop (return None) if trials done >= max_trials.
        2. Repeat MOVES_PER_ROUND times:
             - every group: step(); convert flies to schedules; score them with
               cost_model.predict(context, candidates) in ONE batched call
             - every COMM_INTERVAL moves: swarm.communicate(...)
        3. From all scored positions this round, pick the top ~95% of 16 by prediction
           that are NOT already measured; fill the rest with random valid positions.
        4. Wrap each as MeasureCandidate(schedule, args_info). Check how
           EvolutionarySearch builds args_info.
        Time this step: if converting flies to schedules is slow in Python, note it.
        """

    def notify_runner_results(self, measure_candidates, results):
        """Called with real timings for the candidates just measured.

        TODO:
        - Count trials (failures included).
        - Mark candidates as measured.
        - Optionally put measured positions back into the matching group, using REAL
          latency as their fitness (your decision: document it).
        """

    def post_tuning(self):
        """TODO: write your per-round logs (best, diversity per group) to a file."""

    def clone(self):
        """TODO: return a fresh DFOSearch with the same parameters."""
