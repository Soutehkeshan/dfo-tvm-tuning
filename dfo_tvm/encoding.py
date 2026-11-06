"""Fly position <-> MetaSchedule decisions. Skeleton only: fill in each TODO yourself.

Problem size (512x512 matmul, per sketch):
    i tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    j tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    k tiling : 2 exponents of 2, sum = 9, innermost <= 5  -> 1 free dim,    6 valid
    unroll   : 1 categorical index, 0..3                   -> 1 dim,         4 valid
A fly stores all 11 numbers (4 + 4 + 2 + 1) so movement is simple; repair restores the
sum constraint, so only 8 are effectively free. The sketch is NOT in the position:
each group of flies owns one sketch.
"""
import numpy as np

TOTAL_EXP = 9          # 512 = 2**9; derive from the loop extent, don't hard-code later
MAX_INNER_EXP = 5      # max_innermost_factor = 32 = 2**5; read from the trace instead
NUM_UNROLL = 4         # candidates in the SampleCategorical instruction

# Slices of the position vector. Build these from the trace, not by hand, so another
# workload size or sketch still works.
I_SLICE = slice(0, 4)
J_SLICE = slice(4, 8)
K_SLICE = slice(8, 10)
UNROLL_IDX = 10
DIM = 11


def describe_space(design_space_trace):
    """Inspect one sketch's trace and return its decision layout.

    TODO:
    - Walk trace.insts; for each SamplePerfectTile record n (parts) and
      max_innermost_factor, and the loop extent it splits.
    - For SampleCategorical record the candidate list.
    - Return something you can use to build the slices/limits above per sketch.
    - Check: all 3 sketches give the same layout (your data says yes).
    """


def repair_tile(exps, total=TOTAL_EXP, max_inner=MAX_INNER_EXP):
    """Map one loop's continuous exponents to the NEAREST valid integer tiling.

    Valid = non-negative integers, sum == total, last <= max_inner.

    TODO (your design decision; document it in the report):
    - Clip to [0, total], round.
    - If sum != total, decide where to add/remove units (proportional? largest-first?
      at random?). Aim to move the fly as little as possible.
    - Enforce the innermost cap, pushing any excess outward.
    """


def repair(position):
    """Return a valid copy of a whole fly position (all tilings + unroll).

    TODO: apply repair_tile to I/J/K slices; clip and round the unroll index.
    """


def position_to_decisions(position, design_space_trace):
    """Build the decision values MetaSchedule expects for this sketch.

    TODO:
    - Tile decision = [2**e for e in exponents] (product must equal the extent).
    - Unroll decision = the integer index.
    - Return a Trace with these decisions, e.g. via trace.with_decision(inst, value, ...).
      Check the exact API: grep -n "def with_decision" -A 20 ../tvm/python/tvm/s_tir/schedule/trace.py
    """


def decisions_to_position(trace):
    """Inverse mapping: a measured/known trace -> exponent vector.

    TODO: needed to seed flies from database records and to place the swarm best.
    """


def random_position(rng):
    """A uniformly random VALID position (used for init and disturbance).

    TODO: sample exponents, then repair. Check the distribution isn't badly skewed.
    """
