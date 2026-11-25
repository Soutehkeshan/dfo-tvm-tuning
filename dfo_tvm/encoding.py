"""Convert DFO fly positions to MetaSchedule trace decisions.

Problem size (512x512 matmul, per sketch):
    i tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    j tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    k tiling : 2 exponents of 2, sum = 9, innermost <= 5  -> 1 free dim,    6 valid
    unroll   : 1 categorical index, 0..3                   -> 1 dim,         4 valid
A fly stores all 11 numbers (4 + 4 + 2 + 1) so movement is simple; repair restores the
sum constraint, so only 8 are effectively free. All sizes are read from the design-space
trace. The sketch is NOT in the position: each group of flies owns one sketch.
"""
import math

import numpy as np


def _sampling_instructions(trace):
    """Return supported sampling instructions in trace order."""
    tile_insts = []
    categorical_insts = []
    ordered_insts = []
    for inst in trace.insts:
        kind = inst.kind.name
        if kind == "SamplePerfectTile":
            tile_insts.append(inst)
            ordered_insts.append(inst)
        elif kind == "SampleCategorical":
            categorical_insts.append(inst)
            ordered_insts.append(inst)
        elif kind.startswith("Sample"):
            raise ValueError(f"Unsupported sampling instruction in fly encoding: {kind}")
    return tile_insts, categorical_insts, ordered_insts


def describe_space(design_space_trace):
    """Describe encoded sampling instructions in a design-space trace.

    Tile extent is inferred from the product of its recorded perfect-tile decision;
    the trace stores the sampling instruction's loop RV but not that loop's extent.
    The returned layout records tile constraints, position slices, and categorical
    candidates in trace order.
    """
    trace = getattr(design_space_trace, "trace", design_space_trace)
    tiles = []
    categoricals = []
    offset = 0
    _, _, ordered_insts = _sampling_instructions(trace)
    for inst in ordered_insts:
        if inst.kind.name == "SamplePerfectTile":
            n = int(inst.attrs[0])
            max_inner_factor = int(inst.attrs[1])
            if n < 1 or max_inner_factor < 1:
                raise ValueError("SamplePerfectTile requires positive n and max_innermost_factor")
            decision = trace.get_decision(inst)
            if decision is None:
                raise ValueError("SamplePerfectTile instruction has no recorded decision")
            factors = tuple(int(factor) for factor in decision)
            if len(factors) != n or any(factor <= 0 for factor in factors):
                raise ValueError(f"Invalid SamplePerfectTile decision: {factors}")
            extent = math.prod(factors)
            total_exp = int(round(math.log2(extent)))
            if 2**total_exp != extent:
                raise ValueError(f"Tile extent {extent} is not a power of two")
            max_inner_exp = int(math.floor(math.log2(max_inner_factor)))
            tiles.append({
                "n": n,
                "extent": extent,
                "total_exp": total_exp,
                "max_inner_factor": max_inner_factor,
                "max_inner_exp": max_inner_exp,
                "slice": slice(offset, offset + n),
                "valid_exponents": _valid_tile_exponents(n, total_exp, max_inner_exp),
            })
            offset += n
        else:
            candidates = tuple(int(candidate) for candidate in inst.attrs[0])
            if not candidates:
                raise ValueError("SampleCategorical instruction has no candidates")
            decision = trace.get_decision(inst)
            if decision is None or not 0 <= int(decision) < len(candidates):
                raise ValueError(
                    f"Invalid SampleCategorical decision {decision} for candidates {candidates}"
                )
            categoricals.append({"candidates": candidates, "index": offset})
            offset += 1

    if not tiles:
        raise ValueError("Trace contains no SamplePerfectTile instructions")
    if not categoricals:
        raise ValueError("Trace contains no SampleCategorical instructions")
    return {"tiles": tiles, "categoricals": categoricals, "dim": offset}


def position_bounds(layout):
    """Return low/high vectors for the true encoded domain, independent of fly state."""
    low = np.zeros(layout["dim"], dtype=float)
    high = np.zeros(layout["dim"], dtype=float)
    for tile in layout["tiles"]:
        high[tile["slice"]] = tile["total_exp"]
        high[tile["slice"].stop - 1] = min(tile["total_exp"], tile["max_inner_exp"])
    for categorical in layout["categoricals"]:
        high[categorical["index"]] = len(categorical["candidates"]) - 1
    return low, high


def repair_tile(exps, total, max_inner):
    """Map one loop's continuous exponents to the NEAREST valid integer tiling.

    Valid = non-negative integers, sum == total, last <= max_inner.

    Projects onto the bounded integer simplex by minimizing squared distance from
    the clipped exponents. Ties are resolved by choosing the first coordinate.
    """
    values = np.asarray(exps, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("exps must be a non-empty one-dimensional sequence")
    if total < 0 or max_inner < 0:
        raise ValueError("total and max_inner must be non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("exps must contain only finite values")

    target = np.clip(values, 0, total)
    target[-1] = min(target[-1], max_inner)
    result = np.rint(target).astype(int)
    limits = np.full(result.size, total, dtype=int)
    limits[-1] = min(total, max_inner)
    result = np.minimum(result, limits)

    while result.sum() < total:
        available = np.flatnonzero(result < limits)
        if available.size == 0:
            raise ValueError("No valid tiling satisfies the requested total and bounds")
        # Adding a unit where target-result is largest minimizes squared error.
        idx = available[np.argmax(target[available] - result[available])]
        result[idx] += 1
    while result.sum() > total:
        available = np.flatnonzero(result > 0)
        # Removing a unit where result-target is largest minimizes squared error.
        idx = available[np.argmax(result[available] - target[available])]
        result[idx] -= 1
    return result


def repair(position, layout):
    """Return a valid copy of a whole fly position (tilings plus categories)."""
    position = np.asarray(position, dtype=float)
    if position.shape != (layout["dim"],):
        raise ValueError(f"Expected position shape {(layout['dim'],)}, got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")

    repaired = position.copy()
    for tile in layout["tiles"]:
        tile_slice = tile["slice"]
        repaired[tile_slice] = repair_tile(
            repaired[tile_slice], tile["total_exp"], tile["max_inner_exp"]
        )
    for categorical in layout["categoricals"]:
        idx = categorical["index"]
        repaired[idx] = np.clip(
            np.rint(repaired[idx]), 0, len(categorical["candidates"]) - 1
        )
    return repaired


def position_to_decisions(position, design_space_trace, layout):
    """Return a trace with sample decisions replaced by the repaired position."""
    trace = getattr(design_space_trace, "trace", design_space_trace)
    tile_insts, categorical_insts, _ = _sampling_instructions(trace)
    if len(tile_insts) != len(layout["tiles"]) or len(categorical_insts) != len(
        layout["categoricals"]
    ):
        raise ValueError("Trace sampling instructions do not match the cached layout")
    position = repair(position, layout)
    updated_trace = trace
    for tile, inst in zip(layout["tiles"], tile_insts):
        exponents = position[tile["slice"]].astype(int)
        factors = [2**int(exponent) for exponent in exponents]
        updated_trace = updated_trace.with_decision(inst, factors, remove_postproc=True)
    for categorical, inst in zip(layout["categoricals"], categorical_insts):
        candidate_idx = int(position[categorical["index"]])
        updated_trace = updated_trace.with_decision(
            inst, candidate_idx, remove_postproc=True
        )
    return updated_trace


def _valid_tile_exponents(n, total, max_inner):
    """Enumerate bounded exponent compositions for uniform valid-tile sampling."""
    compositions = []

    def visit(prefix, remaining, parts_left):
        if parts_left == 1:
            if 0 <= remaining <= max_inner:
                compositions.append(prefix + [remaining])
            return
        for value in range(min(total, remaining) + 1):
            visit(prefix + [value], remaining - value, parts_left - 1)

    visit([], total, n)
    return compositions


def random_position(rng, layout):
    """Sample uniformly from valid encoded positions for the described layout."""
    position = np.zeros(layout["dim"], dtype=float)
    for tile in layout["tiles"]:
        choices = tile["valid_exponents"]
        if not choices:
            raise ValueError("No valid tile exponents for the described layout")
        position[tile["slice"]] = choices[int(rng.integers(len(choices)))]
    for categorical in layout["categoricals"]:
        position[categorical["index"]] = rng.integers(len(categorical["candidates"]))
    return position
