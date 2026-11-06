"""Pure DFO on NumPy arrays: no TVM imports.

Test this on a toy fitness (e.g. sphere function) before connecting it to TVM.
"""
from operator import index

import numpy as np


class FlyGroup:
    """One group of flies, all searching inside ONE sketch.

    State:
        positions : (n_flies, dim) float array
        fitness   : (n_flies,) float; minimised, so lower is better
    """

    def __init__(self, sketch_id, n_flies, dim, delta, rng):
        """Store settings; positions and fitness stay None until initialise()."""

        self.sketch_id = sketch_id
        if n_flies < 3:
            raise ValueError(f"n_flies must be >= 3, got {n_flies}")
        self.n_flies = n_flies
        self.dim = dim
        if not 0 <= delta <= 1:
            raise ValueError(f"delta must be between 0 and 1, got {delta}")
        self.delta = delta
        self.rng = rng
        self.positions = None
        self.fitness = None

    def initialise(self, init_positions):
        """Set positions (random valid positions, or seeded from good records) and reset every fly's fitness to unscored."""

        self.positions = np.array(init_positions, dtype=float)
        if self.positions.shape != (self.n_flies, self.dim):
            raise ValueError(f"Expected positions shape {(self.n_flies, self.dim)}, got {self.positions.shape}")
        self.fitness = np.full(self.n_flies, np.inf)

    def best_neighbour(self, idx):
        """Ring topology: return the index of the better of fly idx-1 and idx+1.

        Fitness is minimised; NaN scores lose, and ties choose the left neighbour.
        """
        if self.fitness is None:
            raise RuntimeError("FlyGroup must be initialised before selecting neighbours")
        if not 0 <= idx < self.n_flies:
            raise IndexError(f"Fly index out of range: {idx}")

        idx = index(idx)
        left = (idx - 1) % self.n_flies
        right = (idx + 1) % self.n_flies
        left_fitness = self.fitness[left]
        right_fitness = self.fitness[right]
        if np.isnan(left_fitness):
            left_fitness = np.inf
        if np.isnan(right_fitness):
            right_fitness = np.inf
        return left if left_fitness <= right_fitness else right

    def step(self, swarm_best, repair_fn):
        """One DFO update of every fly except the elite. Returns new positions.

        Per fly i, per dimension d:
            if rng.random() < delta:            # disturbance
                x[i, d] = random value in that dimension's range
            else:
                x[i, d] = x[nb, d] + u * (swarm_best[d] - x[i, d]),  u ~ U(0, 1)

        """
        if self.positions is None or self.fitness is None:
            raise RuntimeError("FlyGroup must be initialised before stepping")

        swarm_best = np.asarray(swarm_best, dtype=float)
        if swarm_best.shape != (self.dim,):
            raise ValueError(f"Expected swarm_best shape {(self.dim,)}, got {swarm_best.shape}")

        # Treat failed predictions as worst possible fitness for selection.
        fitness = np.where(np.isnan(self.fitness), np.inf, self.fitness)
        elite_idx = int(np.argmin(fitness))
        fly_indices = np.arange(self.n_flies)
        left = (fly_indices - 1) % self.n_flies
        right = (fly_indices + 1) % self.n_flies
        neighbour_indices = np.where(fitness[left] <= fitness[right], left, right)

        new_positions = self.positions.copy()
        moving = fly_indices != elite_idx
        current = self.positions[moving]
        neighbours = self.positions[neighbour_indices[moving]]
        u = self.rng.random(current.shape)
        moved = neighbours + u * (swarm_best - current)

        # With no explicit bounds in FlyGroup, use the current group's per-dimension
        # range for disturbance; repair_fn maps samples back into the valid domain.
        low = np.min(self.positions, axis=0)
        high = np.max(self.positions, axis=0)
        disturbed = self.rng.random(current.shape) < self.delta
        random_positions = self.rng.uniform(low, high, size=current.shape)
        proposals = np.where(disturbed, random_positions, moved)

        repaired = np.empty_like(proposals)
        for row, proposal in enumerate(proposals):
            repaired_position = np.asarray(repair_fn(proposal.copy()), dtype=float)
            if repaired_position.shape != (self.dim,):
                raise ValueError(
                    f"repair_fn must return shape {(self.dim,)}, got {repaired_position.shape}"
                )
            repaired[row] = repaired_position
        new_positions[moving] = repaired
        self.positions = new_positions
        return self.positions.copy()

    def best(self):
        """Return a copy of the best position and its minimised fitness."""
        if self.positions is None or self.fitness is None:
            raise RuntimeError("FlyGroup must be initialised before selecting its best fly")
        fitness = np.where(np.isnan(self.fitness), np.inf, self.fitness)
        best_idx = int(np.argmin(fitness))
        return self.positions[best_idx].copy(), float(fitness[best_idx])

    def diversity(self):
        """Mean distance of flies to the group centroid -- log this every round.

        """
        if self.positions is None:
            raise RuntimeError("FlyGroup must be initialised before measuring diversity")
        centroid = np.mean(self.positions, axis=0)
        return float(np.mean(np.linalg.norm(self.positions - centroid, axis=1)))


def communicate(groups, step_count, interval):
    """At each fixed interval, return a copy of the global best position.

    Dr al-Rifaie's advice: groups search separately, then communicate.

    Returns None between communication points so callers can retain the last shared
    best. Positions transfer directly because every sketch uses the same encoding.
    """
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")
    if step_count < 0:
        raise ValueError(f"step_count must be non-negative, got {step_count}")
    if not groups:
        raise ValueError("At least one FlyGroup is required for communication")
    if step_count % interval != 0:
        return None

    best_position = None
    best_fitness = np.inf
    expected_dim = None
    for group in groups:
        position, fitness = group.best()
        if expected_dim is None:
            expected_dim = position.shape
        elif position.shape != expected_dim:
            raise ValueError("All FlyGroups must use the same position dimension")
        if fitness < best_fitness:
            best_position = position
            best_fitness = fitness
    return best_position.copy()
