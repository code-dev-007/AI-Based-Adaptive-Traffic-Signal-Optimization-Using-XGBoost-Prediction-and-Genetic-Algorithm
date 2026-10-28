"""Custom Genetic Algorithm that splits one signal cycle between four approaches.

A chromosome is a list of 4 integer green times (seconds), one per approach.
Rules every chromosome must obey (enforced by `repair`):
  * each green is between MIN_GREEN and MAX_GREEN
  * the greens add up to the green budget = CYCLE_LENGTH - total lost time
Fitness (lower is better) = demand-weighted average delay per vehicle, ESTIMATED
with Webster's uniform-delay formula, plus a penalty for oversaturated approaches and a
penalty for any timing-constraint violation (a safety net: `repair` already keeps every
chromosome feasible). Nothing here is measured: the CSV has no signal timings or waiting times.

Min green, max green, cycle length and lost time are configurable per call; the module
constants below are only the defaults.
"""
import random

N_APPROACHES = 4
MIN_GREEN = 10          # seconds
MAX_GREEN = 60          # seconds
LOST_TIME = 4           # seconds per phase (yellow + all-red)
CYCLE_LENGTH = 120      # seconds
SATURATION_FLOW = 0.5   # vehicles per second of green (= 1800 veh/h)
GREEN_BUDGET = CYCLE_LENGTH - N_APPROACHES * LOST_TIME
CONSTRAINT_PENALTY = 1000.0  # seconds of estimated delay added per second of constraint violation


def green_budget(cycle_length: int = CYCLE_LENGTH, lost_time: int = LOST_TIME) -> int:
    """Seconds of green shared by all approaches = cycle minus the lost time of every phase."""
    return int(cycle_length) - N_APPROACHES * int(lost_time)


def validate_config(cycle_length=CYCLE_LENGTH, min_green=MIN_GREEN, max_green=MAX_GREEN, lost_time=LOST_TIME) -> int:
    """Check the timing settings can be satisfied. Returns the green budget or raises ValueError."""
    if min_green < 1 or lost_time < 0:
        raise ValueError("min_green must be at least 1 second and lost_time cannot be negative.")
    if min_green > max_green:
        raise ValueError(f"min_green ({min_green}) cannot be larger than max_green ({max_green}).")
    budget = green_budget(cycle_length, lost_time)
    if budget < N_APPROACHES * min_green:
        raise ValueError(f"Cycle of {cycle_length}s is too short: {N_APPROACHES} approaches need at least "
                         f"{N_APPROACHES * (min_green + lost_time)}s ({min_green}s green + {lost_time}s lost each).")
    if budget > N_APPROACHES * max_green:
        raise ValueError(f"Cycle of {cycle_length}s is too long for max_green={max_green}s: the greens could only add up to "
                         f"{N_APPROACHES * max_green}s but {budget}s of green must be allocated.")
    return budget


def constraint_violation(greens, budget=GREEN_BUDGET, lo=MIN_GREEN, hi=MAX_GREEN) -> float:
    """Total seconds by which `greens` break the rules (0 = valid): below min, above max, or wrong total."""
    if len(greens) != N_APPROACHES:
        return float("inf")
    out_of_range = sum(max(0, lo - g) + max(0, g - hi) for g in greens)
    return float(out_of_range + abs(sum(greens) - budget))


def equal_split(budget: int = GREEN_BUDGET) -> list[int]:
    """The Fixed-Time plan: the budget shared equally between approaches."""
    base, extra = divmod(budget, N_APPROACHES)
    return [base + (1 if i < extra else 0) for i in range(N_APPROACHES)]


def repair(greens, budget=GREEN_BUDGET, lo=MIN_GREEN, hi=MAX_GREEN) -> list[int]:
    """Clip to [lo, hi], then add/remove single seconds until the greens sum to the budget."""
    greens = [min(hi, max(lo, int(round(g)))) for g in greens]
    while sum(greens) != budget:
        if sum(greens) > budget:  # take a second from the approach with the most room to give
            i = max((k for k in range(len(greens)) if greens[k] > lo), key=lambda k: greens[k])
            greens[i] -= 1
        else:                     # give a second to the approach with the most room to receive
            i = min((k for k in range(len(greens)) if greens[k] < hi), key=lambda k: greens[k])
            greens[i] += 1
    return greens


def approach_delays(greens, demand_per_hour, cycle=CYCLE_LENGTH) -> list[float]:
    """Estimated delay (s per vehicle) of each approach: Webster uniform delay + oversaturation penalty."""
    delays = []
    for g, veh_per_hour in zip(greens, demand_per_hour):
        green_ratio = max(g, 1e-6) / cycle
        flow = veh_per_hour / 3600.0                       # vehicles per second
        saturation = min(1.0, flow / (SATURATION_FLOW * green_ratio))  # degree of saturation x
        delay = cycle * (1 - green_ratio) ** 2 / (2 * (1 - green_ratio * saturation))
        overflow = max(0.0, flow / (SATURATION_FLOW * green_ratio) - 1.0)  # demand above capacity
        delays.append(delay + overflow * cycle * 5)         # penalty so GA avoids oversaturation
    return delays


def fitness(greens, demand_per_hour, cycle=CYCLE_LENGTH, budget=None, lo=MIN_GREEN, hi=MAX_GREEN) -> float:
    """Estimated average delay (seconds per vehicle) for these greens, plus a penalty for violating the
    timing constraints. Lower is better."""
    if budget is None:
        budget = green_budget(cycle)
    total_demand = sum(demand_per_hour) or 1.0
    delays = approach_delays(greens, demand_per_hour, cycle)
    weighted_delay = sum(v * d for v, d in zip(demand_per_hour, delays))
    return weighted_delay / total_demand + CONSTRAINT_PENALTY * constraint_violation(greens, budget, lo, hi)


def _random_individual(rng: random.Random, budget, lo, hi) -> list[int]:
    return repair([rng.randint(lo, hi) for _ in range(N_APPROACHES)], budget, lo, hi)


def _tournament(population, scores, rng, k=3):
    contenders = rng.sample(range(len(population)), k)
    return population[min(contenders, key=lambda i: scores[i])]


def _crossover(a, b, rng, budget, lo, hi):
    """Uniform crossover: each gene is copied from either parent, then repaired."""
    return repair([x if rng.random() < 0.5 else y for x, y in zip(a, b)], budget, lo, hi)


def _mutate(greens, rng, rate, budget, lo, hi, step=8):
    """With probability `rate`, move up to `step` seconds from one approach to another."""
    greens = list(greens)
    if rng.random() < rate:
        src, dst = rng.sample(range(N_APPROACHES), 2)
        greens[src] -= rng.randint(1, step)
        greens[dst] += rng.randint(1, step)
    return repair(greens, budget, lo, hi)


def optimize(demand_per_hour, population_size=40, generations=60, mutation_rate=0.3,
             elite_count=2, seed=42, cycle_length=CYCLE_LENGTH, min_green=MIN_GREEN,
             max_green=MAX_GREEN, lost_time=LOST_TIME) -> dict:
    """Run the GA. `demand_per_hour` = 4 expected vehicle rates (veh/h), one per approach.
    Returns the best green times, the Fixed-Time (equal split) plan for comparison, and the estimated delays."""
    if len(demand_per_hour) != N_APPROACHES:
        raise ValueError(f"Expected {N_APPROACHES} demand values, got {len(demand_per_hour)}.")
    if min(demand_per_hour) < 0:
        raise ValueError("Demand values cannot be negative.")
    budget = validate_config(cycle_length, min_green, max_green, lost_time)
    lo, hi = int(min_green), int(max_green)
    fixed = equal_split(budget)
    rng = random.Random(seed)

    def score(ind):
        return fitness(ind, demand_per_hour, cycle_length, budget, lo, hi)

    population = [_random_individual(rng, budget, lo, hi) for _ in range(population_size - 1)]
    population.append(repair(fixed, budget, lo, hi))  # seed with the fixed plan, so GA is never worse than it
    history = []
    for _ in range(generations):
        scores = [score(ind) for ind in population]
        ranked = sorted(range(len(population)), key=lambda i: scores[i])
        history.append(scores[ranked[0]])
        next_gen = [population[i] for i in ranked[:elite_count]]            # elitism
        while len(next_gen) < population_size:
            child = _crossover(_tournament(population, scores, rng), _tournament(population, scores, rng), rng, budget, lo, hi)
            next_gen.append(_mutate(child, rng, mutation_rate, budget, lo, hi))
        population = next_gen
    scores = [score(ind) for ind in population]
    best = population[min(range(len(population)), key=lambda i: scores[i])]
    fixed_fit, best_fit = score(fixed), score(best)
    return {
        "greens": best,
        "cycle_length": sum(best) + N_APPROACHES * lost_time,
        "fitness": best_fit,
        "fixed_greens": fixed,
        "fixed_fitness": fixed_fit,
        "improvement_pct": 100 * (fixed_fit - best_fit) / fixed_fit if fixed_fit > 0 else 0.0,
        "estimated_delay_by_approach_s": approach_delays(best, demand_per_hour, cycle_length),
        "fixed_estimated_delay_by_approach_s": approach_delays(fixed, demand_per_hour, cycle_length),
        "constraints": {"min_green": lo, "max_green": hi, "cycle_length": int(cycle_length), "lost_time_per_phase": int(lost_time),
                        "green_budget": budget, "violation_s": constraint_violation(best, budget, lo, hi)},
        "history": history,
        "demand_per_hour": list(demand_per_hour),
    }
