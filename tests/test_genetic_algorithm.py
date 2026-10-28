from backend import genetic_algorithm as ga
from backend.simulator import Simulation, compare_controllers


def _valid(greens):
    return (len(greens) == ga.N_APPROACHES
            and all(ga.MIN_GREEN <= g <= ga.MAX_GREEN for g in greens)
            and sum(greens) == ga.GREEN_BUDGET)


def test_repair_enforces_limits_and_budget():
    for bad in ([100, 0, 0, 0], [1, 1, 1, 1], [60, 60, 60, 60], [-5, 200, 33, 7]):
        assert _valid(ga.repair(bad))


def test_optimize_respects_timing_constraints():
    for demand in ([12, 17, 9, 20], [270, 370, 210, 450], [0, 0, 0, 0]):
        result = ga.optimize(demand, seed=3)
        assert _valid(result["greens"])
        assert result["cycle_length"] == ga.CYCLE_LENGTH


def test_optimize_never_worse_than_fixed_and_gives_busy_approach_more_green():
    result = ga.optimize([20, 20, 20, 400], seed=1)
    assert result["fitness"] <= result["fixed_fitness"]
    assert result["greens"][3] == max(result["greens"])


def test_optimize_is_deterministic_for_a_seed():
    assert ga.optimize([30, 40, 20, 50], seed=5)["greens"] == ga.optimize([30, 40, 20, 50], seed=5)["greens"]


def test_simulator_same_seed_same_arrivals_for_both_controllers():
    scenario = [[30, 40, 20, 50]]
    a, b = Simulation(scenario, controller="fixed", seed=9), Simulation(scenario, controller="adaptive", seed=9)
    assert (a.arrivals == b.arrivals).all()


def test_simulator_conserves_vehicles_and_reset_works():
    sim = Simulation([[30, 40, 20, 50]], seed=2, demand_scale=5)
    sim.advance(1800)
    m = sim.metrics()
    assert m["total_arrived"] == m["total_served"] + sum(m["queues"])
    sim.reset()
    assert sim.t == 0 and sim.metrics()["total_arrived"] == 0


def test_compare_controllers_returns_both():
    result = compare_controllers([[30, 40, 20, 50]], seed=1)
    assert set(result) == {"fixed", "adaptive"}
    assert result["fixed"]["metrics"]["total_arrived"] == result["adaptive"]["metrics"]["total_arrived"]


def test_custom_timing_settings_are_respected():
    r = ga.optimize([700, 200, 200, 100], cycle_length=90, min_green=8, max_green=40, lost_time=3, seed=2)
    assert sum(r["greens"]) == 90 - 4 * 3 and all(8 <= g <= 40 for g in r["greens"])
    assert r["constraints"]["violation_s"] == 0 and r["cycle_length"] == 90


def test_impossible_settings_are_rejected():
    for bad in (dict(cycle_length=40), dict(min_green=30, max_green=20), dict(cycle_length=300, max_green=20)):
        try:
            ga.optimize([1, 2, 3, 4], **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_constraint_violations_are_penalised():
    demand = [100, 100, 100, 100]
    valid, invalid = [26, 26, 26, 26], [5, 5, 5, 89]
    assert ga.constraint_violation(valid) == 0 and ga.constraint_violation(invalid) > 0
    assert ga.fitness(invalid, demand) > ga.fitness(valid, demand) + 1000


def test_optimized_plan_beats_equal_split_on_unbalanced_demand():
    r = ga.optimize([450, 140, 140, 70], seed=4)
    assert r["fitness"] < r["fixed_fitness"] and r["improvement_pct"] > 0
