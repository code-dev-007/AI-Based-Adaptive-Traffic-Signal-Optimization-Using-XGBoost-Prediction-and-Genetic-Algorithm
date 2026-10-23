"""One-second-step simulator of a four-approach intersection (one approach is green at a time).

Everything this module produces is SIMULATED. Arrival rates come from the dataset's
junction counts (scaled by `demand_scale`); queues, waits and vehicles served are
simulator estimates, not measurements.

Both controllers see the same pre-generated random arrivals for a given seed, so
Fixed-Time vs Adaptive comparisons are fair.
"""
from collections import deque

import numpy as np

from backend import genetic_algorithm as ga

YELLOW = 3  # seconds of yellow at the start of each lost-time period (the rest is all-red)
HISTORY_EVERY = 10  # record a chart point every N simulated seconds


class Simulation:
    def __init__(self, observed, forecast=None, controller="fixed", seed=42, demand_scale=1.0):
        """observed / forecast: one entry per simulated hour, each = 4 rates in vehicles/hour.
        `observed` drives the random arrivals; `forecast` is what the Adaptive controller plans with."""
        if controller not in ("fixed", "adaptive"):
            raise ValueError("controller must be 'fixed' or 'adaptive'")
        self.controller = controller
        self.observed = [[v * demand_scale for v in hour] for hour in observed]
        self.forecast = [[v * demand_scale for v in hour] for hour in (forecast or observed)]
        self.duration = 3600 * len(self.observed)
        rates = np.repeat(np.array(self.observed) / 3600.0, 3600, axis=0)  # vehicles per second
        self.arrivals = np.random.default_rng(seed).poisson(rates)         # shape (duration, 4)
        self.reset()

    def reset(self):
        self.t = 0
        self.phase = 0              # which approach currently has the right of way
        self.phase_time = 0         # seconds since this phase started
        self.queues = [deque() for _ in range(ga.N_APPROACHES)]  # arrival times of waiting vehicles
        self.credit = [0.0] * ga.N_APPROACHES   # fractional discharge capacity
        self.arrived = [0] * ga.N_APPROACHES
        self.served = [0] * ga.N_APPROACHES
        self.wait_served = [0] * ga.N_APPROACHES  # total seconds waited by vehicles that left
        self.plans = {}             # hour index -> greens chosen by the GA (adaptive only)
        self.greens = ga.equal_split()
        self.history = []
        self.running = False
        self.saved = False          # set once the run summary has been stored

    # ---- controller ---------------------------------------------------
    def _plan_for_cycle(self):
        if self.controller == "fixed":
            return ga.equal_split()
        hour = min(self.t // 3600, len(self.forecast) - 1)
        if hour not in self.plans:  # run the GA once per simulated hour
            self.plans[hour] = ga.optimize(self.forecast[hour], population_size=30, generations=40)["greens"]
        return self.plans[hour]

    def signal_states(self):
        states = ["RED"] * ga.N_APPROACHES
        if self.phase_time < self.greens[self.phase]:
            states[self.phase] = "GREEN"
        elif self.phase_time < self.greens[self.phase] + YELLOW:
            states[self.phase] = "YELLOW"
        return states

    # ---- one simulated second -----------------------------------------
    def step(self):
        if self.t >= self.duration:
            self.running = False
            return
        if self.phase == 0 and self.phase_time == 0:
            self.greens = self._plan_for_cycle()  # plan is fixed for the whole cycle
        for i in range(ga.N_APPROACHES):
            for _ in range(int(self.arrivals[self.t, i])):
                self.queues[i].append(self.t)
            self.arrived[i] += int(self.arrivals[self.t, i])
        i = self.phase
        if self.phase_time < self.greens[i]:  # green: discharge at saturation flow
            self.credit[i] += ga.SATURATION_FLOW
            while self.credit[i] >= 1 and self.queues[i]:
                self.wait_served[i] += self.t - self.queues[i].popleft()
                self.served[i] += 1
                self.credit[i] -= 1
            if not self.queues[i]:
                self.credit[i] = 0.0
        self.t += 1
        self.phase_time += 1
        if self.phase_time >= self.greens[i] + ga.LOST_TIME:  # next phase
            self.phase = (self.phase + 1) % ga.N_APPROACHES
            self.phase_time = 0
        if self.t % HISTORY_EVERY == 0:
            self.history.append({"t": self.t, **{f"queue_{k}": len(q) for k, q in enumerate(self.queues)}})

    def advance(self, seconds):
        for _ in range(int(seconds)):
            self.step()

    # ---- results (all simulated) ----------------------------------------
    def metrics(self):
        waiting_now = [sum(self.t - a for a in q) for q in self.queues]
        total_wait = sum(self.wait_served) + sum(waiting_now)
        total_arrived = sum(self.arrived)
        return {
            "time_s": self.t,
            "queues": [len(q) for q in self.queues],
            "arrived": list(self.arrived),
            "served": list(self.served),
            "total_arrived": total_arrived,
            "total_served": sum(self.served),
            "avg_wait_s": total_wait / total_arrived if total_arrived else 0.0,  # all arrived vehicles
            "avg_wait_by_approach_s": [
                (w + n) / a if a else 0.0 for w, n, a in zip(self.wait_served, waiting_now, self.arrived)
            ],
            "max_queue": max((row[k] for row in self.history for k in row if k.startswith("queue_")), default=0),
        }

    def snapshot(self):
        return {
            "controller": self.controller,
            "running": self.running,
            "finished": self.t >= self.duration,
            "duration_s": self.duration,
            "signals": self.signal_states(),
            "greens": list(self.greens),
            "phase_time": self.phase_time,
            "metrics": self.metrics(),
            "history": self.history,
        }


def compare_controllers(observed, forecast=None, seed=42, demand_scale=1.0) -> dict:
    """Run Fixed-Time and Adaptive on the identical scenario and seed; return both final results."""
    results = {}
    for controller in ("fixed", "adaptive"):
        sim = Simulation(observed, forecast, controller, seed, demand_scale)
        sim.advance(sim.duration)
        results[controller] = sim.snapshot()
    return results
