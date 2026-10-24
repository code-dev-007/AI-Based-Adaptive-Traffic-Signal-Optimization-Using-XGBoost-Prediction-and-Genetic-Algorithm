import { useState } from "react";
import { api, errorMessage } from "../api/client.js";
import { BarBox, LineBox } from "../components/charts.jsx";
import ModelComparison from "../components/ModelComparison.jsx";
import { ScenarioForm, scenarioStart, useScenario } from "../components/ScenarioForm.jsx";
import { Async, Button, Card, ErrorBox, Field, KpiRow, Loading, Note, Tag, Warning } from "../components/ui.jsx";
import { useApi } from "../hooks/useApi.js";
import { APPROACHES, totalQueue } from "../utils.js";
import { Activity, Clock, Repeat, Timer } from "lucide-react";

const METRICS = [["avg_wait_s", "Average wait (s)"], ["total_served", "Vehicles served"], ["max_queue", "Max queue (veh)"], ["end_queue", "Vehicles still queued at end"]];

export default function Analytics() {
  const summary = useApi(api.summary, []);
  return <Async state={summary}>{(s) => <AnalyticsView summary={s} />}</Async>;
}

/** Runs an API call on demand and keeps its result, loading flag and error message. */
function useAction(call, onDone) {
  const [state, setState] = useState({ result: null, loading: false, error: null });
  const run = async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try { setState({ result: await call(), loading: false, error: null }); onDone?.(); }
    catch (e) { setState({ result: null, loading: false, error: errorMessage(e) }); }
  };
  return [state, run];
}

function AnalyticsView({ summary }) {
  const [scenario, setScenario] = useScenario(summary);
  const [refresh, setRefresh] = useState(0);
  const body = { start: scenarioStart(scenario), demand_scale: scenario.demand_scale };
  return (
    <div className="space-y-5">
      <p className="text-muted">Genetic Algorithm results, a Fixed-Time vs Adaptive comparison and a prediction-model comparison. Section A is simulated; section B is measured on held-out data.</p>
      <h2 className="font-display text-xl font-semibold">A. Traffic Signal Controller Comparison</h2>
      <p className="-mt-3 text-sm text-muted">Compares traffic control strategies (Fixed-Time vs Adaptive / Genetic Algorithm) in the simulator. These results are simulated estimates, not field measurements.</p>
      <Card title="Scenario"><ScenarioForm summary={summary} value={scenario} onChange={setScenario} /></Card>
      <GeneticSection body={{ ...body, seed: scenario.seed }} onSaved={() => setRefresh((n) => n + 1)} />
      <CompareSection body={{ ...body, base_seed: scenario.seed }} onSaved={() => setRefresh((n) => n + 1)} />
      <h2 className="pt-2 font-display text-xl font-semibold">B. Machine Learning Model Comparison</h2>
      <ModelComparison />
      <SavedRuns refresh={refresh} />
    </div>
  );
}

function GeneticSection({ body, onSaved }) {
  const [{ result: r, loading, error }, run] = useAction(() => api.optimize(body), onSaved);
  const gain = r && r.fixed_fitness ? 100 * (1 - r.fitness / r.fixed_fitness) : 0;
  return (
    <Card title="Genetic Algorithm result" subtitle="Green times for one signal cycle, optimised for the model-predicted demand of the selected hour">
      <Button variant="primary" onClick={run} disabled={loading}>{loading ? "Running..." : "Run Genetic Algorithm"}</Button>
      {error && <div className="mt-3"><ErrorBox message={error} /></div>}
      {loading && <Loading />}
      {r && (
        <div className="mt-4 space-y-4">
          <Tag kind="simulated" />
          <KpiRow items={[
            { label: "GA Estimated Delay", value: `${r.fitness.toFixed(1)} s`, sub: "Per vehicle (Webster formula estimate)", icon: Timer, tone: "success" },
            { label: "Fixed-Time Estimated Delay", value: `${r.fixed_fitness.toFixed(1)} s`, sub: "Equal green split, same formula", icon: Clock, tone: "info" },
            { label: "Estimated Change", value: `${(-gain).toFixed(1)}%`, sub: "GA vs equal split, on predicted demand", icon: Activity },
            { label: "Cycle Length", value: `${r.cycle_length} s`, sub: `Greens add up to ${r.green_budget} s + lost time`, icon: Repeat, tone: "warning" },
          ]} />
          <div className="grid gap-5 lg:grid-cols-2">
            <BarBox xKey="approach" series={["Fixed-Time", "Genetic Algorithm"]} yLabel="Green time (s)"
                    data={APPROACHES.map((approach, i) => ({ approach, "Fixed-Time": r.fixed_greens[i], "Genetic Algorithm": r.greens[i] }))} />
            <LineBox xKey="generation" series={["Best fitness"]} yLabel="Seconds per vehicle"
                     data={r.history.map((fitness, i) => ({ generation: i + 1, "Best fitness": Number(fitness.toFixed(3)) }))} />
          </div>
          <p className="text-xs text-muted">Demand used (simulated vehicles/hour): {APPROACHES.map((n, i) => `${n}: ${r.demand_per_hour[i].toFixed(0)}`).join(", ")}</p>
          <Note>Fitness = demand-weighted average delay (Webster uniform delay) plus a penalty when an approach is oversaturated. Every plan respects 10-60 s per green and a fixed total cycle.</Note>
        </div>
      )}
    </Card>
  );
}

function CompareSection({ body, onSaved }) {
  const [hours, setHours] = useState(2);
  const [seeds, setSeeds] = useState(5);
  const [{ result: r, loading, error }, run] = useAction(() => api.compare({ ...body, hours, n_seeds: seeds }), onSaved);
  return (
    <Card title="Fixed-Time vs Adaptive" subtitle="Both controllers run on identical simulated arrivals for each random seed">
      <div className="grid gap-4 md:grid-cols-2">
        <Field label={`Hours simulated: ${hours}`}><input type="range" min="1" max="6" className="mt-3 w-full accent-teal-700" value={hours} onChange={(e) => setHours(Number(e.target.value))} /></Field>
        <Field label={`Number of random seeds: ${seeds}`}><input type="range" min="1" max="20" className="mt-3 w-full accent-teal-700" value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} /></Field>
      </div>
      <div className="mt-4"><Button variant="primary" onClick={run} disabled={loading}>{loading ? "Simulating both controllers..." : "Run comparison"}</Button></div>
      {error && <div className="mt-3"><ErrorBox message={error} /></div>}
      {r && <CompareResult r={r} />}
    </Card>
  );
}

function CompareResult({ r }) {
  const pm = (name, key) => `${r.summary[name][key].mean.toFixed(1)} ± ${r.summary[name][key].std.toFixed(1)}`;
  const perSeed = r.runs.fixed.map((row, i) => ({ seed: row.seed, "Fixed-Time": row.avg_wait_s, Adaptive: r.runs.adaptive[i].avg_wait_s }));
  const byApproach = APPROACHES.map((approach, i) => ({ approach, "Fixed-Time": r.example_run.fixed.avg_wait_by_approach_s[i], Adaptive: r.example_run.adaptive.avg_wait_by_approach_s[i] }));
  const queue = r.example_run.fixed.history.map((p, i) => ({ t: p.t, "Fixed-Time": totalQueue(p), Adaptive: totalQueue(r.example_run.adaptive.history[i]) }));
  return (
    <div className="mt-4 space-y-4">
      <Tag kind="simulated" />
      {!r.in_test_period && <Warning>This scenario starts inside the model's training period, so predictions are optimistic.</Warning>}
      <table className="w-full text-sm">
        <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">Metric (mean ± std)</th><th>Fixed-Time</th><th>Adaptive</th></tr></thead>
        <tbody>{METRICS.map(([key, label]) => <tr key={key} className="border-b border-line/60"><td className="py-2">{label}</td><td>{pm("fixed", key)}</td><td>{pm("adaptive", key)}</td></tr>)}</tbody>
      </table>
      {r.wait_change_pct != null && (
        <Note>
          Mean over {r.n_seeds} seed(s): Adaptive changed average waiting time by {r.wait_change_pct.toFixed(1)}% compared with Fixed-Time (demand multiplier {r.demand_scale}).
          Adaptive plans with model predictions, so its result depends on forecast accuracy; a positive change means it was worse in this scenario.
        </Note>
      )}
      <div className="grid gap-5 lg:grid-cols-2">
        <div><h3 className="mb-1 text-sm font-semibold">Average wait per seed (s)</h3><BarBox data={perSeed} xKey="seed" series={["Fixed-Time", "Adaptive"]} height={260} /></div>
        <div><h3 className="mb-1 text-sm font-semibold">Average wait per approach, first seed (s)</h3><BarBox data={byApproach} xKey="approach" series={["Fixed-Time", "Adaptive"]} height={260} /></div>
      </div>
      <div><h3 className="mb-1 text-sm font-semibold">Total simulated queue over time (first seed)</h3><LineBox data={queue} xKey="t" series={["Fixed-Time", "Adaptive"]} yLabel="Vehicles" height={260} /></div>
    </div>
  );
}

function SavedRuns({ refresh }) {
  const [cleared, setCleared] = useState(0);
  const runs = useApi(api.runs, [refresh, cleared]);
  const describe = ({ kind, result: x }) => {
    if (kind === "optimization") return `Greens ${x.greens.join("/")} s, est. delay ${x.fitness.toFixed(1)} s vs fixed ${x.fixed_fitness.toFixed(1)} s`;
    if (kind === "comparison") return `${x.n_seeds} seeds, wait change ${x.wait_change_pct == null ? "n/a" : x.wait_change_pct.toFixed(1) + "%"}`;
    return `${x.controller}, ${x.simulated_seconds}s ${x.finished ? "(finished)" : "(stopped)"}, avg wait ${x.avg_wait_s.toFixed(0)} s`;
  };
  return (
    <Card title="Saved runs" subtitle="Run summaries stored in the SQLite database (simulated results)">
      <Async state={runs}>
        {({ items }) => items.length === 0 ? <p className="text-sm text-muted">No saved runs yet.</p> : (
          <>
            <table className="w-full text-sm">
              <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">#</th><th>Type</th><th>Saved at</th><th>Scenario start</th><th>Summary</th></tr></thead>
              <tbody>{items.map((run) => (
                <tr key={run.id} className="border-b border-line/60"><td className="py-2">{run.id}</td><td className="capitalize">{run.kind}</td><td>{run.created_at.replace("T", " ")}</td><td>{run.scenario_start ?? "manual"}</td><td>{describe(run)}</td></tr>
              ))}</tbody>
            </table>
            <div className="mt-3"><Button onClick={() => api.clearRuns().then(() => setCleared((n) => n + 1))}>Clear saved runs</Button></div>
          </>
        )}
      </Async>
    </Card>
  );
}
