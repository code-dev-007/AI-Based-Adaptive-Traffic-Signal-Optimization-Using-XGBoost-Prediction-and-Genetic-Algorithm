import { motion } from "framer-motion";
import { Clock, Flag, Gauge, Users } from "lucide-react";
import { useState } from "react";
import { api, errorMessage } from "../api/client.js";
import { LineBox } from "../components/charts.jsx";
import { ScenarioForm, scenarioStart, useScenario } from "../components/ScenarioForm.jsx";
import { SignalBox } from "../components/SignalBox.jsx";
import { Async, Button, Card, ErrorBox, Field, inputClass, KpiRow, Note, Tag, Warning } from "../components/ui.jsx";
import { useApi } from "../hooks/useApi.js";
import { useSimulationSocket } from "../hooks/useSimulationSocket.js";
import { APPROACHES } from "../utils.js";

export default function Simulation() {
  const summary = useApi(api.summary, []);
  return <Async state={summary}>{(s) => <SimulationView summary={s} />}</Async>;
}

function SimulationView({ summary }) {
  const [scenario, setScenario] = useScenario(summary);
  const [controller, setController] = useState("adaptive");
  const [hours, setHours] = useState(1);
  const [speed, setSpeed] = useState(60);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(false);
  const { state, setState, connected } = useSimulationSocket(speed);

  const run = (call, after) => async () => {
    setError(null);
    try { const result = await call(); setState(result); after?.(result); } catch (e) { setError(errorMessage(e)); }
  };
  const start = run(
    () => api.simStart({ start: scenarioStart(scenario), hours, demand_scale: scenario.demand_scale, seed: scenario.seed, controller }),
    (r) => setWarning(!r.in_test_period),
  );

  return (
    <div className="space-y-5">
      <p className="text-muted">
        A four-approach intersection with one approach green at a time. Fixed-Time shares the green equally; Adaptive uses the Genetic Algorithm
        with model-predicted demand, re-planned every simulated hour.
      </p>
      <Card title="Scenario and controls">
        <ScenarioForm summary={summary} value={scenario} onChange={setScenario} />
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          <Field label="Controller">
            <select className={inputClass} value={controller} onChange={(e) => setController(e.target.value)}>
              <option value="adaptive">Adaptive (Genetic Algorithm)</option><option value="fixed">Fixed-Time</option>
            </select>
          </Field>
          <Field label={`Hours simulated: ${hours}`}>
            <input type="range" min="1" max="3" className="mt-3 w-full accent-teal-700" value={hours} onChange={(e) => setHours(Number(e.target.value))} />
          </Field>
          <Field label={`Simulated seconds per refresh: ${speed}`}>
            <input type="range" min="10" max="300" step="10" className="mt-3 w-full accent-teal-700" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          </Field>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button variant="primary" onClick={start}>Start</Button>
          <Button onClick={run(api.simStop)}>Stop</Button>
          <Button onClick={run(api.simReset)}>Reset</Button>
          <span className={`text-xs ${connected ? "text-green-700" : "text-red-700"}`}>{connected ? "Live stream connected" : "Live stream disconnected - reconnecting..."}</span>
        </div>
        {error && <div className="mt-3"><ErrorBox message={error} /></div>}
        {warning && <Warning>This start time is inside the model's training period, so predictions look better than they would on unseen data.</Warning>}
      </Card>
      <Note>
        Dataset counts are per junction, not per direction. Approach 1-4 take their arrival rate from Junction 1-4 (a modelling assumption, not real
        North/South/East/West measurements). The multiplier scales those counts so the intersection is meaningfully loaded; queues, waits and vehicles
        served are simulated.
      </Note>
      {state ? <Live state={state} /> : <p className="text-sm text-muted">Press Start to begin a simulation.</p>}
    </div>
  );
}

function Live({ state }) {
  const m = state.metrics;
  const history = state.history.map((p) => Object.fromEntries([["t", p.t], ...APPROACHES.map((name, i) => [name, p[`queue_${i}`]])]));
  const kpis = [
    { label: "Vehicles Arrived", value: m.total_arrived, sub: "Simulated arrivals", icon: Users },
    { label: "Vehicles Served", value: m.total_served, sub: "Passed the stop line", icon: Flag, tone: "success" },
    { label: "Avg Waiting Time", value: `${m.avg_wait_s.toFixed(0)} s`, sub: "Simulated, all arrived vehicles", icon: Clock, tone: "warning" },
    { label: "Max Queue", value: m.max_queue, sub: "Largest simulated queue so far", icon: Gauge, tone: "info" },
  ];
  return (
    <div className="space-y-5">
      <div>
        <Tag kind="simulated" />
        <p className="mb-1 text-xs text-muted">
          Simulated time {String(Math.floor(m.time_s / 60)).padStart(2, "0")}:{String(m.time_s % 60).padStart(2, "0")} of {state.duration_s / 60} min
          {" "}- controller: {state.controller}
        </p>
        <div className="h-2 overflow-hidden rounded-full bg-slate-200">
          <motion.div className="h-full bg-primary" animate={{ width: `${Math.min(100, (100 * m.time_s) / state.duration_s)}%` }} transition={{ duration: 0.4 }} />
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {APPROACHES.map((name, i) => (
          <SignalBox key={name} name={name} signal={state.signals[i]} greenSeconds={state.greens[i]} queue={m.queues[i]} served={m.served[i]} avgWait={m.avg_wait_by_approach_s[i]} />
        ))}
      </div>
      <KpiRow items={kpis} />
      {history.length > 0 && (
        <Card title="Queue length over time" subtitle="Simulated queue per approach">
          <LineBox data={history} xKey="t" series={APPROACHES} yLabel="Queue (vehicles)" height={280} />
        </Card>
      )}
      {state.finished && <Note>Simulation finished. Press Reset to run it again. The run summary was saved to the database.</Note>}
    </div>
  );
}
