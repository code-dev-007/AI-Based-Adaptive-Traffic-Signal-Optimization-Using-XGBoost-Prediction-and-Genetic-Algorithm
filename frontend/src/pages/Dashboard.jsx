import { Activity, BarChart3, Calendar, Gauge, MapPin, TrendingUp } from "lucide-react";
import { api } from "../api/client.js";
import { BarBox, LineBox } from "../components/charts.jsx";
import { Async, Card, KpiRow, Note, Tag } from "../components/ui.jsx";
import { useApi } from "../hooks/useApi.js";
import { APPROACHES, dateOnly, pivot, WEEKDAYS } from "../utils.js";

const DEMAND_SCALE = 3; // simulation assumption, shown on screen next to the results that use it

export default function Dashboard() {
  const state = useApi(api.summary, []);
  return <Async state={state}>{(s) => <DashboardView s={s} />}</Async>;
}

function DashboardView({ s }) {
  const names = s.junctions.map((j) => `Junction ${j}`);
  const busiest = s.junction_stats.reduce((a, b) => (b.mean > a.mean ? b : a));
  const daily = pivot(s.daily, "date", "Junction", "Vehicles", "Junction");
  const monthly = pivot(s.monthly, "month", "Junction", "Vehicles", "Junction");
  const hourly = pivot(s.hourly, "hour", "Junction", "Vehicles", "Junction");
  const weekday = pivot(s.weekday.map((r) => ({ ...r, day: WEEKDAYS[r.weekday] })), "day", "Junction", "Vehicles", "Junction");
  const comparison = s.junction_stats.map((r) => ({ junction: `Junction ${r.Junction}`, "Mean vehicles/hour": r.mean, "Max vehicles/hour": r.max }));
  const kpis = [
    { label: "Dataset Records", value: s.records.toLocaleString(), sub: "Hourly rows in the CSV", icon: BarChart3 },
    { label: "Junctions", value: s.junctions.length, sub: "Junction IDs in the CSV", icon: MapPin, tone: "info" },
    { label: "Mean Vehicles / hour", value: s.mean_vehicles.toFixed(1), sub: "All junctions", icon: Activity, tone: "success" },
    { label: "Max Vehicle Count", value: s.max_vehicles, sub: "Highest hourly count observed", icon: Gauge, tone: "warning" },
    { label: "Busiest Junction", value: `Junction ${busiest.Junction}`, sub: `Mean ${busiest.mean.toFixed(1)} vehicles/hour`, icon: TrendingUp },
    { label: "Date Range", value: dateOnly(s.start), sub: `to ${dateOnly(s.end)}`, icon: Calendar },
  ];
  const r = s.report;
  return (
    <div className="space-y-5">
      <p className="text-muted">
        Hourly vehicle counts per junction from the uploaded CSV. The counts are traffic volume per hour, not individual vehicle movements.
        Sections are tagged as observed, predicted or simulated.
      </p>
      <Tag kind="observed" />
      <KpiRow items={kpis} />
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Junction Comparison" subtitle="Mean and maximum vehicles per hour">
          <BarBox data={comparison} xKey="junction" series={["Mean vehicles/hour", "Max vehicles/hour"]} />
        </Card>
        <Card title="Hourly Traffic Trend" subtitle={`Mean vehicles per hour of day. Busiest hours overall: ${s.busiest_hours.map((h) => `${String(h).padStart(2, "0")}:00`).join(", ")}`}>
          <LineBox data={hourly} xKey="hour" series={names} dots />
        </Card>
      </div>
      <Card title="Daily Average Traffic" subtitle="Mean vehicles per hour for each day (Junction 4 only has data from 2017-01-01)">
        <LineBox data={daily} xKey="date" series={names} yLabel="Vehicles per hour" />
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Weekly Pattern" subtitle="Mean vehicles per hour by day of week">
          <BarBox data={weekday} xKey="day" series={names} />
        </Card>
        <Card title="Monthly Trend" subtitle="Mean vehicles per hour by month">
          <LineBox data={monthly} xKey="month" series={names} />
        </Card>
      </div>
      <Card title="Junction Statistics" subtitle="Vehicles per hour, calculated from the CSV">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="border-b border-line text-left text-muted">{["Junction", "Observations", "Mean", "Std", "Min", "Max", "Share of all vehicles", "First record"].map((h) => <th key={h} className="py-2 pr-3">{h}</th>)}</tr></thead>
            <tbody>
              {s.junction_stats.map((row) => (
                <tr key={row.Junction} className="border-b border-line/60">
                  <td className="py-2 pr-3">Junction {row.Junction}</td><td>{row.count.toLocaleString()}</td><td>{row.mean}</td><td>{row.std}</td>
                  <td>{row.min}</td><td>{row.max}</td><td>{row.share_of_total_pct}%</td><td>{dateOnly(row.first_record)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <h2 className="pt-2 font-display text-xl font-semibold">Model, signal plan and strategy comparison</h2>
      <ModelPanel />
      <SignalPanel start={s.default_scenario_start} />

      <Note>
        Data checks: {r.missing_values} missing values, {r.duplicates_dropped} duplicate records, {r.duplicate_ids} duplicate IDs, {r.hourly_gaps} gaps in the hourly
        sequence; all {r.rows_out.toLocaleString()} records were kept. The CSV has no vehicle categories, queue lengths, waiting times or signal timings, and no
        location information for the junctions.
      </Note>
    </div>
  );
}

function ModelPanel() {
  const info = useApi(api.modelInfo, []);
  return (
    <Card title="XGBoost evaluation (held-out test data)" subtitle="One-hour-ahead prediction of the hourly vehicle count, chronological train/test split">
      <Async state={info}>
        {(m) => {
          const rows = [["XGBoost", m.metrics.xgboost], ["Baseline: same as last hour", m.metrics.baseline_last_hour], ["Baseline: same hour last week", m.metrics.baseline_same_hour_last_week]].filter(([, v]) => v);
          return (
            <>
              <Tag kind="evaluated" />
              <table className="w-full text-sm">
                <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
                <tbody>{rows.map(([label, v]) => <tr key={label} className="border-b border-line/60"><td className="py-2">{label}</td><td>{v.mae.toFixed(3)}</td><td>{v.rmse.toFixed(3)}</td><td>{v.r2.toFixed(3)}</td></tr>)}</tbody>
              </table>
              <p className="mt-2 text-xs text-muted">Trained {dateOnly(m.train_range[0])} to {dateOnly(m.train_range[1])}; tested {dateOnly(m.test_range[0])} to {dateOnly(m.test_range[1])} ({m.test_rows.toLocaleString()} rows). MAE/RMSE are in vehicles per hour.</p>
            </>
          );
        }}
      </Async>
    </Card>
  );
}

function SignalPanel({ start }) {
  const plan = useApi(() => api.optimize({ start: start.replace(" ", "T"), demand_scale: DEMAND_SCALE }), [start]);
  const compare = useApi(() => api.compare({ start: start.replace(" ", "T"), hours: 1, n_seeds: 3, demand_scale: DEMAND_SCALE }), [start]);
  return (
    <>
      <Card title={`Predicted counts and optimized green times for ${start.slice(0, 16)}`}
            subtitle={`Genetic Algorithm on XGBoost-predicted demand (junction counts × ${DEMAND_SCALE}, a simulation assumption)`}>
        <Async state={plan}>
          {(r) => {
            const counts = r.predicted_vehicles.map((p, i) => ({ junction: `Junction ${i + 1}`, "Observed (CSV)": r.observed_vehicles[i], "Predicted (XGBoost)": Number(p.toFixed(1)) }));
            const greens = APPROACHES.map((approach, i) => ({ approach, "Fixed-Time": r.fixed_greens[i], "Genetic Algorithm": r.greens[i] }));
            const gain = r.improvement_pct;
            return (
              <div className="space-y-3">
                <Tag kind="predicted" /><Tag kind="simulated" />
                <div className="grid gap-5 lg:grid-cols-2">
                  <div><h3 className="mb-1 text-sm font-semibold">Vehicles in that hour: observed vs predicted</h3><BarBox data={counts} xKey="junction" series={["Observed (CSV)", "Predicted (XGBoost)"]} height={260} /></div>
                  <div><h3 className="mb-1 text-sm font-semibold">Green time per approach (s)</h3><BarBox data={greens} xKey="approach" series={["Fixed-Time", "Genetic Algorithm"]} height={260} /></div>
                </div>
                <p className="text-sm">
                  Optimized greens: <b>{r.greens.join(" / ")} s</b> (cycle {r.cycle_length} s, min {r.constraints.min_green} s, max {r.constraints.max_green} s). Estimated delay{" "}
                  {r.fitness.toFixed(1)} s per vehicle vs {r.fixed_fitness.toFixed(1)} s for the equal split ({gain.toFixed(1)}% lower, Webster-formula estimate).
                </p>
              </div>
            );
          }}
        </Async>
      </Card>
      <Card title="Fixed-Time vs optimized (simulated)" subtitle="Both strategies run in the simulator on identical random arrivals built from this hour's observed counts">
        <Async state={compare}>
          {(r) => {
            const rows = [["avg_wait_s", "Average wait (s)"], ["total_served", "Vehicles served"], ["max_queue", "Max queue (vehicles)"], ["end_queue", "Still queued at end"]];
            const pm = (name, key) => `${r.summary[name][key].mean.toFixed(1)} ± ${r.summary[name][key].std.toFixed(1)}`;
            return (
              <>
                <Tag kind="simulated" />
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">Metric (mean ± std, {r.n_seeds} seeds)</th><th>Fixed-Time</th><th>Optimized (GA)</th></tr></thead>
                  <tbody>{rows.map(([k, label]) => <tr key={k} className="border-b border-line/60"><td className="py-2">{label}</td><td>{pm("fixed", k)}</td><td>{pm("adaptive", k)}</td></tr>)}</tbody>
                </table>
                {r.wait_change_pct != null && <Note>Simulated average waiting time changed by {r.wait_change_pct.toFixed(1)}% with the optimized plan. These are simulator estimates, not measurements, and they depend on the demand multiplier ({r.demand_scale}×).</Note>}
              </>
            );
          }}
        </Async>
      </Card>
    </>
  );
}
