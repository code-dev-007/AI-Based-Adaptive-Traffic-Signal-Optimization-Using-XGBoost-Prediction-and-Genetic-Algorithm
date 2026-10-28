import { useState } from "react";
import { api } from "../api/client.js";
import { LineBox } from "../components/charts.jsx";
import { Async, Card, Field, inputClass, Note, Tag } from "../components/ui.jsx";
import { useApi } from "../hooks/useApi.js";
import { dateOnly, shortTime } from "../utils.js";

const BASELINES = [["xgboost", "XGBoost"], ["baseline_last_hour", "Baseline: same as last hour"], ["baseline_same_hour_yesterday", "Baseline: same hour yesterday"], ["baseline_same_hour_last_week", "Baseline: same hour last week"]];

export default function Prediction() {
  const summary = useApi(api.summary, []);
  return <Async state={summary}>{(s) => <PredictionView junctions={s.junctions} />}</Async>;
}

function PredictionView({ junctions }) {
  const [junction, setJunction] = useState(junctions[0]);
  const [horizon, setHorizon] = useState(12);
  const forecast = useApi(() => api.predict(junction, horizon), [junction, horizon]);
  const info = useApi(api.modelInfo, []);
  const test = useApi(() => api.predictOnTest(junction), [junction]);

  return (
    <div className="space-y-5">
      <p className="text-muted">Forecast vehicle counts for the hours after the end of the dataset, and see how the model scores on held-out data.</p>
      <div className="grid gap-4 md:grid-cols-2">
        <Field label="Junction">
          <select className={inputClass} value={junction} onChange={(e) => setJunction(Number(e.target.value))}>
            {junctions.map((j) => <option key={j} value={j}>Junction {j}</option>)}
          </select>
        </Field>
        <Field label={`Hours to predict: ${horizon}`}>
          <input type="range" min="1" max="48" className="mt-3 w-full accent-teal-700" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
        </Field>
      </div>

      <Card title={`Junction ${junction} forecast`} subtitle="Last 72 observed hours followed by the predicted hours">
        <Async state={forecast}>
          {(f) => {
            const rows = [
              ...f.recent_observed.map((r) => ({ time: r.DateTime, Observed: r.Vehicles })),
              ...f.predictions.map((r) => ({ time: r.DateTime, Predicted: r.predicted })),
            ];
            return (
              <>
                <Tag kind="observed" /><Tag kind="predicted" />
                <LineBox data={rows} xKey="time" xTick={shortTime} yLabel="Vehicles per hour"
                         series={[{ key: "Observed", color: "#0f766e" }, { key: "Predicted", color: "#f59e0b", dashed: true }]} />
                <p className="mt-2 text-xs text-muted">{f.note}</p>
                <div className="mt-3 max-h-56 overflow-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="text-left text-muted"><th className="py-1">Hour</th><th>Predicted vehicles</th></tr></thead>
                    <tbody>{f.predictions.map((r) => <tr key={r.DateTime} className="border-t border-line/60"><td className="py-1">{r.DateTime}</td><td>{r.predicted}</td></tr>)}</tbody>
                  </table>
                </div>
              </>
            );
          }}
        </Async>
      </Card>

      <Card title="Model evaluation" subtitle="Chronological split: the model is tested on the later part of the timeline">
        <Async state={info}>
          {(m) => {
            const change = m.mae_improvement_vs_best_baseline_pct;
            return (
              <>
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
                  <tbody>
                    {BASELINES.map(([key, label]) => (
                      <tr key={key} className="border-b border-line/60">
                        <td className="py-2">{label}</td><td>{m.metrics[key].mae.toFixed(3)}</td><td>{m.metrics[key].rmse.toFixed(3)}</td><td>{m.metrics[key].r2.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-3 text-xs text-muted">Target: hourly vehicle count per junction (traffic volume, not individual vehicles). MAE and RMSE are in vehicles per hour.</p>
                <Note>
                  Trained on {dateOnly(m.train_range[0])} to {dateOnly(m.train_range[1])}, tested on {dateOnly(m.test_range[0])} to {dateOnly(m.test_range[1])}.{" "}
                  {change != null && `Test-set MAE is ${Math.abs(change).toFixed(1)}% ${change >= 0 ? "lower" : "higher"} than the best baseline. `}
                  Model: {m.model}, trained {m.trained_at}.
                </Note>
              </>
            );
          }}
        </Async>
      </Card>

      <Card title={`Junction ${junction}: actual vs predicted on the test set`} subtitle="Most recent 168 test hours (one-hour-ahead predictions)">
        <Async state={test}>
          {(t) => (
            <>
              <Tag kind="observed" /><Tag kind="predicted" />
              <LineBox data={t.items} xKey="DateTime" xTick={shortTime} yLabel="Vehicles per hour"
                       series={[{ key: "actual", color: "#0f766e" }, { key: "predicted", color: "#f59e0b" }]} />
            </>
          )}
        </Async>
      </Card>
    </div>
  );
}
