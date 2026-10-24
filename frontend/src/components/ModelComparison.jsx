import { useState } from "react";
import { Award, RefreshCw } from "lucide-react";
import { api } from "../api/client.js";
import { useApi } from "../hooks/useApi.js";
import { BarBox } from "./charts.jsx";
import { Button, Card, ErrorBox, Loading, Note, Tag } from "./ui.jsx";

const fmt = (value, digits = 3) => Number(value).toFixed(digits);

/** Prediction-model comparison (Linear Regression vs Random Forest vs XGBoost). Not the Fixed-Time vs Adaptive comparison. */
export default function ModelComparison() {
  const [reload, setReload] = useState(0);
  const { data, error, loading } = useApi(() => api.modelComparison(), [reload]);
  const refresh = (
    <Button onClick={() => setReload((n) => n + 1)} disabled={loading}>
      <span className="inline-flex items-center gap-2"><RefreshCw size={14} className={loading ? "animate-spin" : ""} /> {loading ? "Loading..." : "Refresh"}</span>
    </Button>
  );
  return (
    <Card title="Machine Learning Model Comparison" subtitle="Which model predicts hourly vehicle counts best? This evaluates prediction models, not signal-control strategies.">
      <div className="mb-3">{refresh}</div>
      {loading && !data && <Loading />}
      {error && <ErrorBox message={error} />}
      {data && <Result r={data} />}
    </Card>
  );
}

function Result({ r }) {
  const e = r.evaluation;
  const rows = r.models;
  const chart = (key) => rows.map((m) => ({ model: m.name, [key]: Number(fmt(m[key.toLowerCase().replace("²", "2")], 3)) }));
  return (
    <div className="space-y-4">
      <Tag kind="evaluated" />
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead><tr className="border-b border-line text-left text-muted"><th className="py-2">Model</th><th>MAE (lower is better)</th><th>RMSE (lower is better)</th><th>R² (higher is better)</th></tr></thead>
          <tbody>{rows.map((m) => (
            <tr key={m.name} className="border-b border-line/60">
              <td className="py-2 font-medium">
                {m.name}
                {m.name === r.best_model && (
                  <span className="ml-2 inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-[11px] font-bold text-green-800"><Award size={12} /> Lowest {r.selection_metric}</span>
                )}
              </td>
              <td>{fmt(m.mae)}</td><td>{fmt(m.rmse)}</td><td>{fmt(m.r2)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className="grid gap-5 lg:grid-cols-3">
        <div><h3 className="mb-1 text-sm font-semibold">MAE (vehicles)</h3><BarBox data={chart("MAE")} xKey="model" series={["MAE"]} height={240} /></div>
        <div><h3 className="mb-1 text-sm font-semibold">RMSE (vehicles)</h3><BarBox data={chart("RMSE")} xKey="model" series={[{ key: "RMSE", color: "#f59e0b" }]} height={240} /></div>
        <div><h3 className="mb-1 text-sm font-semibold">R² score</h3><BarBox data={chart("R²")} xKey="model" series={[{ key: "R²", color: "#6366f1" }]} height={240} /></div>
      </div>
      <Note>
        Best model: <strong>{r.best_model}</strong>, chosen by lowest {r.selection_metric} on the test set only. All three models use the same {e.features.length} features
        (hour, weekday, lags, rolling means, junction), predict <code>{e.target}</code>, and are scored on the same {e.test_rows.toLocaleString()} held-out rows
        ({e.test_range[0].slice(0, 10)} to {e.test_range[1].slice(0, 10)}) after training on the earlier {e.train_rows.toLocaleString()} rows (chronological split, no shuffling).
      </Note>
      <p className="text-xs text-muted">
        Source: reproducible backend pipeline, computed {e.computed_at.replace("T", " ")} and cached. These figures differ from Model_Comparison.ipynb, which used a
        different split and no lag features, so the two should not be mixed.
      </p>
    </div>
  );
}
