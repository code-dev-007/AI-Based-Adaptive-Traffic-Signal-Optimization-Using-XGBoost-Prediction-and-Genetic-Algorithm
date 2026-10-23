import { Activity, BarChart3, Calendar, MapPin, TrendingUp } from "lucide-react";
import { api } from "../api/client.js";
import { BarBox, LineBox } from "../components/charts.jsx";
import { Async, Card, KpiRow, Note, Tag } from "../components/ui.jsx";
import { useApi } from "../hooks/useApi.js";
import { dateOnly, pivot, WEEKDAYS } from "../utils.js";

export default function Dashboard() {
  const state = useApi(api.summary, []);
  return <Async state={state}>{(s) => <DashboardView s={s} />}</Async>;
}

function DashboardView({ s }) {
  const names = s.junctions.map((j) => `Junction ${j}`);
  const busiest = s.junction_stats.reduce((a, b) => (b.mean > a.mean ? b : a));
  const daily = pivot(s.daily, "date", "Junction", "Vehicles", "Junction");
  const hourly = pivot(s.hourly, "hour", "Junction", "Vehicles", "Junction");
  const weekday = pivot(s.weekday.map((r) => ({ ...r, day: WEEKDAYS[r.weekday] })), "day", "Junction", "Vehicles", "Junction");
  const kpis = [
    { label: "Records", value: s.records.toLocaleString(), sub: "Hourly rows in the CSV", icon: BarChart3 },
    { label: "Junctions", value: s.junctions.length, sub: "In the dataset", icon: MapPin, tone: "info" },
    { label: "Mean Vehicles / hour", value: s.mean_vehicles.toFixed(1), sub: "All junctions", icon: Activity, tone: "success" },
    { label: "Busiest Junction", value: `Junction ${busiest.Junction}`, sub: `Mean ${busiest.mean.toFixed(1)} vehicles/hour`, icon: TrendingUp, tone: "warning" },
    { label: "Date Range", value: dateOnly(s.start), sub: `to ${dateOnly(s.end)}`, icon: Calendar },
  ];
  return (
    <div className="space-y-5">
      <p className="text-muted">Historical hourly vehicle counts per junction. All values on this page are observed dataset values.</p>
      <Tag kind="observed" />
      <KpiRow items={kpis} />
      <Card title="Daily Average Traffic" subtitle="Mean vehicles per hour for each day">
        <LineBox data={daily} xKey="date" series={names} yLabel="Vehicles per hour" />
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Hourly Profile" subtitle={`Peak hours used as a model feature: ${s.peak_hours.join(", ")}`}>
          <LineBox data={hourly} xKey="hour" series={names} dots />
        </Card>
        <Card title="Weekday Pattern" subtitle="Mean vehicles per hour by day of week">
          <BarBox data={weekday} xKey="day" series={names} />
        </Card>
      </div>
      <Card title="Junction Statistics" subtitle="Vehicles per hour">
        <table className="w-full text-sm">
          <thead><tr className="border-b border-line text-left text-muted">{["Junction", "Mean", "Std", "Min", "Max"].map((h) => <th key={h} className="py-2">{h}</th>)}</tr></thead>
          <tbody>
            {s.junction_stats.map((r) => (
              <tr key={r.Junction} className="border-b border-line/60">
                <td className="py-2">Junction {r.Junction}</td><td>{r.mean}</td><td>{r.std}</td><td>{r.min}</td><td>{r.max}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Note>
        Data checks: {s.report.missing_values} missing values, {s.report.duplicates_dropped} duplicates dropped, {s.report.hourly_gaps} gaps in the
        hourly sequence. The dataset is synthetic, not real-world traffic measurements.
      </Note>
    </div>
  );
}
