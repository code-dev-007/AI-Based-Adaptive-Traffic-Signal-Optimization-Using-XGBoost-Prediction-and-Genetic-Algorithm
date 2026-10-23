import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export const COLORS = ["#0f766e", "#f59e0b", "#6366f1", "#e11d48", "#0ea5e9", "#84cc16"];
const axis = { tick: { fill: "#5b6f6a", fontSize: 12 }, stroke: "#9db3ad" };

// `series` items are a key string or { key, color, dashed }.
const parse = (series, i) => (typeof series === "string" ? { key: series } : series);

function Frame({ chart: Chart, data, xKey, height, yLabel, xTick, children }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <Chart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="#e5ecea" strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tickFormatter={xTick} minTickGap={28} {...axis} />
        <YAxis {...axis} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#5b6f6a", fontSize: 12 } : undefined} />
        <Tooltip /> <Legend />
        {children}
      </Chart>
    </ResponsiveContainer>
  );
}

export function LineBox({ data, xKey, series, height = 300, yLabel, xTick, dots = false }) {
  return (
    <Frame chart={LineChart} {...{ data, xKey, height, yLabel, xTick }}>
      {series.map((item, i) => {
        const s = parse(item, i);
        return <Line key={s.key} type="monotone" dataKey={s.key} stroke={s.color ?? COLORS[i % COLORS.length]} strokeWidth={2}
                     strokeDasharray={s.dashed ? "6 4" : undefined} dot={dots} isAnimationActive={false} />;
      })}
    </Frame>
  );
}

export function BarBox({ data, xKey, series, height = 300, yLabel, xTick }) {
  return (
    <Frame chart={BarChart} {...{ data, xKey, height, yLabel, xTick }}>
      {series.map((item, i) => {
        const s = parse(item, i);
        return <Bar key={s.key} dataKey={s.key} fill={s.color ?? COLORS[i % COLORS.length]} radius={[3, 3, 0, 0]} />;
      })}
    </Frame>
  );
}
