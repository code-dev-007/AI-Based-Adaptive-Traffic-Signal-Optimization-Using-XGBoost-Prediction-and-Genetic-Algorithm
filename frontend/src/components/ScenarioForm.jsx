import { useState } from "react";
import { dateOnly } from "../utils.js";
import { Field, inputClass } from "./ui.jsx";

/** Scenario state shared by the Simulation and Analytics pages. Starts at the first hour of the test period. */
export function useScenario(summary) {
  return useState({ day: dateOnly(summary.test_start), hour: 7, demand_scale: 10, seed: 42 });
}

export const scenarioStart = (s) => `${s.day}T${String(s.hour).padStart(2, "0")}:00:00`;

export function ScenarioForm({ summary, value, onChange }) {
  const set = (key) => (event) => onChange({ ...value, [key]: event.target.type === "date" ? event.target.value : Number(event.target.value) });
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
      <Field label="Start date">
        <input type="date" className={inputClass} value={value.day} min={dateOnly(summary.test_start)} max={dateOnly(summary.end)} onChange={set("day")} />
      </Field>
      <Field label="Start hour">
        <select className={inputClass} value={value.hour} onChange={set("hour")}>
          {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2, "0")}:00</option>)}
        </select>
      </Field>
      <Field label={`Demand multiplier: ${value.demand_scale}x (simulation assumption)`}>
        <input type="range" min="1" max="20" step="0.5" className="mt-3 w-full accent-teal-700" value={value.demand_scale} onChange={set("demand_scale")} />
      </Field>
      <Field label="Random seed">
        <input type="number" className={inputClass} value={value.seed} onChange={set("seed")} />
      </Field>
    </div>
  );
}
