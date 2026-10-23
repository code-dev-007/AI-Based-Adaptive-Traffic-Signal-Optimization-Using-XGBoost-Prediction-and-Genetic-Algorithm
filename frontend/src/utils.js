export const APPROACHES = ["Approach 1", "Approach 2", "Approach 3", "Approach 4"];
export const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const shortTime = (t) => t.slice(5, 16); // "2024-05-04 23:00:00" -> "05-04 23:00"
export const dateOnly = (t) => t.slice(0, 10);

/** Long rows -> wide rows for Recharts: [{x, "Junction 1": v, "Junction 2": v}, ...] */
export function pivot(rows, xKey, seriesKey, valueKey, label) {
  const byX = new Map();
  for (const row of rows) {
    if (!byX.has(row[xKey])) byX.set(row[xKey], { [xKey]: row[xKey] });
    byX.get(row[xKey])[`${label} ${row[seriesKey]}`] = row[valueKey];
  }
  return [...byX.values()];
}

export const totalQueue = (point) => [0, 1, 2, 3].reduce((sum, i) => sum + point[`queue_${i}`], 0);
