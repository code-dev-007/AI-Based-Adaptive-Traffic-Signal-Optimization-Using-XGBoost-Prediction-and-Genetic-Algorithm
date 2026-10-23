import { motion } from "framer-motion";

const LAMPS = [["RED", "#ef4444"], ["YELLOW", "#facc15"], ["GREEN", "#22c55e"]];

/** One approach: signal head with three lamps (only the active one lit) plus simulated statistics. */
export function SignalBox({ name, signal, greenSeconds, queue, served, avgWait }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-[#111c20] p-4 text-center shadow-md">
      <div className="font-display font-semibold text-white">{name}</div>
      <div className="mb-2 text-xs text-teal-200/70">Green plan: {greenSeconds} s</div>
      <div className="flex justify-center gap-1">
        {LAMPS.map(([state, color]) => (
          <motion.span key={state} className="h-8 w-8 rounded-full"
                       animate={{ backgroundColor: signal === state ? color : "#26373c", boxShadow: signal === state ? `0 0 14px ${color}` : "0 0 0 transparent" }}
                       transition={{ duration: 0.25 }} />
        ))}
      </div>
      <div className="mt-2 text-xs leading-6 text-teal-100/80">
        Queue: {queue} veh<br />Served: {served} veh<br />Avg wait: {avgWait.toFixed(0)} s
      </div>
    </div>
  );
}
