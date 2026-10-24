import { motion } from "framer-motion";
import { AlertTriangle, Loader2 } from "lucide-react";

const TAGS = {
  observed: ["Observed (synthetic dataset)", "bg-teal-100 text-teal-800"],
  predicted: ["Predicted (XGBoost)", "bg-indigo-100 text-indigo-800"],
  simulated: ["Simulated", "bg-amber-100 text-amber-800"],
  evaluated: ["Evaluated on held-out test data", "bg-indigo-100 text-indigo-800"],
};
const TONES = { primary: "border-l-teal-700", success: "border-l-green-600", warning: "border-l-amber-500", info: "border-l-sky-500" };
const ICON_TONES = { primary: "bg-teal-100 text-teal-700", success: "bg-green-100 text-green-700", warning: "bg-amber-100 text-amber-700", info: "bg-sky-100 text-sky-700" };

export const inputClass = "mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-ink focus:border-primary focus:outline-none";

/** Label that tells the viewer where a number comes from: observed, predicted or simulated. */
export function Tag({ kind }) {
  const [label, style] = TAGS[kind];
  return <span className={`mb-2 mr-2 inline-block rounded px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide ${style}`}>{label}</span>;
}

export function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-line border-t-[3px] border-t-primary bg-white p-5 shadow-sm ${className}`}>
      {title && <h2 className="font-display text-lg font-semibold">{title}</h2>}
      {subtitle && <p className="mb-3 text-sm text-muted">{subtitle}</p>}
      {children}
    </section>
  );
}

export function KpiCard({ label, value, sub, icon: Icon, tone = "primary" }) {
  return (
    <motion.div whileHover={{ y: -2 }} className={`rounded-xl border border-line border-l-4 bg-white p-4 shadow-sm ${TONES[tone]}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</span>
        <span className={`flex h-7 w-7 items-center justify-center rounded-md ${ICON_TONES[tone]}`}><Icon size={16} /></span>
      </div>
      <div className="mt-1 font-display text-2xl font-bold">{value}</div>
      <div className="text-xs text-muted">{sub}</div>
    </motion.div>
  );
}

export function KpiRow({ items }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5">
      {items.map((item) => <KpiCard key={item.label} {...item} />)}
    </div>
  );
}

export const Note = ({ children }) => (
  <p className="my-3 rounded-lg border border-teal-200 border-l-4 border-l-primary bg-teal-50 px-4 py-3 text-sm text-teal-900">{children}</p>
);

export const Warning = ({ children }) => (
  <p className="my-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">{children}</p>
);

export function ErrorBox({ message }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
      <AlertTriangle size={18} className="mt-0.5 shrink-0" /> <span>{message}</span>
    </div>
  );
}

export const Loading = () => (
  <div className="flex items-center gap-2 py-6 text-sm text-muted"><Loader2 className="animate-spin" size={18} /> Loading...</div>
);

export const Field = ({ label, children }) => (
  <label className="block text-sm font-medium"><span>{label}</span>{children}</label>
);

export function Button({ children, variant = "secondary", ...props }) {
  const style = variant === "primary" ? "bg-primary text-white hover:bg-teal-800" : "border border-slate-400 bg-white text-ink hover:border-primary hover:text-primary";
  return <button {...props} className={`rounded-lg px-4 py-2 text-sm font-semibold transition disabled:opacity-50 ${style}`}>{children}</button>;
}

/** Shows Loading / ErrorBox while a useApi result is not ready; otherwise renders children(data). */
export function Async({ state, children }) {
  if (state.loading && !state.data) return <Loading />;
  if (state.error) return <ErrorBox message={state.error} />;
  return children(state.data);
}
