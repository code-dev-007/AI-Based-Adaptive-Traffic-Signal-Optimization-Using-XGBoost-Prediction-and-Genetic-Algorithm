import { motion } from "framer-motion";
import { LogOut, UserCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api/client.js";

/** Green = connected, amber = setup needed (dataset/model missing), red = API offline. */
export function StatusBadge() {
  const [status, setStatus] = useState({ state: "checking", label: "Checking API...", detail: [] });
  useEffect(() => {
    const check = () =>
      api.health()
        .then((h) => setStatus(h.dataset_available && h.model_available
          ? { state: "ok", label: "API connected", detail: [] }
          : { state: "warn", label: "API connected - setup needed", detail: h.detail }))
        .catch(() => setStatus({ state: "off", label: "API offline", detail: [] }));
    check();
    const timer = setInterval(check, 5000);
    return () => clearInterval(timer);
  }, []);
  const style = {
    ok: "bg-sidebar text-emerald-100", warn: "border border-amber-500 bg-amber-100 text-amber-900",
    off: "border border-red-800 bg-red-600 text-white shadow-md shadow-red-600/40", checking: "bg-slate-200 text-slate-700",
  }[status.state];
  const dot = { ok: "bg-green-500", warn: "bg-amber-500", off: "animate-blink bg-white", checking: "bg-slate-400" }[status.state];
  return (
    <div className="flex flex-col items-end gap-1">
      <span className={`inline-flex items-center gap-2 whitespace-nowrap rounded-md px-3 py-1.5 text-xs font-bold ${style}`}>
        <i className={`h-2 w-2 rounded-full ${dot}`} /> {status.label}
      </span>
      {status.detail.map((text) => <span key={text} className="max-w-xs text-right text-xs text-amber-800">{text}</span>)}
    </div>
  );
}

export function Sidebar({ pages, current, onSelect, username, onLogout }) {
  return (
    <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col bg-sidebar px-3 py-4">
      <div className="mb-3 border-b border-white/10 px-2 pb-4">
        <div className="font-display text-2xl font-bold text-white">Traffic<span className="text-accent">AI</span></div>
        <div className="text-xs text-teal-200/70">Adaptive Signal Optimization</div>
      </div>
      <nav className="flex flex-col gap-1">
        {pages.map(({ id, title, icon: Icon }) => (
          <button key={id} onClick={() => onSelect(id)}
                  className={`relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition ${current === id ? "font-bold text-white" : "text-teal-100/70 hover:bg-white/5 hover:text-white"}`}>
            {current === id && <motion.span layoutId="active-nav" className="absolute inset-0 rounded-lg border-l-[3px] border-accent bg-amber-500/15" />}
            <Icon size={18} className="relative" /> <span className="relative">{title}</span>
          </button>
        ))}
      </nav>
      <div className="mt-auto border-t border-white/10 pt-3">
        <div className="mb-2 flex items-center justify-between rounded-lg bg-white/5 px-3 py-2 text-sm text-white">
          <span className="flex min-w-0 items-center gap-2"><UserCircle size={18} className="shrink-0 text-accent" /><span className="truncate">{username}</span></span>
          <button onClick={onLogout} title="Sign out" className="rounded p-1 text-teal-100/70 hover:bg-white/10 hover:text-white"><LogOut size={16} /></button>
        </div>
        <p className="px-2 text-xs leading-relaxed text-teal-200/70">
          Dataset: synthetic (Kaggle). Queues, waits and signal timings on the simulation pages are simulated estimates.
        </p>
      </div>
    </aside>
  );
}

export function Header({ title }) {
  return (
    <header className="mb-5 flex items-start justify-between gap-4">
      <div>
        <h1 className="border-l-[5px] border-accent pl-3 font-display text-3xl font-bold">{title}</h1>
        <p className="ml-4 mt-1 text-sm text-muted">AI-Based Adaptive Traffic Signal Optimization (XGBoost + Genetic Algorithm)</p>
      </div>
      <StatusBadge />
    </header>
  );
}
