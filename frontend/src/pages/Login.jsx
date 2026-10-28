import { motion } from "framer-motion";
import { BrainCircuit, Dna, Eye, EyeOff, Loader2, LogIn, TrafficCone, UserPlus } from "lucide-react";
import { useEffect, useState } from "react";
import { api, errorMessage } from "../api/client.js";
import { ErrorBox, Field, inputClass } from "../components/ui.jsx";

const FEATURES = [
  [BrainCircuit, "XGBoost traffic forecasting", "Predicts hourly vehicle counts per junction"],
  [Dna, "Genetic Algorithm optimisation", "Finds green-light timings for a four-way intersection"],
  [TrafficCone, "Live signal simulation", "Fixed-Time vs Adaptive control, streamed in real time"],
];
const LAMPS = ["#ef4444", "#facc15", "#22c55e"];

/** A signal head that cycles red -> green -> yellow, just for decoration. */
function AnimatedSignal() {
  const [active, setActive] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setActive((n) => (n + 1) % 3), 1400);
    return () => clearInterval(timer);
  }, []);
  return (
    <div className="inline-flex flex-col gap-2 rounded-2xl border border-white/10 bg-black/30 p-3">
      {LAMPS.map((color, i) => (
        <motion.span key={color} className="h-9 w-9 rounded-full" transition={{ duration: 0.3 }}
                     animate={{ backgroundColor: active === i ? color : "#26373c", boxShadow: active === i ? `0 0 18px ${color}` : "0 0 0 transparent" }} />
      ))}
    </div>
  );
}

export default function Login({ onLogin }) {
  const [mode, setMode] = useState("login"); // "login" or "register"
  const [form, setForm] = useState({ username: "", password: "", confirm: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const registering = mode === "register";
  const set = (key) => (event) => setForm({ ...form, [key]: event.target.value });

  const submit = async (event) => {
    event.preventDefault();
    setError(null);
    if (registering && form.password !== form.confirm) return setError("The two passwords do not match.");
    setBusy(true);
    try {
      const { token, username } = await (registering ? api.register : api.login)({ username: form.username, password: form.password });
      onLogin(token, username);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const switchMode = () => { setMode(registering ? "login" : "register"); setError(null); };

  return (
    <div className="flex min-h-screen">
      <div className="hidden w-1/2 flex-col justify-between bg-sidebar p-12 text-white lg:flex">
        <div>
          <div className="font-display text-3xl font-bold">Traffic<span className="text-accent">AI</span></div>
          <div className="text-sm text-teal-200/70">Adaptive Traffic Signal Optimization</div>
        </div>
        <div className="space-y-8">
          <AnimatedSignal />
          <h1 className="font-display text-4xl font-bold leading-tight">Smarter signals,<br />shorter queues.</h1>
          <ul className="space-y-4">
            {FEATURES.map(([Icon, title, text]) => (
              <li key={title} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/10 text-accent"><Icon size={17} /></span>
                <span><span className="block font-semibold">{title}</span><span className="text-sm text-teal-100/70">{text}</span></span>
              </li>
            ))}
          </ul>
        </div>
        <p className="text-xs text-teal-200/60">MCA project prototype. Hourly vehicle counts per junction; signal results are simulated, not live traffic control.</p>
      </div>

      <div className="flex flex-1 items-center justify-center px-6 py-10">
        <motion.form onSubmit={submit} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-sm space-y-4">
          <div className="lg:hidden font-display text-2xl font-bold">Traffic<span className="text-accent">AI</span></div>
          <div>
            <h2 className="font-display text-3xl font-bold">{registering ? "Create account" : "Welcome back"}</h2>
            <p className="mt-1 text-sm text-muted">{registering ? "Choose a username and password to get started." : "Sign in to open the dashboard."}</p>
          </div>
          <Field label="Username">
            <input className={inputClass} value={form.username} onChange={set("username")} autoComplete="username" autoFocus required />
          </Field>
          <Field label="Password">
            <div className="relative">
              <input className={`${inputClass} pr-10`} type={showPassword ? "text" : "password"} value={form.password} onChange={set("password")}
                     autoComplete={registering ? "new-password" : "current-password"} required />
              <button type="button" onClick={() => setShowPassword(!showPassword)} title={showPassword ? "Hide password" : "Show password"}
                      className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted hover:text-ink">
                {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </Field>
          {registering && (
            <Field label="Confirm password">
              <input className={inputClass} type={showPassword ? "text" : "password"} value={form.confirm} onChange={set("confirm")} autoComplete="new-password" required />
            </Field>
          )}
          {error && <ErrorBox message={error} />}
          <button type="submit" disabled={busy}
                  className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary py-2.5 font-semibold text-white transition hover:bg-teal-800 disabled:opacity-60">
            {busy ? <Loader2 size={18} className="animate-spin" /> : registering ? <UserPlus size={18} /> : <LogIn size={18} />}
            {busy ? "Please wait..." : registering ? "Create account" : "Sign in"}
          </button>
          <p className="text-center text-sm text-muted">
            {registering ? "Already have an account?" : "New here?"}{" "}
            <button type="button" onClick={switchMode} className="font-semibold text-primary hover:underline">{registering ? "Sign in" : "Create an account"}</button>
          </p>
          <p className="text-center text-xs text-muted">Username: 3-30 letters, numbers, dot, dash or underscore. Password: at least 6 characters.</p>
        </motion.form>
      </div>
    </div>
  );
}
