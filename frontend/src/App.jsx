import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, LayoutDashboard, LineChart, TrafficCone } from "lucide-react";
import { useEffect, useState } from "react";
import { clearSession, getToken, getUsername, saveSession } from "./api/client.js";
import { Header, Sidebar } from "./components/Layout.jsx";
import Analytics from "./pages/Analytics.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Login from "./pages/Login.jsx";
import Simulation from "./pages/Simulation.jsx";
import Prediction from "./pages/Prediction.jsx";

const PAGES = [
  { id: "dashboard", title: "Dashboard", icon: LayoutDashboard, Page: Dashboard },
  { id: "prediction", title: "Traffic Prediction", icon: LineChart, Page: Prediction },
  { id: "simulation", title: "Signal Simulation", icon: TrafficCone, Page: Simulation },
  { id: "analytics", title: "Analytics", icon: BarChart3, Page: Analytics },
];

export default function App() {
  const [current, setCurrent] = useState(() => (PAGES.some((p) => p.id === location.hash.slice(1)) ? location.hash.slice(1) : "dashboard"));
  const [username, setUsername] = useState(() => (getToken() ? getUsername() : null));
  useEffect(() => { location.hash = current; }, [current]);
  useEffect(() => {  // the API client fires this when a token is rejected or expires
    const signedOut = () => setUsername(null);
    window.addEventListener("auth:logout", signedOut);
    return () => window.removeEventListener("auth:logout", signedOut);
  }, []);
  const logout = () => { clearSession(); setUsername(null); };
  if (!username) return <Login onLogin={(token, name) => { saveSession(token, name); setUsername(name); }} />;
  const { title, Page } = PAGES.find((p) => p.id === current);
  return (
    <div className="flex min-h-screen">
      <Sidebar pages={PAGES} current={current} onSelect={setCurrent} username={username} onLogout={logout} />
      <main className="min-w-0 flex-1 px-8 py-6">
        <Header title={title} />
        <AnimatePresence mode="wait">
          <motion.div key={current} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
            <Page />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
