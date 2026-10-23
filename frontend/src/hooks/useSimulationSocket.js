import { useEffect, useRef, useState } from "react";
import { forceLogout, getToken } from "../api/client.js";

/** Live simulation state from the backend WebSocket (one message per second). Reconnects automatically. */
export function useSimulationSocket(speed) {
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);
  const socket = useRef(null);

  useEffect(() => {
    let closed = false;
    let timer;
    const connect = () => {
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/simulation?token=${encodeURIComponent(getToken() ?? "")}`);
      socket.current = ws;
      ws.onopen = () => { setConnected(true); ws.send(JSON.stringify({ speed })); };
      ws.onmessage = (event) => { const data = JSON.parse(event.data); if (!data.idle) setState(data); };
      ws.onclose = (event) => {
        setConnected(false);
        if (event.code === 4401) { closed = true; forceLogout(); return; } // not signed in: go back to the login page
        if (!closed) timer = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => { closed = true; clearTimeout(timer); socket.current?.close(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ speed }));
  }, [speed]);

  return { state, setState, connected };
}
