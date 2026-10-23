
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Forward API and WebSocket requests to FastAPI
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: true,

    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
