export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: { ink: "#10231f", sidebar: "#0b1f24", page: "#f3f6f5", muted: "#5b6f6a", line: "#d9e3e0", primary: "#0f766e", accent: "#f59e0b" },
      fontFamily: { sans: ["DM Sans", "system-ui", "sans-serif"], display: ["Space Grotesk", "sans-serif"] },
      keyframes: { blink: { "50%": { opacity: 0.25 } } },
      animation: { blink: "blink 1.2s infinite" },
    },
  },
  plugins: [],
};
