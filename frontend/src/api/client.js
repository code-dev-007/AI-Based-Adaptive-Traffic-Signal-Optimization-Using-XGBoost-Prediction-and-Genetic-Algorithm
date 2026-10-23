import axios from "axios";

const http = axios.create({ baseURL: "/api/v1", timeout: 60000 });

const TOKEN_KEY = "traffic_ai_token";
const USER_KEY = "traffic_ai_user";
export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const getUsername = () => localStorage.getItem(USER_KEY);
export function saveSession(token, username) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username);
}
export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}
/** Tell the app to show the login page (token expired or rejected). */
export function forceLogout() {
  clearSession();
  window.dispatchEvent(new Event("auth:logout"));
}

http.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && getToken()) forceLogout(); // a wrong password at login has no token, so it is not affected
    return Promise.reject(error);
  },
);

/** Turn an Axios error into a message a person can act on. */
export function errorMessage(error) {
  if (error.response) {
    const detail = error.response.data?.detail;
    return typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Request failed (${error.response.status})`;
  }
  return "Cannot reach the API. Is the FastAPI server running on port 8000?";
}

const get = (url, params) => http.get(url, { params }).then((r) => r.data);
const post = (url, body) => http.post(url, body ?? {}).then((r) => r.data);

export const api = {
  login: (body) => post("/auth/login", body),
  register: (body) => post("/auth/register", body),
  health: () => get("/health"),
  summary: () => get("/traffic/summary"),
  modelInfo: () => get("/model-info"),
  predict: (junction, horizon) => post("/predict", { junction, horizon }),
  predictOnTest: (junction) => get("/predict/test", { junction, limit: 168 }),
  optimize: (body) => post("/optimize", body),
  compare: (body) => post("/compare", body),
  simStart: (body) => post("/simulation/start", body),
  simStop: () => post("/simulation/stop"),
  simReset: () => post("/simulation/reset"),
  runs: () => get("/runs", { limit: 15 }),
  clearRuns: () => http.delete("/runs").then((r) => r.data),
};
