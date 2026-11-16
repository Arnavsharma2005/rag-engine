// Shared styles and helpers for the Compare and Evaluate views
// An empty VITE_API_URL means "same origin", used when the backend serves the built UI
export const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000"

export const card = {
  background: "#1a1a1a", border: "1px solid #2a2a2a", borderRadius: 8,
  padding: "16px 20px"
}
export const label = { color: "#4fc3f7", margin: "0 0 10px", fontSize: "0.8rem", letterSpacing: 1 }
export const muted = { color: "#888", fontSize: "0.8rem" }

export async function readError(res) {
  try { return (await res.json()).error || res.statusText } catch { return res.statusText }
}
