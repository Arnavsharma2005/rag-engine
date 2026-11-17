// Shared styles and helpers for the Compare and Evaluate views.
// Colors come from the CSS variables in index.css, so both themes follow.
// An empty VITE_API_URL means "same origin", used when the backend serves the built UI
export const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5000"

export const card = {
  background: "var(--paper)", border: "1px solid var(--line)", borderRadius: 10,
  padding: "18px 22px"
}
export const label = { color: "var(--ink)", margin: "0 0 12px", fontSize: "1rem", fontWeight: 650 }
export const muted = { color: "var(--muted)", fontSize: "0.85rem" }

export async function readError(res) {
  try { return (await res.json()).error || res.statusText } catch { return res.statusText }
}
