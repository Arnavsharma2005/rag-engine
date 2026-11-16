import { useState, useEffect, useCallback, useRef } from "react"
import { API, card, label, muted, readError } from "./ui"

const ACCENT = "#ab47bc"
const BAR = "#3987e5"
const GOOD = "#66bb6a"
const BAD = "#ef5350"
// Latency stages, in pipeline order (categorical slots 1-5, dark-surface steps)
const STAGES = [
  ["embed_ms", "embed", "#3987e5"],
  ["plan_ms", "plan", "#8e7cc3"],
  ["rewrite_ms", "rewrite", "#d95926"],
  ["search_ms", "search", "#199e70"],
  ["rerank_ms", "rerank", "#c98500"],
  ["generate_ms", "generate", "#d55181"],
  ["check_ms", "self-check", "#5fa8a8"],
]
const TABLE_KEYS = ["hit1", "hit5", "mrr", "ndcg5", "correctness", "faithfulness", "relevance", "f1", "refusal", "grounded", "total_ms"]

const input = {
  background: "#0f0f0f", color: "#e0e0e0", border: "1px solid #333", borderRadius: 4,
  padding: "4px 6px", fontFamily: "inherit", fontSize: "0.85rem", width: 60
}
const button = (enabled, color = "#4fc3f7") => ({
  padding: "8px 16px", borderRadius: 6, border: "none", fontFamily: "inherit",
  background: enabled ? color : "#333", color: enabled ? "#000" : "#666",
  cursor: enabled ? "pointer" : "default", fontWeight: "bold", fontSize: "0.85rem"
})
const th = { textAlign: "right", padding: "6px 8px", fontWeight: "normal", color: "#888", whiteSpace: "nowrap" }
const td = { textAlign: "right", padding: "6px 8px", whiteSpace: "nowrap" }

const fmt = (v, key) => v == null ? "–" : key === "total_ms" ? `${Math.round(v)}` : v.toFixed(2)
const fmtMs = ms => ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`

async function getJson(path, opts) {
  const res = await fetch(`${API}${path}`, opts)
  if (!res.ok) throw new Error(await readError(res))
  return res.json()
}
const post = (path, body) => getJson(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {})
})

// ---------- test set ----------

function TestSet({ dataset, busy, onGenerate, onDelete }) {
  const [n, setN] = useState(25)
  const items = dataset?.items || []
  return (
    <div style={card}>
      <p style={label}>1 · TEST SET</p>
      <p style={{ ...muted, margin: "0 0 12px", lineHeight: 1.6 }}>
        llama3.2 reads chunks sampled across the whole PDF and writes a question, a short reference answer and
        the exact sentence that contains it. Every chunk containing that sentence counts as the right chunk to retrieve.
      </p>
      {items.length > 0 && (
        <p style={{ fontSize: "0.85rem", margin: "0 0 12px", color: dataset.current ? "#e0e0e0" : "#ffa726" }}>
          {dataset.current ? "✓" : "⚠"} {items.length} questions from {dataset.doc.filename}
          <span style={muted}> · written {dataset.generated_at?.replace("T", " ")}
            {dataset.stats && ` · ${dataset.stats.attempted} chunks tried`}
            {dataset.stats?.rejected && Object.keys(dataset.stats.rejected).length > 0 &&
              ` · rejected: ${Object.entries(dataset.stats.rejected).map(([k, v]) => `${v} ${k}`).join(", ")}`}
          </span>
          {!dataset.current && " — made for a different document, write a new one"}
        </p>
      )}
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ fontSize: "0.85rem" }}>
          Questions <input type="number" min={1} max={200} value={n} style={input}
            onChange={e => setN(+e.target.value)} disabled={busy} />
        </label>
        <button style={button(!busy)} disabled={busy} onClick={() => onGenerate(n)}>
          {items.length ? "Write a new test set" : "Write test set"}
        </button>
        <span style={muted}>about one LLM call per question</span>
      </div>
      {items.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: "pointer", fontSize: "0.85rem" }}>Review questions (remove bad ones before running)</summary>
          <div style={{ maxHeight: 360, overflowY: "auto", marginTop: 8 }}>
            {items.map(it => (
              <div key={it.id} style={{ borderTop: "1px solid #2a2a2a", padding: "8px 0", display: "flex", gap: 10 }}>
                <div style={{ flex: 1, minWidth: 0, fontSize: "0.82rem", lineHeight: 1.6 }}>
                  <div>Q{it.id + 1}. {it.question}</div>
                  <div style={muted}>Answer: {it.answer}</div>
                  <div style={muted}>
                    {it.pages} · {it.style} · relevant chunks {it.relevant.map(r => `#${r}`).join(", ")}
                    {!it.exact_evidence && " · evidence paraphrased"}
                  </div>
                </div>
                <button title="Remove this question" disabled={busy} onClick={() => onDelete(it.id)}
                  style={{ background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: "1rem" }}>×</button>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

// ---------- run settings ----------

function RunPanel({ meta, dataset, busy, onRun }) {
  const [picked, setPicked] = useState(() => new Set(meta.configs.map(c => c.id)))
  const [judge, setJudge] = useState(true)
  const [workers, setWorkers] = useState(1)
  const [limit, setLimit] = useState("")
  const ready = dataset?.current && dataset.items.length > 0
  const nq = Math.min(dataset?.items.length || 0, +limit || Infinity)
  const chosen = meta.configs.filter(c => picked.has(c.id))
  const rag = chosen.filter(c => c.kind !== "plain").length
  const rewrites = chosen.some(c => c.flags.rewrite) ? 1 : 0
  const reranks = chosen.filter(c => c.flags.rerank).length
  const calls = nq * (chosen.length + rewrites + reranks + (judge ? chosen.length + rag : 0))
  const toggle = id => {
    const s = new Set(picked)
    s.has(id) ? s.delete(id) : s.add(id)
    setPicked(s)
  }
  const groups = [
    ["Baselines and agent", meta.configs.filter(c => c.kind !== "advanced")],
    ["Advanced steps (every on/off combination, for ablation)", meta.configs.filter(c => c.kind === "advanced")],
  ]
  return (
    <div style={card}>
      <p style={label}>2 · RUN</p>
      {groups.map(([name, cs]) => (
        <div key={name} style={{ marginBottom: 10 }}>
          <p style={{ ...muted, margin: "0 0 6px" }}>{name}</p>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: "0.85rem" }}>
            {cs.map(c => (
              <label key={c.id} style={{ cursor: "pointer", color: picked.has(c.id) ? "#e0e0e0" : "#666" }}>
                <input type="checkbox" checked={picked.has(c.id)} disabled={busy} onChange={() => toggle(c.id)}
                  style={{ marginRight: 6, accentColor: ACCENT }} />
                {c.name}
              </label>
            ))}
          </div>
        </div>
      ))}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", fontSize: "0.85rem", margin: "12px 0" }}>
        <label style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={judge} disabled={busy} onChange={e => setJudge(e.target.checked)}
            style={{ marginRight: 6, accentColor: ACCENT }} />
          LLM judge (correctness, faithfulness, relevance)
        </label>
        <label>Questions <input type="number" min={1} placeholder="all" value={limit} style={input}
          onChange={e => setLimit(e.target.value)} disabled={busy} /></label>
        <label title="Questions answered at once. Raise it only if Ollama runs with OLLAMA_NUM_PARALLEL > 1">
          Parallel <input type="number" min={1} max={8} value={workers} style={{ ...input, width: 44 }}
            onChange={e => setWorkers(+e.target.value)} disabled={busy} />
        </label>
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <button style={button(ready && !busy && chosen.length > 0, ACCENT)}
          disabled={!ready || busy || !chosen.length}
          onClick={() => onRun({ configs: [...picked], judge, workers, limit: limit ? +limit : undefined })}>
          Run evaluation
        </button>
        <span style={muted}>
          {ready
            ? `${nq} questions × ${chosen.length} configurations · up to ${calls} LLM calls (repeated prompts are reused)`
            : "Write a test set for the indexed document first"}
        </span>
      </div>
    </div>
  )
}

function Progress({ job, onCancel }) {
  if (!job || job.status === "idle") return null
  const pct = job.total ? job.done / job.total : 0
  const running = job.status === "running"
  return (
    <div style={{ ...card, borderColor: job.status === "error" ? BAD : "#2a2a2a" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: 8, gap: 12 }}>
        <span>
          {job.kind === "dataset" ? "Writing test set" : "Evaluating"}
          {running ? ` · ${job.done} / ${job.total || "?"}` : ` · ${job.status}`}
        </span>
        {running && <button onClick={onCancel} style={{ ...button(true, "#555"), padding: "2px 10px", color: "#e0e0e0" }}>Cancel</button>}
      </div>
      {running && (
        <div style={{ height: 6, background: "#2a2a2a", borderRadius: 3, overflow: "hidden" }}>
          <div style={{ width: `${pct * 100}%`, height: "100%", background: ACCENT, transition: "width 0.3s" }} />
        </div>
      )}
      <p style={{ ...muted, margin: "8px 0 0", color: job.status === "error" ? BAD : "#888" }}>
        {job.status === "error" ? job.error : job.message}
      </p>
    </div>
  )
}

// ---------- results ----------

function Delta({ d, metric }) {
  if (!d || d.mean == null || Math.abs(d.mean) < 1e-9) return null
  const better = metric.higher_better ? d.mean > 0 : d.mean < 0
  const sig = d.p < 0.05
  const text = metric.key === "total_ms" ? `${d.mean > 0 ? "+" : ""}${Math.round(d.mean)}` : `${d.mean > 0 ? "+" : ""}${d.mean.toFixed(2)}`
  return (
    <span title={`vs basic RAG: ${text} [${fmt(d.lo, metric.key)}, ${fmt(d.hi, metric.key)}], p = ${d.p.toFixed(3)}`}
      style={{ fontSize: "0.72rem", marginLeft: 4, color: sig ? (better ? GOOD : BAD) : "#666" }}>
      {text}{sig && "*"}
    </span>
  )
}

function Leaderboard({ run, metrics }) {
  const s = run.summary
  const keys = TABLE_KEYS.filter(k => run.configs.some(c => s.configs[c.id][k]))
  const best = {}
  for (const k of keys) {
    const m = metrics[k]
    const vals = run.configs.map(c => s.configs[c.id][k]?.mean).filter(v => v != null)
    best[k] = m.higher_better ? Math.max(...vals) : Math.min(...vals)
  }
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: "0.82rem", width: "100%" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #333" }}>
            <th style={{ ...th, textAlign: "left" }}>Configuration</th>
            {keys.map(k => <th key={k} style={th} title={metrics[k].higher_better ? "higher is better" : "lower is better"}>
              {metrics[k].label}{metrics[k].higher_better ? " ↑" : " ↓"}</th>)}
            <th style={th}>p95 ms</th>
          </tr>
        </thead>
        <tbody>
          {run.configs.map(c => {
            const m = s.configs[c.id]
            return (
              <tr key={c.id} style={{ borderBottom: "1px solid #222", background: c.id === "basic" ? "#1f1f1f" : undefined }}>
                <td style={{ ...td, textAlign: "left" }}>{c.name}</td>
                {keys.map(k => (
                  <td key={k} style={{ ...td, fontWeight: m[k] && m[k].mean === best[k] ? "bold" : "normal" }}
                    title={m[k] ? `95% CI ${fmt(m[k].lo, k)} – ${fmt(m[k].hi, k)} (n = ${m[k].n})` : ""}>
                    {fmt(m[k]?.mean, k)}
                    <Delta d={s.deltas[c.id]?.[k]} metric={metrics[k]} />
                  </td>
                ))}
                <td style={td}>{fmt(m.latency.p95, "total_ms")}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p style={{ ...muted, margin: "8px 0 0" }}>
        Bold is the best in each column. Small numbers are the paired difference from basic RAG (shaded row) on the same
        questions; * marks p &lt; 0.05 by bootstrap, green better, red worse. Hover a cell for its 95% interval.
      </p>
    </div>
  )
}

function MetricBars({ run, metrics }) {
  const options = Object.values(metrics).filter(m => m.key !== "total_ms" && run.configs.some(c => run.summary.configs[c.id][m.key]))
  const [key, setKey] = useState("hit5")
  const k = options.some(o => o.key === key) ? key : options[0]?.key
  if (!k) return null
  const rows = run.configs.map(c => ({ c, v: run.summary.configs[c.id][k] })).filter(r => r.v)
  const W = 520, rowH = 24, left = 150, right = 44, plotW = W - left - right
  const x = v => left + v * plotW
  return (
    <div style={{ flex: "1 1 420px", minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <p style={{ ...muted, margin: 0 }}>Score with 95% interval</p>
        <select value={k} onChange={e => setKey(e.target.value)} style={{ ...input, width: "auto" }}>
          {options.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
        </select>
      </div>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH + 20}`} style={{ width: "100%", fontFamily: "monospace" }}>
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={rows.length * rowH} stroke="#2a2a2a" />
            <text x={x(t)} y={rows.length * rowH + 14} fill="#888" fontSize="10" textAnchor="middle">{t}</text>
          </g>
        ))}
        {rows.map(({ c, v }, i) => {
          const y = i * rowH
          return (
            <g key={c.id}>
              <title>{`${c.name}: ${v.mean.toFixed(3)} (95% CI ${v.lo.toFixed(3)} – ${v.hi.toFixed(3)}, n = ${v.n})`}</title>
              <rect x={0} y={y} width={W} height={rowH} fill="transparent" />
              <text x={left - 8} y={y + rowH / 2 + 4} fill={c.id === "basic" ? "#e0e0e0" : "#aaa"} fontSize="11" textAnchor="end">
                {c.name.length > 20 ? c.name.slice(0, 19) + "…" : c.name}
              </text>
              <rect x={left} y={y + 5} width={Math.max(0, x(v.mean) - left)} height={rowH - 10} rx={3} fill={BAR} />
              <line x1={x(v.lo)} x2={x(v.hi)} y1={y + rowH / 2} y2={y + rowH / 2} stroke="#e0e0e0" strokeWidth={1.5} />
              <line x1={x(v.lo)} x2={x(v.lo)} y1={y + 8} y2={y + rowH - 8} stroke="#e0e0e0" strokeWidth={1.5} />
              <line x1={x(v.hi)} x2={x(v.hi)} y1={y + 8} y2={y + rowH - 8} stroke="#e0e0e0" strokeWidth={1.5} />
              <text x={W - 4} y={y + rowH / 2 + 4} fill="#e0e0e0" fontSize="11" textAnchor="end">{v.mean.toFixed(2)}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function LatencyBars({ run }) {
  const rows = run.configs.map(c => ({ c, l: run.summary.configs[c.id].latency })).filter(r => r.l.stages && r.l.p50 != null)
  const max = Math.max(1, ...rows.map(r => STAGES.reduce((a, [k]) => a + (r.l.stages[k] || 0), 0)))
  const W = 520, rowH = 24, left = 150, right = 56, plotW = W - left - right
  const used = STAGES.filter(([k]) => rows.some(r => r.l.stages[k] > 0))
  return (
    <div style={{ flex: "1 1 420px", minWidth: 0 }}>
      <p style={{ ...muted, margin: "0 0 8px" }}>Mean latency per stage</p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: "0.75rem", marginBottom: 6 }}>
        {used.map(([k, name, color]) => (
          <span key={k}><span style={{ display: "inline-block", width: 10, height: 10, background: color, borderRadius: 2, marginRight: 4 }} />{name}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH}`} style={{ width: "100%", fontFamily: "monospace" }}>
        {rows.map(({ c, l }, i) => {
          const y = i * rowH
          let acc = left
          const total = STAGES.reduce((a, [k]) => a + (l.stages[k] || 0), 0)
          return (
            <g key={c.id}>
              <title>{`${c.name}: ${STAGES.filter(([k]) => l.stages[k] > 0).map(([k, n]) => `${n} ${fmtMs(l.stages[k])}`).join(", ")} · p50 ${fmtMs(l.p50)}, p95 ${fmtMs(l.p95)}`}</title>
              <rect x={0} y={y} width={W} height={rowH} fill="transparent" />
              <text x={left - 8} y={y + rowH / 2 + 4} fill="#aaa" fontSize="11" textAnchor="end">
                {c.name.length > 20 ? c.name.slice(0, 19) + "…" : c.name}
              </text>
              {STAGES.map(([k, , color]) => {
                const w = (l.stages[k] || 0) / max * plotW
                if (w <= 0) return null
                const r = <rect key={k} x={acc} y={y + 5} width={Math.max(1, w - 2)} height={rowH - 10} rx={2} fill={color} />
                acc += w
                return r
              })}
              <text x={W - 4} y={y + rowH / 2 + 4} fill="#e0e0e0" fontSize="11" textAnchor="end">{fmtMs(total)}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function Tradeoff({ run, metrics }) {
  const key = run.configs.some(c => run.summary.configs[c.id].correctness) ? "correctness" : "hit5"
  const pts = run.configs.map(c => ({ c, q: run.summary.configs[c.id][key]?.mean, t: run.summary.configs[c.id].latency.p50 }))
    .filter(p => p.q != null && p.t != null)
  if (pts.length < 2) return null
  // A configuration is on the frontier if nothing else is both faster and at least as good
  const frontier = new Set(pts.filter(p => !pts.some(o => o !== p && o.t <= p.t && o.q >= p.q && (o.t < p.t || o.q > p.q))).map(p => p.c.id))
  const W = 520, H = 240, L = 44, B = 30, T = 10, R = 16
  const tMax = Math.max(...pts.map(p => p.t)) * 1.1
  const x = t => L + t / tMax * (W - L - R)
  const y = q => T + (1 - q) * (H - T - B)
  const line = pts.filter(p => frontier.has(p.c.id)).sort((a, b) => a.t - b.t)
  return (
    <div style={{ flex: "1 1 420px", minWidth: 0 }}>
      <p style={{ ...muted, margin: "0 0 8px" }}>{metrics[key].label} vs median latency · labelled points are the best trade-offs</p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", fontFamily: "monospace" }}>
        {[0, 0.5, 1].map(q => (
          <g key={q}>
            <line x1={L} x2={W - R} y1={y(q)} y2={y(q)} stroke="#2a2a2a" />
            <text x={L - 6} y={y(q) + 4} fill="#888" fontSize="10" textAnchor="end">{q}</text>
          </g>
        ))}
        <text x={L} y={H - 8} fill="#888" fontSize="10">0</text>
        <text x={W - R} y={H - 8} fill="#888" fontSize="10" textAnchor="end">{fmtMs(tMax)}</text>
        <polyline points={line.map(p => `${x(p.t)},${y(p.q)}`).join(" ")} fill="none" stroke={ACCENT} strokeWidth={2} strokeDasharray="4 3" />
        {pts.map(p => (
          <g key={p.c.id}>
            <title>{`${p.c.name}: ${metrics[key].label} ${p.q.toFixed(2)}, p50 ${fmtMs(p.t)}`}</title>
            <circle cx={x(p.t)} cy={y(p.q)} r={10} fill="transparent" />
            <circle cx={x(p.t)} cy={y(p.q)} r={5} fill={frontier.has(p.c.id) ? ACCENT : "#666"} stroke="#1a1a1a" strokeWidth={2} />
            {frontier.has(p.c.id) && (
              <text x={x(p.t) + 8} y={y(p.q) - 8} fill="#e0e0e0" fontSize="10"
                textAnchor={x(p.t) > W - 130 ? "end" : "start"}>{p.c.name}</text>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}

function Effects({ run, metrics }) {
  const effects = run.summary.effects
  const steps = Object.keys(effects)
  if (!steps.length) return null
  const keys = TABLE_KEYS.filter(k => steps.some(s => effects[s][k]))
  const names = { rewrite: "Query rewriting", hybrid: "Hybrid search", rerank: "Reranking" }
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: "0.82rem", width: "100%" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #333" }}>
            <th style={{ ...th, textAlign: "left" }}>Turning on…</th>
            {keys.map(k => <th key={k} style={th}>{metrics[k].label}</th>)}
          </tr>
        </thead>
        <tbody>
          {steps.map(s => (
            <tr key={s} style={{ borderBottom: "1px solid #222" }}>
              <td style={{ ...td, textAlign: "left" }}>{names[s]}
                <span style={muted}> ({effects[s].pairs.length} pair{effects[s].pairs.length > 1 ? "s" : ""})</span></td>
              {keys.map(k => {
                const e = effects[s][k]
                if (!e) return <td key={k} style={td}>–</td>
                const better = metrics[k].higher_better ? e.mean > 0 : e.mean < 0
                const sig = e.p < 0.05 && Math.abs(e.mean) > 1e-9
                return (
                  <td key={k} style={{ ...td, color: sig ? (better ? GOOD : BAD) : "#aaa" }}
                    title={`95% CI ${fmt(e.lo, k)} – ${fmt(e.hi, k)}, p = ${e.p.toFixed(3)}`}>
                    {e.mean > 0 ? "+" : ""}{fmt(e.mean, k)}{sig && "*"}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ ...muted, margin: "8px 0 0" }}>
        Average change from switching each step on, over every pair of configurations that differ only in that step,
        so each number is measured with the other steps both on and off. * p &lt; 0.05.
      </p>
    </div>
  )
}

function Explorer({ run }) {
  const [qid, setQid] = useState(run.items[0]?.id)
  const item = run.items.find(i => i.id === qid) || run.items[0]
  if (!item) return null
  const relevant = new Set(item.relevant)
  const recs = run.configs.map(c => ({ c, r: run.records.find(r => r.item === item.id && r.config === c.id) })).filter(x => x.r)
  const missed = new Set(run.summary.never_retrieved)
  return (
    <div>
      <select value={item.id} onChange={e => setQid(+e.target.value)} style={{ ...input, width: "100%", marginBottom: 10 }}>
        {run.items.map(i => (
          <option key={i.id} value={i.id}>
            {missed.has(i.id) ? "✗ " : ""}Q{i.id + 1}. {i.question}
          </option>
        ))}
      </select>
      <p style={{ fontSize: "0.82rem", margin: "0 0 10px", lineHeight: 1.6 }}>
        <span style={muted}>Reference: </span>{item.answer}<br />
        <span style={muted}>Evidence ({item.pages}, chunks {item.relevant.map(r => `#${r}`).join(", ")}): </span>
        <span style={muted}>{item.evidence}</span>
      </p>
      {recs.map(({ c, r }) => (
        <details key={c.id} style={{ borderTop: "1px solid #2a2a2a", padding: "8px 0" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.82rem" }}>
            {c.name}
            {r.retrieved && (
              <span style={muted}> · retrieved {r.retrieved.map(i => relevant.has(i) ? `[#${i}]` : `#${i}`).join(" ")}
                {r.rank ? ` · evidence at rank ${r.rank}` : " · evidence missed"}</span>
            )}
            {r.correctness != null && <span style={muted}> · correctness {r.correctness.toFixed(2)}</span>}
            <span style={muted}> · {fmtMs(r.total_ms)}</span>
          </summary>
          <div style={{ fontSize: "0.82rem", lineHeight: 1.6, padding: "6px 0 0 14px" }}>
            {r.query && <p style={{ margin: "0 0 6px" }}><span style={muted}>Rewritten query: </span>{r.query}</p>}
            <p style={{ margin: "0 0 6px", whiteSpace: "pre-wrap" }}>{r.answer}</p>
            {r.judge_reason && <p style={{ ...muted, margin: 0 }}>Judge: {r.judge_reason}</p>}
            {r.faithfulness_reason && <p style={{ ...muted, margin: 0 }}>Faithfulness ({r.faithfulness?.toFixed(2)}): {r.faithfulness_reason}</p>}
          </div>
        </details>
      ))}
      <p style={{ ...muted, margin: "8px 0 0" }}>[#n] marks a chunk that contains the evidence. ✗ marks questions no configuration retrieved evidence for.</p>
    </div>
  )
}

function Results({ runs, run, onPick, metrics }) {
  if (!runs.length) return null
  const download = () => {
    const blob = new Blob([JSON.stringify(run, null, 1)], { type: "application/json" })
    const a = document.createElement("a")
    a.href = URL.createObjectURL(blob)
    a.download = `rag-eval-${run.id}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={card}>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: run ? 12 : 0 }}>
          <p style={{ ...label, margin: 0 }}>3 · RESULTS</p>
          <select value={run?.id || ""} onChange={e => onPick(e.target.value)} style={{ ...input, width: "auto" }}>
            {runs.map(r => (
              <option key={r.id} value={r.id}>
                {r.created_at.replace("T", " ")} · {r.doc} · {r.questions}q × {r.configs}{r.status !== "complete" ? ` (${r.status})` : ""}
              </option>
            ))}
          </select>
          {run && <>
            <button onClick={download} style={{ ...button(true, "#333"), color: "#e0e0e0", padding: "4px 10px" }}>JSON</button>
            <a href={`${API}/eval/runs/${run.id}/report.md`} style={{ ...button(true, "#333"), color: "#e0e0e0", padding: "4px 10px", textDecoration: "none" }}>Markdown report</a>
          </>}
        </div>
        {run && (
          <p style={{ ...muted, margin: 0 }}>
            {run.items.length} questions × {run.configs.length} configurations in {fmtMs(run.wall_seconds * 1000)} ·
            {" "}{run.llm_calls} LLM calls, {run.cached_calls} reused · answers by {run.models.llm}
            {run.models.judge ? `, judged by ${run.models.judge}` : ", no judge"}
          </p>
        )}
      </div>
      {run && <>
        <div style={card}>
          <p style={label}>SCORES</p>
          <Leaderboard run={run} metrics={metrics} />
        </div>
        <div style={{ ...card, display: "flex", gap: 24, flexWrap: "wrap" }}>
          <MetricBars run={run} metrics={metrics} />
          <LatencyBars run={run} />
        </div>
        <div style={{ ...card, display: "flex", gap: 24, flexWrap: "wrap" }}>
          <Tradeoff run={run} metrics={metrics} />
          <div style={{ flex: "1 1 420px", minWidth: 0 }}>
            <p style={{ ...label, color: ACCENT }}>WHAT EACH ADVANCED STEP ADDS</p>
            <Effects run={run} metrics={metrics} />
          </div>
        </div>
        <div style={card}>
          <p style={label}>QUESTION BY QUESTION</p>
          <Explorer key={run.id} run={run} />
        </div>
      </>}
    </div>
  )
}

export default function Evaluation({ doc }) {
  const [meta, setMeta] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [job, setJob] = useState(null)
  const [runs, setRuns] = useState([])
  const [run, setRun] = useState(null)
  const [error, setError] = useState("")
  const busy = job?.status === "running"
  const shownRun = useRef(null)
  useEffect(() => { shownRun.current = run?.id }, [run])

  const pickRun = useCallback(id => {
    if (!id) return setRun(null)
    getJson(`/eval/runs/${id}`).then(setRun).catch(e => setError(e.message))
  }, [])

  // Reload the test set and the list of runs; latest=true also shows the newest run
  const refresh = useCallback((latest = false) =>
    Promise.all([getJson("/eval/dataset"), getJson("/eval/runs")])
      .then(([ds, rs]) => {
        setDataset(ds)
        setRuns(rs.runs)
        if (rs.runs.length && (latest || !shownRun.current)) pickRun(rs.runs[0].id)
      })
      .catch(e => setError(e.message)), [pickRun])

  useEffect(() => {
    getJson("/eval/configs").then(setMeta).catch(e => setError(e.message))
    getJson("/eval/job").then(setJob).catch(() => {})
    refresh()
  }, [refresh, doc?.filename, doc?.chunks])

  // Poll while a job runs, then reload what it produced
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(async () => {
      try {
        const j = await getJson("/eval/job")
        setJob(j)
        if (j.status !== "running") refresh(j.kind === "run")
      } catch (e) {
        setError(e.message)
      }
    }, 1000)
    return () => clearInterval(timer)
  }, [busy, refresh])

  async function start(path, body) {
    setError("")
    try {
      await post(path, body)
      setJob(await getJson("/eval/job"))
    } catch (e) {
      setError(e.message)
    }
  }

  async function remove(id) {
    try {
      await getJson(`/eval/dataset/${id}`, { method: "DELETE" })
      refresh()
    } catch (e) {
      setError(e.message)
    }
  }

  if (!doc) return <p style={muted}>Upload a PDF to evaluate retrieval and answers on it.</p>
  if (!meta) return <p style={muted}>{error || "Loading…"}</p>
  const metrics = Object.fromEntries(meta.metrics.map(m => [m.key, m]))

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <p style={{ ...muted, margin: 0, lineHeight: 1.6 }}>
        Measure what each part of the pipeline is worth on this PDF: retrieval quality (did the right chunk come back,
        and how high), answer quality graded by an LLM judge, and latency, for no RAG, basic RAG, exact search and every
        combination of the advanced steps. A full run makes several hundred LLM calls; on a laptop expect tens of minutes.
      </p>
      <TestSet dataset={dataset} busy={busy} onGenerate={n => start("/eval/dataset", { n })} onDelete={remove} />
      <RunPanel meta={meta} dataset={dataset} busy={busy} onRun={body => start("/eval/run", body)} />
      <Progress job={job} onCancel={() => post("/eval/cancel")} />
      {error && <p style={{ color: BAD, margin: 0 }}>{error}</p>}
      <Results runs={runs} run={run} onPick={pickRun} metrics={metrics} />
    </div>
  )
}
