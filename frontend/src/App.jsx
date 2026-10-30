import { useState, useRef, useEffect } from "react"

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000"

const card = {
  background: "#1a1a1a", border: "1px solid #2a2a2a", borderRadius: 8,
  padding: "16px 20px"
}
const label = { color: "#4fc3f7", margin: "0 0 10px", fontSize: "0.8rem", letterSpacing: 1 }
const muted = { color: "#888", fontSize: "0.8rem" }

async function readError(res) {
  try { return (await res.json()).error || res.statusText } catch { return res.statusText }
}

// Stream a text/plain response into setText
async function streamInto(res, setText) {
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let result = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    result += decoder.decode(value, { stream: true })
    setText(result)
  }
}

function Answer({ title, subtitle, text, loading, accent }) {
  return (
    <div style={{ ...card, flex: "1 1 300px", minWidth: 0, borderTop: `3px solid ${accent}` }}>
      <p style={{ ...label, color: accent }}>{title}</p>
      <p style={{ ...muted, margin: "-6px 0 12px" }}>{subtitle}</p>
      <div style={{ lineHeight: 1.7, whiteSpace: "pre-wrap", fontSize: "0.92rem" }}>
        {text || (loading ? "..." : "")}
      </div>
    </div>
  )
}

function Retrieval({ data }) {
  const { hnsw, exact, recall, embed_ms } = data
  const exactIds = new Set(exact.hits.map(h => h.id))
  const saved = exact.distance_calls ? 1 - hnsw.distance_calls / exact.distance_calls : 0
  const stat = (name, h, e) => (
    <tr>
      <td style={{ ...muted, padding: "4px 12px 4px 0" }}>{name}</td>
      <td style={{ padding: "4px 12px" }}>{h}</td>
      <td style={{ padding: "4px 12px" }}>{e}</td>
    </tr>
  )
  return (
    <div style={{ ...card, width: "100%" }}>
      <p style={label}>BASIC RETRIEVAL — HNSW VS BRUTE FORCE</p>
      <table style={{ fontSize: "0.85rem", borderCollapse: "collapse", marginBottom: 12 }}>
        <thead>
          <tr style={muted}>
            <th></th>
            <th style={{ textAlign: "left", padding: "4px 12px" }}>HNSW</th>
            <th style={{ textAlign: "left", padding: "4px 12px" }}>Brute force</th>
          </tr>
        </thead>
        <tbody>
          {stat("Search time", `${hnsw.micros.toFixed(0)} µs`, `${exact.micros.toFixed(0)} µs`)}
          {stat("Distance computations", hnsw.distance_calls, exact.distance_calls)}
          {stat("Same top results", `${Math.round(recall * 100)}%`, "100% (ground truth)")}
        </tbody>
      </table>
      <p style={{ ...muted, margin: "0 0 14px" }}>
        {saved > 0
          ? `HNSW did ${Math.round(saved * 100)}% fewer distance computations than scanning all ${exact.index_size} chunks.`
          : `With only ${exact.index_size} chunks, scanning everything is as cheap as HNSW.`}
        {" "}HNSW pays off as the index grows to thousands of chunks.
        Question embedding took {embed_ms} ms.
      </p>
      {hnsw.hits.map((h, i) => (
        <details key={h.id} style={{ borderTop: "1px solid #2a2a2a", padding: "8px 0" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.85rem" }}>
            [{i + 1}] {h.pages} · similarity {(1 - h.distance).toFixed(3)}
            {!exactIds.has(h.id) && <span style={{ color: "#ffa726" }}> · not in exact top {exact.hits.length}</span>}
          </summary>
          <p style={{ ...muted, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{h.text}</p>
        </details>
      ))}
    </div>
  )
}

const STEPS = [
  { key: "rewrite", name: "Query rewriting", help: "LLM turns the question into a search query with key terms and synonyms" },
  { key: "hybrid", name: "Hybrid search", help: "BM25 keyword search alongside HNSW, merged with reciprocal rank fusion" },
  { key: "rerank", name: "Reranking", help: `LLM reorders the top 15 candidates by usefulness, keeps 5` },
]

function HitList({ title, hits, score }) {
  if (!hits) return null
  return (
    <div style={{ flex: "1 1 180px", minWidth: 0 }}>
      <p style={{ ...muted, margin: "0 0 6px" }}>{title}</p>
      <ol style={{ margin: 0, paddingLeft: 20, fontSize: "0.8rem", lineHeight: 1.6 }}>
        {hits.slice(0, 8).map(h => (
          <li key={h.id}>#{h.id} {h.pages}{score && <span style={muted}> · {score(h)}</span>}</li>
        ))}
      </ol>
    </div>
  )
}

function Pipeline({ adv, question }) {
  const changed = adv.final.filter(h => h.new).length
  const t = adv.timings
  return (
    <div style={{ ...card, width: "100%" }}>
      <p style={{ ...label, color: "#ab47bc" }}>ADVANCED PIPELINE — WHAT EACH STEP DID</p>
      {adv.query !== question && (
        <p style={{ fontSize: "0.85rem", margin: "0 0 12px" }}>
          <span style={muted}>Rewritten query: </span>{adv.query}
        </p>
      )}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
        <HitList title="HNSW (vector)" hits={adv.vector} score={h => (1 - h.distance).toFixed(3)} />
        <HitList title="BM25 (keyword)" hits={adv.keyword} score={h => h.score.toFixed(2)} />
        <HitList title="Fused (RRF)" hits={adv.fused} score={h => h.score.toFixed(4)} />
      </div>
      <p style={{ ...muted, margin: "0 0 14px" }}>
        {changed} of {adv.final.length} final chunks differ from basic RAG.
        {t.rewrite_ms !== undefined && ` Rewrite ${t.rewrite_ms} ms.`}
        {` Search ${t.search_ms} ms.`}
        {t.rerank_ms !== undefined && ` Rerank ${t.rerank_ms} ms.`}
      </p>
      {adv.final.map((h, i) => (
        <details key={h.id} style={{ borderTop: "1px solid #2a2a2a", padding: "8px 0" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.85rem" }}>
            [{i + 1}] #{h.id} {h.pages}
            {h.new && <span style={{ color: "#ab47bc" }}> · not used by basic RAG</span>}
          </summary>
          <p style={{ ...muted, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{h.text}</p>
        </details>
      ))}
    </div>
  )
}

export default function App() {
  const [doc, setDoc] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [plain, setPlain] = useState("")
  const [rag, setRag] = useState("")
  const [advanced, setAdvanced] = useState("")
  const [steps, setSteps] = useState({ rewrite: true, hybrid: true, rerank: true })
  const [retrieval, setRetrieval] = useState(null)
  const [error, setError] = useState("")
  const fileRef = useRef()

  // Pick up the document that is already indexed after a page refresh
  useEffect(() => {
    fetch(`${API}/status`).then(r => r.json())
      .then(s => s.indexed && setDoc(s))
      .catch(() => setError(`Can't reach the backend at ${API}`))
  }, [])

  async function handleUpload(e) {
    const file = e.target.files[0]
    e.target.value = ""  // allow picking the same file again
    if (!file) return
    setUploading(true)
    setError("")
    setPlain(""); setRag(""); setAdvanced(""); setRetrieval(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const res = await fetch(`${API}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await readError(res))
      setDoc(await res.json())
    } catch (err) {
      setError(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  async function handleAsk() {
    if (!question.trim() || !doc || loading) return
    setLoading(true)
    setError("")
    setPlain(""); setRag(""); setAdvanced(""); setRetrieval(null)
    const useAdvanced = Object.values(steps).some(Boolean)

    const ask = body => fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...body })
    })

    try {
      // The baseline doesn't need retrieval, so start it right away
      const plainDone = ask({ mode: "plain" }).then(res => streamInto(res, setPlain))

      const res = await fetch(`${API}/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, advanced: useAdvanced ? steps : undefined })
      })
      if (!res.ok) throw new Error(await readError(res))
      const r = await res.json()
      setRetrieval(r)

      const ragDone = ask({ mode: "rag", chunk_ids: r.hnsw.hits.map(h => h.id) })
        .then(res => streamInto(res, setRag))
      const advDone = r.advanced
        ? ask({ mode: "rag", chunk_ids: r.advanced.final.map(h => h.id) })
          .then(res => streamInto(res, setAdvanced))
        : null
      await Promise.all([plainDone, ragDone, advDone])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f0f", color: "#e0e0e0",
      fontFamily: "monospace", display: "flex", flexDirection: "column",
      alignItems: "center", padding: "40px 16px"
    }}>
      <div style={{ width: "100%", maxWidth: 1200, display: "flex", flexDirection: "column", gap: 20 }}>
        <div>
          <h1 style={{ color: "#4fc3f7", fontSize: "1.6rem", margin: "0 0 4px" }}>
            RAG vs no RAG
          </h1>
          <p style={{ color: "#666", margin: 0, fontSize: "0.85rem" }}>
            Ask a question about your PDF and see llama3.2 answer with no retrieval, basic RAG and advanced RAG.
            HNSW vector search, fully local.
          </p>
        </div>

        {/* Upload */}
        <div style={{
          background: "#1a1a1a", border: "1px dashed #333", borderRadius: 8,
          padding: "20px 24px", textAlign: "center", cursor: "pointer"
        }} onClick={() => !uploading && fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".pdf"
            style={{ display: "none" }} onChange={handleUpload} />
          {uploading
            ? <p style={{ color: "#ffa726", margin: 0 }}>⏳ Processing PDF and building index...</p>
            : doc
              ? <p style={{ color: "#66bb6a", margin: 0 }}>
                  ✓ {doc.filename} — {doc.pages} pages, {doc.chunks} chunks indexed
                  <span style={muted}> (click to replace)</span>
                </p>
              : <p style={{ color: "#888", margin: 0 }}>Click to upload a PDF</p>
          }
        </div>

        {/* Question input */}
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAsk()}
            placeholder={doc ? "Ask something only your document knows..." : "Upload a PDF first"}
            disabled={!doc || loading}
            style={{
              flex: 1, minWidth: 0, padding: "10px 14px", borderRadius: 6, border: "1px solid #333",
              background: "#1a1a1a", color: "#e0e0e0", fontSize: "0.95rem", outline: "none"
            }}
          />
          <button onClick={handleAsk} disabled={!doc || loading || !question.trim()}
            style={{
              padding: "10px 20px", borderRadius: 6, border: "none",
              background: doc ? "#4fc3f7" : "#333",
              color: doc ? "#000" : "#666", cursor: doc ? "pointer" : "default",
              fontWeight: "bold", fontSize: "0.9rem"
            }}>
            {loading ? "..." : "Compare"}
          </button>
        </div>

        {/* Advanced pipeline switches */}
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", fontSize: "0.85rem" }}>
          {STEPS.map(s => (
            <label key={s.key} title={s.help} style={{ cursor: "pointer", color: steps[s.key] ? "#e0e0e0" : "#666" }}>
              <input type="checkbox" checked={steps[s.key]} disabled={loading}
                onChange={e => setSteps({ ...steps, [s.key]: e.target.checked })}
                style={{ marginRight: 6, accentColor: "#ab47bc" }} />
              {s.name}
            </label>
          ))}
          <span style={muted}>(advanced RAG; turn all off to compare basic RAG only)</span>
        </div>

        {error && <p style={{ color: "#ef5350", margin: 0 }}>{error}</p>}

        {/* Answers side by side */}
        {(plain || rag || advanced || loading) && (
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <Answer title="WITHOUT RAG" subtitle="llama3.2 on its own, no document"
              text={plain} loading={loading} accent="#888" />
            <Answer title="BASIC RAG" subtitle="llama3.2 + top 5 chunks from HNSW, cites [n]"
              text={rag} loading={loading} accent="#66bb6a" />
            {(advanced || (loading && Object.values(steps).some(Boolean))) && (
              <Answer title="ADVANCED RAG"
                subtitle={STEPS.filter(s => steps[s.key]).map(s => s.name.toLowerCase()).join(" + ")}
                text={advanced} loading={loading} accent="#ab47bc" />
            )}
          </div>
        )}

        {retrieval?.advanced && <Pipeline adv={retrieval.advanced} question={question} />}
        {retrieval && <Retrieval data={retrieval} />}
      </div>
    </div>
  )
}
