import { useState, useRef, useEffect } from "react"
import { API, readError } from "./ui"
import Evaluation from "./Evaluation"

// The four ways a question gets answered. `tone` is the CSS color token for that method.
const METHODS = {
  plain: { name: "No retrieval", tone: "var(--none)" },
  basic: { name: "Basic RAG", tone: "var(--basic)" },
  advanced: { name: "Advanced RAG", tone: "var(--advanced)" },
  agent: { name: "Agent", tone: "var(--agent)" },
}

const STEPS = [
  { key: "rewrite", name: "Query rewriting", help: "The model turns your question into a search query with key terms and synonyms" },
  { key: "hybrid", name: "Hybrid search", help: "Keyword search (BM25) runs next to vector search and the two rankings are merged" },
  { key: "rerank", name: "Reranking", help: "The model reorders the top 15 passages by usefulness and keeps 5" },
]

const TOOL_NAMES = { search: "Hybrid search", keyword: "Keyword search", read_neighbors: "Read nearby passages", read_page: "Read page" }

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

// Read a newline-delimited JSON stream, calling onEvent for each object
async function readEvents(res, onEvent) {
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop()
    lines.filter(Boolean).forEach(l => onEvent(JSON.parse(l)))
  }
  if (buf.trim()) onEvent(JSON.parse(buf))
}

function describeEvent(e) {
  switch (e.type) {
    case "route": return [`Chose the ${e.route} route`, e.reason]
    case "tool": return [`${TOOL_NAMES[e.tool] || e.tool} for "${e.input}"`,
      e.error ? `Failed: ${e.error}`
        : `Found ${e.found.length} passages (${e.pages.map(pageName).join(", ")})${e.new && e.new.length < e.found.length ? `, ${e.new.length} of them new` : ""}`]
    case "think": return ["Decided it had enough evidence", e.thought]
    case "map": return [`Summarized ${e.pages.join(", ")}`, e.summary]
    case "escalate": return ["Switched to the research route", e.reason]
    case "correct": return ["Searched again for unsupported claims", `${e.claims.length} claim${e.claims.length === 1 ? "" : "s"}, found ${e.pages.join(", ")}`]
    case "check": return [e.refused ? "Self-check: the answer declined" : e.grounded ? "Self-check passed" : "Self-check found unsupported claims",
      e.claims?.length ? `${e.claims.filter(c => c.supported).length} of ${e.claims.length} claims are backed by a cited passage` : ""]
    default: return [e.type, ""]
  }
}

// Answer text with [n] citations turned into buttons that open the passage
function Prose({ text, onCite, focus }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <div className="prose">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/)
        if (!m || !onCite) return part
        const n = +m[1]
        return (
          <button key={i} type="button" className="cite" aria-pressed={focus === n}
            aria-label={`Show source ${n}`} onClick={() => onCite(n)}>{n}</button>
        )
      })}
    </div>
  )
}

function Answer({ method, how, text, loading, onCite, focus, verdict }) {
  const { name, tone } = METHODS[method]
  return (
    <section className="answer" style={{ "--tone": tone }} aria-busy={loading && !text}>
      <h2>{name}</h2>
      <p className="how">{how}</p>
      {text
        ? <Prose text={text} onCite={onCite} focus={focus} />
        : loading && <div className="wait" aria-label="Writing the answer"><span /><span /><span /></div>}
      {verdict && <p className="verdict" data-ok={verdict.ok}>{verdict.text}</p>}
    </section>
  )
}

const pageName = pages => pages.replace("pp.", "Pages").replace("p.", "Page")

function Passage({ n, pages, text, tag, focused }) {
  const ref = useRef()
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" })
  }, [focused])
  return (
    <details className="passage" ref={ref} open={focused || undefined} data-focus={focused}>
      <summary>
        <span className="num">{n}</span>
        <span>{pageName(pages)}</span>
        {tag && <span className="tag">{tag}</span>}
      </summary>
      <p>{text}</p>
    </details>
  )
}

function BasicPane({ data, focus }) {
  const exactIds = new Set(data.exact.hits.map(h => h.id))
  return (
    <div className="pane" style={{ "--tone": METHODS.basic.tone }}>
      <p className="note">The 5 passages closest to your question in the vector index. Basic RAG answers from these alone.</p>
      {data.hnsw.hits.map((h, i) => (
        <Passage key={h.id} n={i + 1} pages={h.pages} text={h.text} focused={focus === i + 1}
          tag={`similarity ${(1 - h.distance).toFixed(3)}${exactIds.has(h.id) ? "" : ", missed by exact search"}`} />
      ))}
    </div>
  )
}

function RankList({ title, hits, score }) {
  if (!hits) return null
  return (
    <div>
      <h3>{title}</h3>
      <ol>
        {hits.slice(0, 8).map(h => <li key={h.id}>{pageName(h.pages)} <span>{score(h)}</span></li>)}
      </ol>
    </div>
  )
}

function AdvancedPane({ adv, question, focus }) {
  const changed = adv.final.filter(h => h.new).length
  const t = adv.timings
  const times = [t.rewrite_ms !== undefined && `rewriting ${t.rewrite_ms} ms`, `search ${t.search_ms} ms`,
    t.rerank_ms !== undefined && `reranking ${t.rerank_ms} ms`].filter(Boolean).join(", ")
  return (
    <div className="pane" style={{ "--tone": METHODS.advanced.tone }}>
      {adv.query !== question && <p>Searched for: <em>{adv.query}</em></p>}
      <div className="ranks">
        <RankList title="Vector search" hits={adv.vector} score={h => (1 - h.distance).toFixed(3)} />
        <RankList title="Keyword search" hits={adv.keyword} score={h => h.score.toFixed(2)} />
        <RankList title="Merged ranking" hits={adv.fused} score={h => h.score.toFixed(4)} />
      </div>
      <p className="note">
        {changed} of the {adv.final.length} passages below differ from basic RAG. Time spent: {times}.
      </p>
      {adv.final.map((h, i) => (
        <Passage key={h.id} n={i + 1} pages={h.pages} text={h.text} focused={focus === i + 1}
          tag={h.new ? "not used by basic RAG" : ""} />
      ))}
    </div>
  )
}

function AgentPane({ events, result, focus }) {
  return (
    <div className="pane" style={{ "--tone": METHODS.agent.tone }}>
      <ol className="trace">
        {events.filter(e => e.type !== "answer").map((e, i) => {
          const [head, detail] = describeEvent(e)
          return <li key={i}><b>{head}</b>{detail && <span>{detail}</span>}</li>
        })}
      </ol>
      {result?.check?.claims?.filter(c => !c.supported).map((c, i) => (
        <p key={i} className="unsupported">Not backed by a source: {c.claim}</p>
      ))}
      {result && (
        <p className="note">
          {result.llm_calls} model calls in {(result.wall_ms / 1000).toFixed(1)} s.
          {result.citations.length > 0 && " The agent answered from these passages:"}
        </p>
      )}
      {result?.citations.map(c => (
        <Passage key={c.id} n={c.n} pages={c.pages} text={c.text} focused={focus === c.n} />
      ))}
    </div>
  )
}

function SpeedPane({ data }) {
  const { hnsw, exact, recall, embed_ms } = data
  const saved = exact.distance_calls ? 1 - hnsw.distance_calls / exact.distance_calls : 0
  return (
    <div className="pane">
      <table className="speed">
        <thead><tr><th></th><th>HNSW index</th><th>Checking every passage</th></tr></thead>
        <tbody>
          <tr><th>Search time</th><td>{hnsw.micros.toFixed(0)} µs</td><td>{exact.micros.toFixed(0)} µs</td></tr>
          <tr><th>Comparisons made</th><td>{hnsw.distance_calls}</td><td>{exact.distance_calls}</td></tr>
          <tr><th>Same top 5</th><td>{Math.round(recall * 100)}%</td><td>100%, by definition</td></tr>
        </tbody>
      </table>
      <p className="note">
        {saved > 0
          ? `The index made ${Math.round(saved * 100)}% fewer comparisons than checking all ${exact.index_size} passages.`
          : `With only ${exact.index_size} passages, checking every one costs about the same as the index.`}
        {" "}The gap widens as documents grow to thousands of passages. Turning your question into a vector took {embed_ms} ms.
      </p>
    </div>
  )
}

function Evidence({ retrieval, question, agentEvents, agentResult, tab, setTab, focus }) {
  const tabs = [
    agentEvents.length > 0 && ["agent", "How the agent worked"],
    retrieval?.advanced && ["advanced", "Advanced pipeline"],
    retrieval && ["basic", "Basic RAG sources"],
    retrieval && ["speed", "Search speed"],
  ].filter(Boolean)
  if (!tabs.length) return null
  const current = tabs.some(([k]) => k === tab) ? tab : tabs[0][0]
  const at = k => (focus?.method === k ? focus.n : null)
  return (
    <section className="evidence" aria-label="Evidence">
      <div className="tabs" role="tablist">
        {tabs.map(([k, name]) => (
          <button key={k} type="button" role="tab" aria-selected={current === k}
            style={{ "--tone": METHODS[k]?.tone }} onClick={() => setTab(k)}>{name}</button>
        ))}
      </div>
      {current === "agent" && <AgentPane events={agentEvents} result={agentResult} focus={at("agent")} />}
      {current === "advanced" && <AdvancedPane adv={retrieval.advanced} question={question} focus={at("advanced")} />}
      {current === "basic" && <BasicPane data={retrieval} focus={at("basic")} />}
      {current === "speed" && <SpeedPane data={retrieval} />}
    </section>
  )
}

export default function App() {
  const [doc, setDoc] = useState(null)
  const [model, setModel] = useState("the model")
  const [uploading, setUploading] = useState(false)
  const [over, setOver] = useState(false)
  const [question, setQuestion] = useState("")
  const [asked, setAsked] = useState("")
  const [loading, setLoading] = useState(false)
  const [plain, setPlain] = useState("")
  const [rag, setRag] = useState("")
  const [advanced, setAdvanced] = useState("")
  const [steps, setSteps] = useState({ rewrite: true, hybrid: true, rerank: true })
  const [useAgent, setUseAgent] = useState(true)
  const [ran, setRan] = useState({ advanced: false, agent: false })
  const [retrieval, setRetrieval] = useState(null)
  const [agentEvents, setAgentEvents] = useState([])
  const [agentResult, setAgentResult] = useState(null)
  const [tab, setTab] = useState(null)
  const [focus, setFocus] = useState(null)
  const [error, setError] = useState("")
  const [view, setView] = useState("compare")
  const fileRef = useRef()

  // Pick up the document that is already indexed after a page refresh
  useEffect(() => {
    fetch(`${API}/status`).then(r => r.json())
      .then(s => s.indexed && setDoc(s))
      .catch(() => setError(`Can't reach the backend at ${API || "this address"}. Start it with "python app.py" and reload.`))
    fetch(`${API}/health`).then(r => r.json()).then(h => h.llm && setModel(h.llm)).catch(() => {})
  }, [])

  function clearResults() {
    setPlain(""); setRag(""); setAdvanced(""); setRetrieval(null)
    setAgentEvents([]); setAgentResult(null); setTab(null); setFocus(null)
  }

  async function upload(file) {
    if (!file || uploading) return
    setUploading(true)
    setError("")
    clearResults()
    try {
      const form = new FormData()
      form.append("file", file)
      const res = await fetch(`${API}/upload`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await readError(res))
      setDoc(await res.json())
    } catch (err) {
      setError(`Couldn't index that file: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  function pickFile(e) {
    const file = e.target.files[0]
    e.target.value = ""  // allow picking the same file again
    upload(file)
  }

  async function handleAsk(e) {
    e?.preventDefault()
    if (!question.trim() || !doc || loading) return
    setLoading(true)
    setError("")
    clearResults()
    setAsked(question)
    const useAdvanced = Object.values(steps).some(Boolean)
    setRan({ advanced: useAdvanced, agent: useAgent })

    const post = (path, body) => fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...body })
    })

    try {
      // The baseline and the agent don't wait for retrieval, so start them right away
      const plainDone = post("/ask", { mode: "plain" }).then(res => streamInto(res, setPlain))
      const agentDone = useAgent
        ? post("/agent", {}).then(async res => {
          if (!res.ok) throw new Error(await readError(res))
          await readEvents(res, e => {
            if (e.type === "error") throw new Error(`The agent stopped: ${e.error}`)
            setAgentEvents(prev => [...prev, e])
            if (e.type === "answer") setAgentResult(e)
          })
        })
        : null

      const res = await post("/retrieve", { advanced: useAdvanced ? steps : undefined })
      if (!res.ok) throw new Error(await readError(res))
      const r = await res.json()
      setRetrieval(r)

      const ragDone = post("/ask", { mode: "rag", chunk_ids: r.hnsw.hits.map(h => h.id) })
        .then(res => streamInto(res, setRag))
      const advDone = r.advanced
        ? post("/ask", { mode: "rag", chunk_ids: r.advanced.final.map(h => h.id) })
          .then(res => streamInto(res, setAdvanced))
        : null
      await Promise.all([plainDone, ragDone, advDone, agentDone])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const cite = method => n => { setTab(method); setFocus({ method, n }) }
  const at = method => (focus?.method === method ? focus.n : null)
  const showAnswers = loading || plain || rag || advanced || agentResult
  const cols = 2 + (ran.advanced ? 1 : 0) + (ran.agent ? 1 : 0)
  const lastEvent = agentEvents[agentEvents.length - 1]

  return (
    <div className="shell">
      <header className="top">
        <span className="brand">rag-engine</span>
        <nav className="views" aria-label="Views">
          {[["compare", "Compare answers"], ["evaluate", "Evaluate"]].map(([key, name]) => (
            <button key={key} type="button" aria-current={view === key ? "page" : undefined}
              onClick={() => setView(key)}>{name}</button>
          ))}
        </nav>
        <input ref={fileRef} type="file" accept=".pdf" hidden onChange={pickFile} />
        {doc && (
          <div className="doc">
            {uploading
              ? <span>Indexing the new PDF…</span>
              : <>
                <strong title={doc.filename}>{doc.filename}</strong>
                <span>{doc.pages} pages, {doc.chunks} passages</span>
                <button type="button" className="link" onClick={() => fileRef.current.click()}>Replace PDF</button>
              </>}
          </div>
        )}
      </header>

      {error && <p className="error" role="alert">{error}</p>}

      {!doc && (
        <div className="drop" data-over={over}
          onDragOver={e => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files[0]) }}>
          <h1>Ask a PDF one question.<br />Read four answers.</h1>
          <p>
            The same model answers with no document, with basic retrieval, with an advanced pipeline and as an
            agent that checks its own citations. You see every passage each one used.
          </p>
          <button type="button" className="primary" disabled={uploading} onClick={() => fileRef.current.click()}>
            {uploading ? "Indexing the PDF…" : "Choose a PDF"}
          </button>
          <p className="note" style={{ margin: "16px auto 0" }}>or drop one here. Text PDFs up to 50 MB.</p>
        </div>
      )}

      {doc && view === "evaluate" && <div style={{ marginTop: 28, display: "flex", flexDirection: "column", gap: 20 }}><Evaluation doc={doc} /></div>}

      {doc && view === "compare" && <>
        <form className="ask" onSubmit={handleAsk}>
          <h1>What do you want to know?</h1>
          <div className="askrow">
            <input value={question} onChange={e => setQuestion(e.target.value)} disabled={loading}
              aria-label="Your question" placeholder="Ask something only this document can answer" />
            <button type="submit" className="primary" disabled={loading || !question.trim()}>
              {loading ? "Comparing…" : "Compare answers"}
            </button>
          </div>
          <fieldset className="methods" disabled={loading}>
            <legend>Advanced RAG uses</legend>
            {STEPS.map(s => (
              <label key={s.key} className="pill" title={s.help} style={{ "--tone": METHODS.advanced.tone }}>
                <input type="checkbox" checked={steps[s.key]}
                  onChange={e => setSteps({ ...steps, [s.key]: e.target.checked })} />
                {s.name}
              </label>
            ))}
            <span className="gap" />
            <label className="pill" style={{ "--tone": METHODS.agent.tone }}
              title="Picks a route for the question, searches with tools until it has enough evidence, then checks every claim against its sources">
              <input type="checkbox" checked={useAgent} onChange={e => setUseAgent(e.target.checked)} />
              Run the agent
            </label>
          </fieldset>
        </form>

        {showAnswers && (
          <div className="answers" style={{ "--cols": cols }}>
            <Answer method="plain" how={`${model} alone, without the document`} text={plain} loading={loading} />
            <Answer method="basic" how="The 5 closest passages from the vector index" text={rag} loading={loading}
              onCite={cite("basic")} focus={at("basic")} />
            {ran.advanced && (
              <Answer method="advanced" text={advanced} loading={loading}
                how={STEPS.filter(s => steps[s.key]).map(s => s.name).join(", ")}
                onCite={cite("advanced")} focus={at("advanced")} />
            )}
            {ran.agent && (
              <Answer method="agent" text={agentResult?.answer} loading={loading}
                how={agentResult ? `Took the ${agentResult.route} route` : lastEvent ? `${describeEvent(lastEvent)[0]}…` : "Plans its own searches"}
                onCite={cite("agent")} focus={at("agent")}
                verdict={agentResult?.check && (agentResult.check.refused
                  ? { ok: null, text: "Declined: the passages don't contain the answer" }
                  : agentResult.check.grounded
                    ? { ok: true, text: "Every claim is backed by a cited passage" }
                    : { ok: false, text: "Some claims aren't backed by a source" })} />
            )}
          </div>
        )}

        <Evidence retrieval={retrieval} question={asked} agentEvents={agentEvents} agentResult={agentResult}
          tab={tab} setTab={k => { setTab(k); setFocus(null) }} focus={focus} />
      </>}
    </div>
  )
}
