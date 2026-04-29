import { useState, useRef } from "react"

const API = "http://127.0.0.1:5000"

export default function App() {
  const [uploaded, setUploaded] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [loading, setLoading] = useState(false)
  const [chunks, setChunks] = useState(0)
  const [filename, setFilename] = useState("")
  const fileRef = useRef()

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    setFilename(file.name)
    setUploading(true)
    setUploaded(false)
    setAnswer("")

    const form = new FormData()
    form.append("file", file)

    const res = await fetch(`${API}/upload`, { method: "POST", body: form })
    const data = await res.json()
    setChunks(data.chunks)
    setUploading(false)
    setUploaded(true)
  }

  async function handleAsk() {
    if (!question.trim() || !uploaded) return
    setLoading(true)
    setAnswer("")

    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let result = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      result += decoder.decode(value)
      setAnswer(result)
    }

    setLoading(false)
  }

  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f0f", color: "#e0e0e0",
      fontFamily: "monospace", display: "flex", flexDirection: "column",
      alignItems: "center", padding: "40px 20px"
    }}>
      <h1 style={{ color: "#4fc3f7", fontSize: "1.6rem", marginBottom: 4 }}>
        Local Document Q&A
      </h1>
      <p style={{ color: "#666", marginBottom: 32, fontSize: "0.85rem" }}>
        Powered by HNSW vector search + llama3.2 — fully local, zero data leaves your machine
      </p>

      {/* Upload */}
      <div style={{
        background: "#1a1a1a", border: "1px dashed #333", borderRadius: 8,
        padding: "24px 32px", width: "100%", maxWidth: 640,
        textAlign: "center", marginBottom: 24, cursor: "pointer"
      }} onClick={() => fileRef.current.click()}>
        <input ref={fileRef} type="file" accept=".pdf"
          style={{ display: "none" }} onChange={handleUpload} />
        {uploading
          ? <p style={{ color: "#ffa726" }}>⏳ Processing PDF and building index...</p>
          : uploaded
            ? <p style={{ color: "#66bb6a" }}>✓ {filename} — {chunks} chunks indexed</p>
            : <p style={{ color: "#888" }}>Click to upload a PDF</p>
        }
      </div>

      {/* Question input */}
      <div style={{ width: "100%", maxWidth: 640, display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleAsk()}
          placeholder={uploaded ? "Ask anything about your document..." : "Upload a PDF first"}
          disabled={!uploaded || loading}
          style={{
            flex: 1, padding: "10px 14px", borderRadius: 6, border: "1px solid #333",
            background: "#1a1a1a", color: "#e0e0e0", fontSize: "0.95rem", outline: "none"
          }}
        />
        <button onClick={handleAsk} disabled={!uploaded || loading || !question.trim()}
          style={{
            padding: "10px 20px", borderRadius: 6, border: "none",
            background: uploaded ? "#4fc3f7" : "#333",
            color: uploaded ? "#000" : "#666", cursor: uploaded ? "pointer" : "default",
            fontWeight: "bold", fontSize: "0.9rem"
          }}>
          {loading ? "..." : "Ask"}
        </button>
      </div>

      {/* Answer */}
      {answer && (
        <div style={{
          width: "100%", maxWidth: 640, background: "#1a1a1a",
          border: "1px solid #2a2a2a", borderRadius: 8, padding: "20px 24px",
          lineHeight: 1.7, whiteSpace: "pre-wrap", fontSize: "0.95rem"
        }}>
          <p style={{ color: "#4fc3f7", margin: "0 0 10px", fontSize: "0.8rem" }}>ANSWER</p>
          {answer}
        </div>
      )}
    </div>
  )
}