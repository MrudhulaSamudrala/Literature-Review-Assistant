export default function AnswerPanel({ result, error }) {
  if (error) {
    return (
      <section className="panel">
        <h2>Answer</h2>
        <p className="error">{error}</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel">
        <h2>Answer</h2>
        <p className="muted">
          No answer yet. Index a paper for VectorRAG, then ask a question.
        </p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Answer</h2>
      <p className="badge">{result.retrieval_method || result.mode || "VectorRAG"}</p>
      <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, margin: "0.8rem 0" }}>
        {result.answer}
      </div>
      {result.total_latency_ms != null ? (
        <p className="muted">
          Embedding {result.embedding_latency_ms} ms · retrieval {result.retrieval_latency_ms} ms ·
          generation {result.generation_latency_ms} ms · total {result.total_latency_ms} ms
        </p>
      ) : null}
    </section>
  );
}
