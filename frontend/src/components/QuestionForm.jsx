export default function QuestionForm({ question, onQuestionChange, onSubmit, loading, canAsk, hint, mode }) {
  const modeLabel =
    mode === "graph"
      ? "GraphRAG"
      : mode === "hybrid"
      ? "HybridRAG"
      : mode === "compare"
      ? "Compare All"
      : "VectorRAG";

  return (
    <section className="panel">
      <h2>Ask a research question</h2>
      <textarea
        rows={4}
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="What methods does this paper compare for literature chatbots?"
      />
      <button type="button" onClick={onSubmit} disabled={!canAsk || loading}>
        {loading ? "Asking..." : `Ask with ${modeLabel}`}
      </button>
      <p className="muted">{hint}</p>
    </section>
  );
}
