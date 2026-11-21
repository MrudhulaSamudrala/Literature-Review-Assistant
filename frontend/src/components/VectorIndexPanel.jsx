export default function VectorIndexPanel({
  result,
  indexing,
  indexResult,
  indexError,
  onIndex,
  graphIndexing,
  graphIndexResult,
  graphIndexError,
  onGraphIndex,
}) {
  if (!result) {
    return null;
  }
  const vectorIndexed = indexResult?.status === "indexed";
  const graphIndexed = graphIndexResult?.status === "indexed";
  return (
    <section className="panel">
      <h2>Index status</h2>
      <dl className="stats">
        <dt>Uploaded</dt>
        <dd>yes</dd>
        <dt>Ingested</dt>
        <dd>{result.status}</dd>
        <dt>Vector Indexed</dt>
        <dd>{vectorIndexed ? "yes" : "no"}</dd>
        <dt>Graph Indexed</dt>
        <dd>{graphIndexed ? "yes" : "no"}</dd>
      </dl>
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", flexWrap: "wrap" }}>
        <button type="button" onClick={onIndex} disabled={indexing}>
          {indexing ? "Indexing Vector..." : "Index VectorRAG"}
        </button>
        <button type="button" onClick={onGraphIndex} disabled={graphIndexing}>
          {graphIndexing ? "Indexing Graph..." : "Index GraphRAG"}
        </button>
      </div>
      {indexError ? <p className="error">Vector error: {indexError}</p> : null}
      {graphIndexError ? <p className="error">Graph error: {graphIndexError}</p> : null}
      {vectorIndexed ? (
        <p className="muted">
          Vector: {indexResult.paper_id} indexed {indexResult.chunk_count} chunks with {indexResult.embedding_model}.
          Newly embedded: {indexResult.embedded_chunk_count}.
        </p>
      ) : null}
      {graphIndexed ? (
        <p className="muted">
          Graph: {graphIndexResult.paper_id} indexed {graphIndexResult.chunk_count} chunks into Neo4j ({graphIndexResult.entity_count} entities, {graphIndexResult.relationship_count} relationships).
        </p>
      ) : null}
    </section>
  );
}
