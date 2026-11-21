export default function SourceList({ sources }) {
  return (
    <section className="panel">
      <h2>Sources and citations</h2>
      {!sources?.length ? (
        <p className="muted">No citations yet. VectorRAG citations appear here after a question is answered.</p>
      ) : (
        <ul className="source-list">
          {sources.map((source, index) => (
            <li key={`${source.chunk_id ?? source.document_id ?? "source"}-${index}`}>
              <strong>{source.title ?? `Source ${index + 1}`}</strong>
              {source.paper_id ? <span> · {source.paper_id}</span> : null}
              {source.chunk_id ? <span> · {source.chunk_id}</span> : null}
              {source.page != null ? <span> · page {source.page}</span> : null}
              {source.source_filename ? <span> · {source.source_filename}</span> : null}
              {source.score != null ? <span> · score {source.score.toFixed(3)}</span> : null}
              {source.snippet ? <p>{source.snippet}</p> : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
