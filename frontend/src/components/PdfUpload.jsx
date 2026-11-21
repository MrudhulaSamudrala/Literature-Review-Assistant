export default function PdfUpload({ file, onFileChange, onUpload, status, error, uploading, result }) {
  return (
    <section className="panel">
      <h2>Literature PDF</h2>
      <p className="muted">Upload a scientific paper. The file is extracted and split into semantic chunks.</p>
      <input
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
      />
      <button type="button" onClick={onUpload} disabled={!file || uploading}>
        {uploading ? "Uploading..." : "Upload PDF"}
      </button>
      {file ? <p className="muted">Selected: {file.name}</p> : null}
      {uploading ? <p className="muted">Uploading and extracting the PDF…</p> : null}
      {error ? <p className="error">{error}</p> : null}
      {status ? <p className="muted">{status}</p> : null}
      {result ? (
        <dl className="stats">
          <dt>Paper ID</dt>
          <dd>{result.paper_id}</dd>
          <dt>File</dt>
          <dd>{result.filename}</dd>
          <dt>Status</dt>
          <dd>{result.status}</dd>
          <dt>Pages</dt>
          <dd>{result.page_count}</dd>
          <dt>Chunks</dt>
          <dd>{result.chunk_count}</dd>
          <dt>Time</dt>
          <dd>{result.processing_time_seconds}s</dd>
        </dl>
      ) : null}
    </section>
  );
}
