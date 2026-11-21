import { useEffect, useState } from "react";
import AnswerPanel from "../components/AnswerPanel.jsx";
import ModeSelector from "../components/ModeSelector.jsx";
import PdfUpload from "../components/PdfUpload.jsx";
import QuestionForm from "../components/QuestionForm.jsx";
import SourceList from "../components/SourceList.jsx";
import VectorIndexPanel from "../components/VectorIndexPanel.jsx";
import {
  getHealth,
  indexGraph,
  indexVectorPaper,
  queryGraphRAG,
  queryHybridRAG,
  queryVectorRAG,
  uploadDocument,
} from "../services/api.js";

export default function HomePage() {
  const [mode, setMode] = useState("vector");
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [indexResult, setIndexResult] = useState(null);
  const [indexError, setIndexError] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [graphIndexResult, setGraphIndexResult] = useState(null);
  const [graphIndexError, setGraphIndexError] = useState("");
  const [graphIndexing, setGraphIndexing] = useState(false);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [health, setHealth] = useState("checking API...");

  useEffect(() => {
    getHealth()
      .then((payload) => setHealth(`${payload.service} · ${payload.status}`))
      .catch(() => setHealth("API unavailable — start the FastAPI backend on port 8000"));
  }, []);

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setUploadStatus("");
    setUploadError("");
    setUploadResult(null);
    setIndexResult(null);
    setIndexError("");
    setGraphIndexResult(null);
    setGraphIndexError("");
    try {
      const payload = await uploadDocument(file);
      setUploadResult(payload);
      setUploadStatus(
        `${payload.paper_id} ingested with ${payload.page_count} pages and ${payload.chunk_count} chunks.`
      );
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleIndex() {
    if (!uploadResult?.paper_id) return;
    setIndexing(true);
    setIndexError("");
    setIndexResult(null);
    try {
      const payload = await indexVectorPaper(uploadResult.paper_id);
      setIndexResult(payload);
    } catch (err) {
      setIndexError(err.message);
    } finally {
      setIndexing(false);
    }
  }

  async function handleGraphIndex() {
    if (!uploadResult?.paper_id) return;
    setGraphIndexing(true);
    setGraphIndexError("");
    setGraphIndexResult(null);
    try {
      const payload = await indexGraph(uploadResult.paper_id);
      setGraphIndexResult(payload);
    } catch (err) {
      setGraphIndexError(err.message);
    } finally {
      setGraphIndexing(false);
    }
  }

  async function handleAsk() {
    if (!uploadResult?.paper_id || !question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const paperIds = [uploadResult.paper_id];
      let payload;
      if (mode === "vector") {
        payload = await queryVectorRAG({ question, paperIds });
      } else if (mode === "graph") {
        payload = await queryGraphRAG({ question, paperIds });
      } else if (mode === "hybrid") {
        payload = await queryHybridRAG({ question, paperIds });
      } else if (mode === "compare") {
        const [vRes, gRes, hRes] = await Promise.allSettled([
          queryVectorRAG({ question, paperIds }),
          queryGraphRAG({ question, paperIds }),
          queryHybridRAG({ question, paperIds }),
        ]);
        const comparison = {
          vector: {
            answer: vRes.status === "fulfilled" ? vRes.value.answer : `Error: ${vRes.reason?.message || "Failed"}`,
            sources: vRes.status === "fulfilled" ? (vRes.value.sources || []) : [],
          },
          graph: {
            answer: gRes.status === "fulfilled" ? gRes.value.answer : `Error: ${gRes.reason?.message || "Failed"}`,
            sources: gRes.status === "fulfilled" ? (gRes.value.sources || []) : [],
          },
          hybrid: {
            answer: hRes.status === "fulfilled" ? hRes.value.answer : `Error: ${hRes.reason?.message || "Failed"}`,
            sources: hRes.status === "fulfilled" ? (hRes.value.sources || []) : [],
          },
        };
        const parts = [
          `VectorRAG:\n${comparison.vector.answer}`,
          `GraphRAG:\n${comparison.graph.answer}`,
          `HybridRAG:\n${comparison.hybrid.answer}`,
        ];
        const combinedSources = [
          ...comparison.vector.sources,
          ...comparison.graph.sources,
          ...comparison.hybrid.sources,
        ];
        payload = {
          retrieval_method: "Compare All",
          comparison,
          answer: parts.join("\n\n"),
          sources: combinedSources,
        };
      }
      setResult(payload);
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const vectorIndexed = indexResult?.status === "indexed";
  const graphIndexed = graphIndexResult?.status === "indexed";
  const hasQuestion = question.trim().length > 0;
  const canAsk =
    uploadResult?.paper_id &&
    hasQuestion &&
    (mode === "vector"
      ? vectorIndexed
      : mode === "graph"
      ? graphIndexed
      : mode === "hybrid"
      ? vectorIndexed || graphIndexed
      : vectorIndexed || graphIndexed);

  let askHint = "";
  if (!uploadResult?.paper_id) {
    askHint = "Upload a PDF first.";
  } else if (mode === "vector" && !vectorIndexed) {
    askHint = "Index this paper for VectorRAG before asking.";
  } else if (mode === "graph" && !graphIndexed) {
    askHint = "Index this paper for GraphRAG before asking.";
  } else if (mode === "hybrid" && !vectorIndexed && !graphIndexed) {
    askHint = "Index this paper before asking with HybridRAG.";
  } else {
    askHint = `${
      mode === "vector"
        ? "VectorRAG"
        : mode === "graph"
        ? "GraphRAG"
        : mode === "hybrid"
        ? "HybridRAG"
        : "Compare All"
    } will retrieve evidence and generate an answer using local FLAN-T5.`;
  }

  return (
    <main className="layout">
      <header>
        <p className="eyebrow">Smart Literature Review Assistant</p>
        <h1>Compare VectorRAG, GraphRAG, and HybridRAG</h1>
        <p className="muted">{health}</p>
      </header>
      <ModeSelector value={mode} onChange={setMode} />
      <div className="grid">
        <PdfUpload
          file={file}
          onFileChange={setFile}
          onUpload={handleUpload}
          status={uploadStatus}
          error={uploadError}
          result={uploadResult}
          uploading={uploading}
        />
        <QuestionForm
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={handleAsk}
          loading={loading}
          canAsk={canAsk}
          hint={askHint}
          mode={mode}
        />
      </div>
      <VectorIndexPanel
        result={uploadResult}
        indexing={indexing}
        indexResult={indexResult}
        indexError={indexError}
        onIndex={handleIndex}
        graphIndexing={graphIndexing}
        graphIndexResult={graphIndexResult}
        graphIndexError={graphIndexError}
        onGraphIndex={handleGraphIndex}
      />
      <AnswerPanel result={result} error={error} />
      <SourceList sources={result?.sources} />
    </main>
  );
}
