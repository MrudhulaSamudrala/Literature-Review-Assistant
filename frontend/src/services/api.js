const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const NETWORK_ERROR = "Unable to connect to the backend. Please make sure the FastAPI server is running.";

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error(NETWORK_ERROR);
  }
  return handleResponse(response);
}

async function handleResponse(response) {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

async function readError(response) {
  const text = await response.text();
  if (!text) {
    return `Request failed with status ${response.status}`;
  }
  if (text.includes("Traceback")) {
    return "The backend could not complete the request.";
  }
  try {
    const body = JSON.parse(text);
    if (typeof body.detail === "string") {
      return body.detail.includes("Traceback")
        ? "The backend could not complete the request."
        : body.detail;
    }
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || "Request failed").join("; ");
    }
  } catch {
    return "The backend could not complete the request.";
  }
  return "The backend could not complete the request.";
}

export function uploadPaper(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/papers/upload`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      const response = {
        ok: xhr.status >= 200 && xhr.status < 300,
        status: xhr.status,
        text: async () => xhr.responseText,
        json: async () => {
          if (!xhr.responseText) {
            return null;
          }
          return JSON.parse(xhr.responseText);
        },
      };
      handleResponse(response).then(resolve).catch(reject);
    };
    xhr.onerror = () => reject(new Error(NETWORK_ERROR));
    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}

export async function getPapers() {
  return request("/api/papers");
}

export async function getPaper(paperId) {
  return request(`/api/papers/${paperId}`);
}

export async function getPaperChunks(paperId) {
  return request(`/api/papers/${paperId}/chunks`);
}

export async function indexVector(paperId) {
  return request(`/api/rag/vector/index/${paperId}`, { method: "POST" });
}

export async function getVectorIndexStatus(paperId) {
  return request(`/api/rag/vector/index/${paperId}`);
}

export async function indexGraph(paperId) {
  return request(`/api/rag/graph/index/${paperId}`, { method: "POST" });
}

export async function queryVectorRAG({ question, paperIds, topK = 5 }) {
  return request("/api/rag/vector/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper_ids: paperIds, top_k: topK }),
  });
}

export async function retrieveVectorRAG({ question, paperIds, topK = 5 }) {
  return request("/api/rag/vector/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, paper_ids: paperIds, top_k: topK }),
  });
}

export async function queryGraphRAG({ question, paperIds, topK = 5, pathDepth = 2 }) {
  return request("/api/rag/graph/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      paper_ids: paperIds,
      top_k: topK,
      path_depth: pathDepth,
    }),
  });
}

export async function retrieveGraphRAG({ question, paperIds, topK = 5, pathDepth = 2 }) {
  return request("/api/rag/graph/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      paper_ids: paperIds,
      top_k: topK,
      path_depth: pathDepth,
    }),
  });
}

export async function queryHybridRAG({ question, paperIds, topK = 5, pathDepth = 2 }) {
  return request("/api/rag/hybrid/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      paper_ids: paperIds,
      top_k: topK,
      path_depth: pathDepth,
    }),
  });
}

export async function getHealth() {
  return request("/api/health");
}

export const uploadDocument = uploadPaper;
export const indexVectorPaper = indexVector;
export const queryVector = queryVectorRAG;
