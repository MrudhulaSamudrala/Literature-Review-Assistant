export const RAG_MODES = [
  {
    id: "vector",
    label: "VectorRAG",
    description: "Semantic similarity-based retrieval",
  },
  {
    id: "graph",
    label: "GraphRAG",
    description: "Entity and relationship-based retrieval",
  },
  {
    id: "hybrid",
    label: "HybridRAG",
    description: "Combines vector and graph evidence",
  },
  {
    id: "compare",
    label: "Compare All",
    description: "Run the same question through all three methods",
  },
];

export const RAG_METHODS = RAG_MODES;

export const EXAMPLE_QUESTIONS = [
  "What methodology does this paper propose?",
  "What are the main findings?",
  "How do the approaches in these papers differ?",
  "What relationship exists between the identified concepts?",
  "What limitations are discussed?",
];

export const GRAPH_INDEX_STORAGE_KEY = "slra-graph-index-confirmed";
