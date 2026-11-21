"""Grounded prompt construction, validation, and answer cleanup for RAG pipelines."""

import re
from typing import Any

from app.rag.vector_rag.models import RetrievedChunk

NO_EVIDENCE_ANSWER = (
    "No relevant evidence was found in the indexed papers. No answer was generated."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The available evidence does not provide enough information to answer this question."
)

PROMPT_INSTRUCTION_PHRASES = [
    "do not copy the evidence",
    "do not include chunk ids",
    "do not invent facts",
    "do not invent citations",
    "do not invent graph relationships",
    "do not mention retrieval",
    "do not mention the evidence",
    "if the evidence is insufficient",
    "if the evidence does not answer",
    "give a concise",
    "give a short",
    "answer the question using only",
    "answer the question using the evidence",
    "answer the question based only",
    "use the following evidence",
    "the available evidence is insufficient",
    "the retrieved evidence is insufficient",
    "the available evidence does not provide enough information",
    "retrieved evidence:",
    "question:",
    "instructions:",
    "graph evidence:",
]

CHUNK_ID_PATTERN = re.compile(r"\[?(?:chunk_id=)?[A-Za-z0-9]+_C\d{3,}\]?", re.IGNORECASE)
METADATA_TAG_PATTERN = re.compile(
    r"\[?(?:paper_id|source_filename|section|page|origin)=[^\]]+\]?", re.IGNORECASE
)
SOURCE_TAG_PATTERN = re.compile(r"\[Source:\s*[^\]]+\]", re.IGNORECASE)


def build_grounded_prompt(question: str, context: str | list[RetrievedChunk]) -> str:
    """Build a concise, direct prompt asking FLAN-T5 for a 1-3 sentence grounded answer."""
    sources_summary = ""
    if isinstance(context, list):
        evidence_text = "\n\n".join(filter_table_noise(c.text) for c in context)
        sources_summary = "\n\nSources:\n" + "\n".join(
            f"[chunk_id={chunk.chunk_id} paper_id={chunk.paper_id} page={chunk.page} source_filename={chunk.source_filename}]"
            for chunk in context
        )
    else:
        evidence_text = str(context).strip()

    return (
        "Answer the question using the evidence.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Give a short, direct answer in 1-3 sentences.\n"
        "- Use only the evidence.\n"
        "- Do not copy the evidence verbatim.\n"
        "- Do not invent facts.\n"
        "- Do not invent citations.\n"
        "- Do not invent graph relationships.\n"
        "- Do not mention the evidence, instructions, sources, chunk IDs, or model.\n"
        f'- If the available evidence does not provide enough information to answer this question, say: "{INSUFFICIENT_EVIDENCE_ANSWER}"{sources_summary}'
    )


def build_strict_retry_prompt(question: str, context: str) -> str:
    """Build a strict single-sentence fallback prompt if the initial generation leaked artifacts."""
    return (
        f"Evidence:\n{context.strip()[:800]}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "State the answer in 1-2 direct sentences using only facts from the evidence above. "
        "Do not repeat instructions or copy whole sentences. "
        f'If unknown, say "{INSUFFICIENT_EVIDENCE_ANSWER}":'
    )


def filter_table_noise(text: str) -> str:
    """Filter out lines that are predominantly tabular floating-point numbers or evaluation metrics."""
    lines = []
    for line in text.split("\n"):
        tokens = line.split()
        if not tokens:
            continue
        num_floats = sum(1 for t in tokens if re.match(r"^-?\d+(?:\.\d+)?%?$", t.rstrip(".,;:")))
        if len(tokens) >= 3 and num_floats / len(tokens) > 0.4:
            continue
        lines.append(line)
    clean_text = "\n".join(lines).strip()
    return clean_text or text.strip()


def is_leaked_or_invalid(raw_answer: str, context: str, question: str) -> bool:
    """Check if the raw model output leaked prompt text, metadata, table noise, or copied a chunk."""
    if not raw_answer or not raw_answer.strip():
        return True

    text = raw_answer.strip()
    lower = text.lower()

    # 1. Obvious leakage patterns
    leakage_phrases = [
        "instructions:",
        "sources:",
        "evidence:",
        "question:",
        "### vectorrag",
        "### graphrag",
        "### hybridrag",
        "development fallback",
        "not equivalent to gpt-4o-mini",
        "[chunk_id=",
        "[source:",
        "chunk_id=",
        "paper_id=",
        "source_filename=",
    ]
    for phrase in leakage_phrases:
        if phrase in lower:
            return True

    # 2. Prompt instruction echoes
    for phrase in PROMPT_INSTRUCTION_PHRASES:
        if phrase in lower and len(text) <= len(phrase) + 30:
            return True

    # 3. Pure bracketed citations like [7], [1, 2], [P001_C001]
    if re.fullmatch(r"\[[A-Za-z0-9_,\s-]+\]\.?", text):
        return True

    # 4. Check if text is predominantly numbers or table metrics
    words = text.split()
    if words:
        num_floats = sum(1 for w in words if re.match(r"^-?\d+(?:\.\d+)?%?$", w.rstrip(".,;:")))
        if num_floats / len(words) > 0.4 or (num_floats >= 3 and len(words) <= 6):
            return True

    # 5. Check if output is basically identical to an entire retrieved chunk (>40 words verbatim copy)
    clean_answer = re.sub(r"\s+", " ", lower).strip()
    if len(words) >= 40 and context:
        clean_context = re.sub(r"\s+", " ", context.lower())
        if clean_answer in clean_context:
            return True

    # 6. Check if output simply echoes the question
    q_lower = re.sub(r"\s+", " ", question.lower()).strip()
    if clean_answer == q_lower:
        return True

    return False


def clean_and_validate_answer(
    raw_answer: str,
    question: str,
    context_text: str = "",
) -> str:
    """Validate and clean the generated answer, ensuring no raw chunk IDs, table dumps, or leaked metadata."""
    if not raw_answer or not raw_answer.strip():
        return INSUFFICIENT_EVIDENCE_ANSWER

    text = raw_answer.strip()

    # 1. If it leaked prompt phrases, reject
    if is_leaked_or_invalid(text, context_text, question):
        return INSUFFICIENT_EVIDENCE_ANSWER

    # 2. Check if raw output is purely a chunk ID or metadata tag
    stripped_tags = SOURCE_TAG_PATTERN.sub("", text)
    stripped_tags = CHUNK_ID_PATTERN.sub("", stripped_tags)
    stripped_tags = METADATA_TAG_PATTERN.sub("", stripped_tags).strip()
    residual = re.sub(r"^[\[\]\(\)\s:;,\.-]+|[\[\]\(\)\s:;,\.-]+$", "", stripped_tags).strip()
    if not residual:
        return INSUFFICIENT_EVIDENCE_ANSWER

    # 3. Clean up leaked tags embedded inside sentences
    cleaned = SOURCE_TAG_PATTERN.sub("", text)
    cleaned = CHUNK_ID_PATTERN.sub("", cleaned)
    cleaned = METADATA_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    if len(cleaned) < 3:
        return INSUFFICIENT_EVIDENCE_ANSWER

    # 4. Check if output is predominantly numbers/table metrics
    words = cleaned.split()
    num_numeric = sum(1 for w in words if re.match(r"^-?\d+(?:\.\d+)?%?$", w.rstrip(".,;:")))
    if words and (num_numeric / len(words) > 0.4 or (num_numeric >= 3 and len(words) <= 6)):
        return INSUFFICIENT_EVIDENCE_ANSWER

    # 5. Check if the model echoed an unsupported entity from the question
    q_lower = question.lower().strip()
    c_lower = cleaned.lower()
    if len(cleaned.split()) <= 3 and c_lower in q_lower:
        if context_text and c_lower not in context_text.lower():
            return INSUFFICIENT_EVIDENCE_ANSWER

    # Capitalize first letter and ensure ending punctuation
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."

    return cleaned
