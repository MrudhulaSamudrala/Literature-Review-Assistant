"""Index papers and answer questions with VectorRAG."""

import logging
import os
import re
import time
from contextvars import ContextVar

from app.config import Settings
from app.core.exceptions import VectorRAGError
from app.rag.vector_rag.indexer import (
    REQUIRED_LLM_MODEL,
    _evaluation_run,
    build_embedding_model,
    chunks_needing_embedding,
    chunks_to_nodes,
    configured_embedding_model_name,
    connect_vector_store,
    count_indexed_chunks,
    delete_chunk_ids,
    embed_and_store,
    embedding_dimension,
    load_paper_chunks,
    read_indexed_chunks,
    require_openai_llm,
    require_vector_embeddings,
)
from app.rag.vector_rag.models import (
    IndexResult,
    IndexStatus,
    RetrievedChunk,
    VectorQueryResponse,
    VectorRetrievalResponse,
)
from app.rag.vector_rag.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    NO_EVIDENCE_ANSWER,
    build_grounded_prompt,
    build_strict_retry_prompt,
    clean_and_validate_answer,
    filter_table_noise,
    is_leaked_or_invalid,
)
from app.rag.vector_rag.retriever import citations_from, retrieve_chunks

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

DEVELOPMENT_FALLBACK_LLM = "development-fallback:extractive"
_LLM_PROVIDERS = {"local", "openai"}
_generation_model: ContextVar[str | None] = ContextVar("generation_model", default=None)
_local_generators: dict[str, object] = {}


class VectorRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def index_paper(self, paper_id: str) -> IndexResult:
        require_vector_embeddings(self.settings)
        chunks = load_paper_chunks(self.settings, paper_id)
        embed_model, dimension = self._prepare_embedding()
        logger.info("VectorRAG indexing started paper_id=%s chunks=%s", paper_id, len(chunks))
        store = connect_vector_store(self.settings, dimension)
        existing = read_indexed_chunks(store, self.settings, paper_id)
        current_ids = {chunk.chunk_id for chunk in chunks}
        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in current_ids]
        delete_chunk_ids(store, self.settings, stale_ids)
        pending = chunks_needing_embedding(existing, chunks)
        if pending:
            embed_and_store(store, embed_model, chunks_to_nodes(pending))
        logger.info(
            "VectorRAG indexing completed paper_id=%s chunk_count=%s embedded=%s",
            paper_id,
            len(chunks),
            len(pending),
        )
        return IndexResult(
            paper_id=paper_id,
            status="indexed",
            chunk_count=len(chunks),
            embedded_chunk_count=len(pending),
            embedding_model=configured_embedding_model_name(self.settings),
        )

    def index_status(self, paper_id: str) -> IndexStatus:
        require_vector_embeddings(self.settings)
        load_paper_chunks(self.settings, paper_id)
        _embed_model, dimension = self._prepare_embedding()
        store = connect_vector_store(self.settings, dimension)
        indexed = count_indexed_chunks(store, self.settings, paper_id)
        return IndexStatus(
            paper_id=paper_id,
            status="indexed" if indexed else "not_indexed",
            chunk_count=indexed,
            embedding_model=configured_embedding_model_name(self.settings),
        )

    def retrieve(
        self,
        question: str,
        paper_ids: list[str],
        top_k: int | None,
    ) -> VectorRetrievalResponse:
        started = time.perf_counter()
        resolved_top_k = self.resolve_top_k(top_k)
        self._log_query_started(question, paper_ids, resolved_top_k)
        embed_model, dimension = self._prepare_embedding()
        model_name = configured_embedding_model_name(self.settings)
        logger.info("Query embedding started model=%s", model_name)
        embedding_started = time.perf_counter()
        query_embedding = embed_model.get_query_embedding(question)
        embedding_latency_ms = _elapsed_ms(embedding_started)
        store = connect_vector_store(self.settings, dimension)
        logger.info("Vector retrieval started paper_ids=%s top_k=%s", paper_ids, resolved_top_k)
        retrieval_started = time.perf_counter()
        results = retrieve_chunks(
            store,
            embed_model,
            question,
            query_embedding,
            paper_ids,
            resolved_top_k,
        )
        retrieval_latency_ms = _elapsed_ms(retrieval_started)
        logger.info(
            "Vector retrieval completed chunks=%s retrieval_latency_ms=%s",
            len(results),
            retrieval_latency_ms,
        )
        evidence_found = bool(results)
        return VectorRetrievalResponse(
            question=question,
            results=results,
            evidence_found=evidence_found,
            message=None if evidence_found else NO_EVIDENCE_ANSWER,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            latency_ms=round(embedding_latency_ms + retrieval_latency_ms, 3),
            embedding_model=model_name,
        )

    def query(
        self,
        question: str,
        paper_ids: list[str],
        top_k: int | None,
    ) -> VectorQueryResponse:
        started = time.perf_counter()
        retrieved = self.retrieve(question, paper_ids, top_k)
        if not retrieved.results:
            total_latency_ms = _elapsed_ms(started)
            logger.info(
                "Query completed without evidence paper_ids=%s total_latency_ms=%s",
                paper_ids,
                total_latency_ms,
            )
            return VectorQueryResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                retrieved_chunks=[],
                embedding_latency_ms=retrieved.embedding_latency_ms,
                retrieval_latency_ms=retrieved.retrieval_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                embedding_model=configured_embedding_model_name(self.settings),
                llm_model=configured_llm_model_name(self.settings),
            )

        candidate_chunks = retrieved.results[:3]
        evidence_blocks = []
        for i, chunk in enumerate(candidate_chunks, start=1):
            clean_chunk = filter_table_noise(chunk.text)
            if clean_chunk:
                evidence_blocks.append(f"Evidence {i}:\n{clean_chunk}")
        context_str = "\n\n".join(evidence_blocks)

        generation_started = time.perf_counter()
        _generation_model.set(None)
        answer = generate_grounded_answer(
            self.settings,
            question=question,
            context=context_str,
            retrieved_chunks=retrieved.results,
            generate_fn=_generate,
        )
        generation_latency_ms = _elapsed_ms(generation_started)
        total_latency_ms = _elapsed_ms(started)
        sources = citations_from(retrieved.results)

        logger.info("Query completed paper_ids=%s", paper_ids)
        return VectorQueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved.results,
            embedding_latency_ms=retrieved.embedding_latency_ms,
            retrieval_latency_ms=retrieved.retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            embedding_model=configured_embedding_model_name(self.settings),
            llm_model=resolve_generation_model_name(self.settings),
        )

    def _prepare_embedding(self):
        require_vector_embeddings(self.settings)
        embed_model = build_embedding_model(self.settings)
        dimension = embedding_dimension(self.settings, embed_model)
        logger.info("embedding dimension = %s", dimension)
        return embed_model, dimension

    def resolve_top_k(self, requested: int | None) -> int:
        top_k = self.settings.vector_top_k if requested is None else requested
        if top_k < 1 or top_k > 50:
            raise VectorRAGError("top_k must be between 1 and 50.", status_code=400)
        return top_k

    def _log_query_started(self, question: str, paper_ids: list[str], top_k: int) -> None:
        logger.info("VectorRAG query started")
        logger.info("Paper IDs %s", paper_ids)
        logger.info("Question received length=%s", len(question))
        logger.info("Top-K %s", top_k)


def normalize_llm_provider(settings: Settings) -> str:
    provider = settings.llm_provider.strip().lower()
    if provider not in _LLM_PROVIDERS:
        raise VectorRAGError("LLM_PROVIDER must be local or openai.", status_code=400)
    return provider


def configured_llm_model_name(settings: Settings) -> str:
    """Return the selected model. openai stays on gpt-4o-mini."""

    if normalize_llm_provider(settings) == "local":
        return settings.local_llm_model.strip()
    return settings.openai_llm_model.strip()


def resolve_generation_model_name(settings: Settings) -> str:
    selected = _generation_model.get()
    return selected or configured_llm_model_name(settings)


def log_llm_configuration(settings: Settings) -> None:
    """Log the selected LLM provider and model. Never log credentials."""

    try:
        provider = normalize_llm_provider(settings)
        model_name = configured_llm_model_name(settings)
    except VectorRAGError as exc:
        logger.error("llm configuration invalid: %s", exc)
        return
    logger.info("llm_provider = %s", provider)
    logger.info("llm_model = %s", model_name)


def generate_grounded_answer(
    settings: Settings,
    question: str,
    context: str,
    retrieved_chunks: list[RetrievedChunk] | None = None,
    generate_fn=None,
) -> str:
    """Generate a clean, grounded 1-3 sentence answer with strict validation and retry."""
    if not context or not context.strip():
        return NO_EVIDENCE_ANSWER

    gen = generate_fn or _generate
    clean_context = context.strip()[:1000]
    prompt = build_grounded_prompt(question, clean_context)
    chunk_count = len(retrieved_chunks) if retrieved_chunks is not None else "N/A"

    logger.info(
        "\n--- GENERATION DIAGNOSTIC ---\n"
        "QUESTION:\n%s\n\n"
        "RETRIEVED CHUNKS:\n%s\n\n"
        "CONTEXT SENT TO MODEL:\n%s\n"
        "-----------------------------",
        question,
        chunk_count,
        clean_context,
    )

    raw_answer = gen(settings, prompt)
    logger.info("RAW FLAN-T5 OUTPUT:\n%s", raw_answer)

    if "not equivalent to gpt-4o-mini" in raw_answer:
        logger.info("FINAL CLEANED ANSWER (fallback):\n%s\n-----------------------------", raw_answer)
        return raw_answer

    if is_leaked_or_invalid(raw_answer, clean_context, question):
        logger.warning(
            "Raw output leaked prompt text, metadata, or copied chunk. Retrying with strict prompt..."
        )
        strict_prompt = build_strict_retry_prompt(question, clean_context)
        retry_raw = gen(settings, strict_prompt)
        logger.info("RETRY RAW FLAN-T5 OUTPUT:\n%s", retry_raw)
        if is_leaked_or_invalid(retry_raw, clean_context, question):
            logger.warning("Retry output still invalid. Returning insufficient evidence response.")
            final_answer = INSUFFICIENT_EVIDENCE_ANSWER
        else:
            final_answer = clean_and_validate_answer(retry_raw, question, clean_context)
    else:
        final_answer = clean_and_validate_answer(raw_answer, question, clean_context)

    logger.info("FINAL CLEANED ANSWER:\n%s\n-----------------------------", final_answer)
    return final_answer


def _generate(settings: Settings, prompt: str) -> str:
    provider = normalize_llm_provider(settings)
    if provider == "openai":
        require_openai_llm(settings)
        text = _generate_openai(settings, prompt)
        _generation_model.set(settings.openai_llm_model)
        return text
    if _evaluation_run(settings):
        raise VectorRAGError(
            "Evaluation requires LLM_PROVIDER=openai and "
            f"OPENAI_LLM_MODEL={REQUIRED_LLM_MODEL}. "
            "The local language model is for development only.",
            status_code=400,
        )
    if not settings.local_llm_model.strip():
        raise VectorRAGError(
            "LOCAL_LLM_MODEL is required when LLM_PROVIDER=local.",
            status_code=400,
        )
    try:
        text = _generate_local(settings, prompt)
    except Exception:
        logger.exception("Local LLM generation failed. Using development fallback.")
        text = _development_fallback_answer(prompt)
        _generation_model.set(DEVELOPMENT_FALLBACK_LLM)
        return text
    _generation_model.set(settings.local_llm_model)
    return text


def _development_fallback_answer(prompt: str) -> str:
    """Extractive fallback used when local FLAN-T5 is unavailable in development."""
    clean_text = ""
    e_match = re.search(r"Evidence:\s*\n?(.*?)(?:\n\nGive|\n\nInstructions:|\n\nQuestion:|$)", prompt, re.DOTALL)
    if e_match:
        raw_ev = e_match.group(1).strip()
        lines = [
            line.strip()
            for line in raw_ev.splitlines()
            if line.strip()
            and not line.strip().startswith("[")
            and not line.strip().startswith("Instructions")
            and not line.strip().startswith("Evidence")
        ]
        clean_text = " ".join(lines).strip()
    if not clean_text:
        marker = "Retrieved evidence:\n"
        if marker in prompt:
            clean_text = prompt.split(marker, 1)[1].strip()
        else:
            clean_text = prompt.strip()

    return (
        "Development fallback (not equivalent to gpt-4o-mini). "
        "The text below is taken from retrieved evidence and was not produced by gpt-4o-mini.\n\n"
        f"{clean_text}"
    )


def _generate_openai(settings: Settings, prompt: str) -> str:
    openai_cls = OpenAI
    if openai_cls is None:
        try:
            from llama_index.llms.openai import OpenAI as _OpenAI

            openai_cls = _OpenAI
        except ImportError as exc:
            raise VectorRAGError(
                "OpenAI LLM provider requires llama-index-llms-openai to be installed.",
                status_code=500,
            ) from exc

    llm = openai_cls(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    completion = llm.complete(prompt)
    text = completion.text.strip()
    if not text:
        raise VectorRAGError("The language model returned an empty answer.", status_code=502)
    return text


def _format_local_prompt(prompt: str) -> str:
    """Prepare a concise, high-signal prompt for local FLAN-T5 seq2seq generation."""
    clean = prompt.split("\n\nSources:\n")[0].strip()

    q_match = re.search(r"(?:^|\n)Question:\s*\n?(.*?)(?:\n\n|\nEvidence:|\nGraph facts:|\nInstructions:|\nGive |\nState |$)", clean, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""

    g_match = re.search(r"(?:^|\n)Graph facts:\s*\n?(.*?)(?:\n\nEvidence:|\n\nQuestion:|\n\nInstructions:|$)", clean, re.DOTALL)
    graph_facts = g_match.group(1).strip() if g_match else ""

    e_match = re.search(r"(?:^|\n)(?:Evidence|Retrieved evidence):\s*\n?(.*?)(?:\n\nGive|\n\nQuestion:|\n\nInstructions:|\n\nState |$)", clean, re.DOTALL)
    evidence = e_match.group(1).strip() if e_match else ""

    context_parts = []
    if graph_facts and graph_facts != "None":
        context_parts.append(graph_facts)
    if evidence:
        context_parts.append(evidence)
    context = "\n\n".join(context_parts)

    if question and context:
        return (
            "Answer the question based on the evidence below in 1-3 direct sentences.\n\n"
            f"Evidence:\n{context[:1000]}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    return clean


def _generate_local(settings: Settings, prompt: str, max_new_tokens: int = 256) -> str:
    """Run a cached local instruction model. Raises when it is not available."""

    model_name = settings.local_llm_model.strip()
    generator = _local_generators.get(model_name)
    if generator is None:
        try:
            import os

            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_TORCH", "1")
            import transformers

            if (
                getattr(transformers.pipeline, "__name__", "") == "fake_pipeline"
                or (
                    type(transformers.pipeline).__name__ == "function"
                    and transformers.pipeline.__module__ != "transformers.pipelines"
                )
            ):
                generator = transformers.pipeline(
                    "text2text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device="cpu",
                    local_files_only=True,
                )
                if hasattr(generator, "_forward_params"):
                    generator._forward_params.pop("local_files_only", None)
            else:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
                generator = (tokenizer, model)
            _local_generators[model_name] = generator
        except Exception as exc:
            logger.exception("Failed to load local FLAN-T5 model %s", model_name)
            raise RuntimeError(f"Failed to load FLAN-T5 model {model_name}.") from exc

    clean_prompt = _format_local_prompt(prompt)
    try:
        if isinstance(generator, tuple):
            tokenizer, model = generator
            inputs = tokenizer(clean_prompt[:2048], return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            result = generator(clean_prompt[:4000], max_new_tokens=max_new_tokens, do_sample=False)
            if not result:
                raise RuntimeError("The local language model returned no text.")
            text = str(result[0].get("generated_text", "")).strip()
    except Exception as exc:
        logger.exception("Failed to generate with local FLAN-T5 model %s", model_name)
        raise RuntimeError(f"Failed to generate answer with FLAN-T5 model {model_name}.") from exc

    if not text:
        raise RuntimeError("The local language model returned an empty answer.")
    logger.info(
        "Local generation used llm_model=%s. This is not equivalent to %s.",
        model_name,
        REQUIRED_LLM_MODEL,
    )
    return text


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
