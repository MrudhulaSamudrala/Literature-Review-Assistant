"""CLI and programmatic runner for reproducing RAG evaluation benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evaluation.dataset import EvalDataset, get_default_evaluation_dataset
from app.evaluation.evaluator import QueryEvaluationResult, RAGEvaluator
from app.evaluation.report import BASE_PAPER_TABLE_2, EvaluationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation_runner")


def run_evaluation(
    scenario: str = "all",
    dataset_path: str | Path | None = None,
    output_dir: str | Path = "data/evaluation",
    methods: list[str] | None = None,
    limit_per_scenario: int | None = None,
) -> dict[str, Any]:
    """Run full evaluation suite across methods and scenarios."""
    t_start = time.perf_counter()
    target_methods = methods or ["vector", "graph", "hybrid"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    if dataset_path:
        logger.info("Loading evaluation dataset from: %s", dataset_path)
        dataset = EvalDataset.load(dataset_path)
    else:
        logger.info("Using default curated evaluation dataset (20 questions).")
        dataset = get_default_evaluation_dataset()

    # 2. Filter samples
    samples = dataset.filter_by_scenario(scenario)
    if limit_per_scenario:
        single_samples = [s for s in samples if s.scenario == "single_paper"][:limit_per_scenario]
        multi_samples = [s for s in samples if s.scenario == "multi_paper"][:limit_per_scenario]
        samples = single_samples + multi_samples

    logger.info(
        "Starting evaluation: %d total samples across %d methods: %s",
        len(samples),
        len(target_methods),
        target_methods,
    )

    # 3. Initialize Evaluator & Reporter
    settings = get_settings()
    evaluator = RAGEvaluator(settings)
    reporter = EvaluationReporter(output_dir=output_path)

    # 4. Execute queries
    all_results: list[QueryEvaluationResult] = []
    total_evals = len(samples) * len(target_methods)
    eval_idx = 0

    for sample in samples:
        for method in target_methods:
            eval_idx += 1
            logger.info(
                "[%d/%d] Running %s on sample %s (%s)...",
                eval_idx,
                total_evals,
                method.upper(),
                sample.id,
                sample.scenario,
            )
            res = evaluator.evaluate_sample_method(sample, method)
            all_results.append(res)
            logger.info(
                "  -> Result: Cosine=%.3f, Faithfulness=%.3f, TotalLatency=%.1fms, Answer='%s'",
                res.cosine_similarity,
                res.faithfulness,
                res.total_latency_ms,
                res.generated_answer[:60],
            )

    # 5. Aggregate results
    aggregated = reporter.aggregate_results(all_results)

    # 6. Export artifacts
    json_path = reporter.export_json(all_results, aggregated)
    csv_path = reporter.export_csv(all_results)
    chart_paths = reporter.generate_charts(aggregated)
    md_path = reporter.generate_markdown_report(all_results, aggregated)

    elapsed_total = round(time.perf_counter() - t_start, 2)
    logger.info("Evaluation complete in %.2f seconds.", elapsed_total)
    logger.info("Generated JSON: %s", json_path)
    logger.info("Generated CSV: %s", csv_path)
    logger.info("Generated Markdown: %s", md_path)
    logger.info("Generated Charts: %d images in %s", len(chart_paths), output_path)

    # Print summary tables to stdout
    _print_terminal_summary(aggregated)

    return {
        "total_queries": len(all_results),
        "elapsed_seconds": elapsed_total,
        "aggregated": aggregated,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "report_path": str(md_path),
        "chart_paths": [str(p) for p in chart_paths],
    }


def _print_terminal_summary(aggregated: dict[str, Any]) -> None:
    """Print clean terminal summary tables for immediate viewing."""
    print("\n" + "=" * 80)
    print("                    EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    for scenario_key, scenario_title in [
        ("single_paper", "SINGLE-PAPER RAG (Paper P017)"),
        ("multi_paper", "MULTI-PAPER / DATABASE RAG (Papers P017 & P022)"),
    ]:
        print(f"\n--- {scenario_title} ---")
        print(f"{'Method':<12} | {'Cosine Sim (Mean±SD)':<22} | {'Faithfulness':<22} | {'Hit@5':<7} | {'MRR':<7} | {'Latency':<8}")
        print("-" * 88)
        for m in ["vector", "graph", "hybrid"]:
            data = aggregated.get(f"{scenario_key}_{m}", {})
            m_name = {"vector": "VectorRAG", "graph": "GraphRAG", "hybrid": "HybridRAG"}.get(m, m)
            cos = data.get("cosine_formatted", "N/A")
            faith = data.get("faithfulness_formatted", "N/A")
            hit5 = f"{data.get('hit_at_5', 0.0):.3f}"
            mrr = f"{data.get('mrr', 0.0):.3f}"
            lat = f"{data.get('avg_total_latency_ms', 0.0)/1000:.2f}s"
            print(f"{m_name:<12} | {cos:<22} | {faith:<22} | {hit5:<7} | {mrr:<7} | {lat:<8}")

    print("\n--- COMPARISON AGAINST BASE PAPER TABLE 2 ---")
    print(f"{'Method':<12} | {'Metric':<20} | {'Base Paper':<16} | {'Our System':<16} | {'Difference'}")
    print("-" * 80)
    for m in ["vector", "graph", "hybrid"]:
        base = BASE_PAPER_TABLE_2[m]
        sp_data = aggregated.get(f"single_paper_{m}", {})
        mp_data = aggregated.get(f"multi_paper_{m}", {})
        m_name = {"vector": "VectorRAG", "graph": "GraphRAG", "hybrid": "HybridRAG"}[m]

        # Single Cosine
        sp_cos = sp_data.get("cosine_similarity_mean", 0.0)
        diff_sc = sp_cos - base["single_paper_cosine"]
        print(f"{m_name:<12} | {'Single cosine':<20} | {base['single_paper_cosine']:.3f} ± {base['single_paper_cosine_std']:.3f}   | {sp_data.get('cosine_formatted', 'N/A'):<16} | {diff_sc:+.3f}")

        # Single Faithfulness
        sp_faith = sp_data.get("faithfulness_mean", 0.0)
        diff_sf = sp_faith - base["single_paper_faithfulness"]
        print(f"{m_name:<12} | {'Single faithfulness':<20} | {base['single_paper_faithfulness']:.3f} ± {base['single_paper_faithfulness_std']:.3f}   | {sp_data.get('faithfulness_formatted', 'N/A'):<16} | {diff_sf:+.3f}")

        # Database Cosine
        mp_cos = mp_data.get("cosine_similarity_mean", 0.0)
        diff_dc = mp_cos - base["database_cosine"]
        print(f"{m_name:<12} | {'Database cosine':<20} | {base['database_cosine']:.3f} ± {base['database_cosine_std']:.3f}   | {mp_data.get('cosine_formatted', 'N/A'):<16} | {diff_dc:+.3f}")

        # Database Faithfulness
        mp_faith = mp_data.get("faithfulness_mean", 0.0)
        diff_df = mp_faith - base["database_faithfulness"]
        print(f"{m_name:<12} | {'Database faithfulness':<20} | {base['database_faithfulness']:.3f} ± {base['database_faithfulness_std']:.3f}   | {mp_data.get('faithfulness_formatted', 'N/A'):<16} | {diff_df:+.3f}")

    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Benchmark.")
    parser.add_argument("--scenario", choices=["all", "single_paper", "multi_paper"], default="all")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom evaluation dataset JSON")
    parser.add_argument("--output-dir", type=str, default="data/evaluation", help="Directory for output reports & charts")
    parser.add_argument("--methods", type=str, default="vector,graph,hybrid", help="Comma-separated RAG methods")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per scenario")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_evaluation(
        scenario=args.scenario,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        methods=methods,
        limit_per_scenario=args.limit,
    )


if __name__ == "__main__":
    main()
