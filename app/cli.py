import os
from typing import Any, List

import typer

app = typer.Typer(help="LLM Evaluation Framework CLI")


def get_metrics_list(metric_names: str = None) -> List[Any]:
    from app.evaluation.metrics.bertscore import BERTScoreMetric
    from app.evaluation.metrics.bias import BiasMetric
    from app.evaluation.metrics.bleu import BLEUMetric
    from app.evaluation.metrics.cost import CostMetric
    from app.evaluation.metrics.custom_metric import KeywordMatchMetric
    from app.evaluation.metrics.faithfulness import FaithfulnessMetric
    from app.evaluation.metrics.hallucination import HallucinationMetric
    from app.evaluation.metrics.latency import LatencyMetric
    from app.evaluation.metrics.llm_judge import LLMJudgeMetric
    from app.evaluation.metrics.rouge import ROUGEMetric
    from app.evaluation.metrics.toxicity import ToxicityMetric

    mapping = {
        "bleu": BLEUMetric,
        "rouge": ROUGEMetric,
        "bertscore": BERTScoreMetric,
        "faithfulness": FaithfulnessMetric,
        "hallucination": HallucinationMetric,
        "toxicity": ToxicityMetric,
        "bias": BiasMetric,
        "latency": LatencyMetric,
        "cost": CostMetric,
        "judge": LLMJudgeMetric,
        "custom": KeywordMatchMetric
    }

    if not metric_names:
        return [cls() for cls in mapping.values()]

    names = [n.strip().lower() for n in metric_names.split(",")]
    instances = []
    for name in names:
        if name in mapping:
            instances.append(mapping[name]())
        else:
            typer.echo(f"Warning: Unknown metric '{name}' ignored.")
    return instances


@app.command()
def evaluate(
    dataset_path: str,
    metrics_list: str = typer.Option(
        None,
        "--metrics",
        "-m",
        help="Comma-separated list of metrics (e.g., bleu,rouge,bias)"
    ),
    experiment: str = typer.Option(
        "LLM-Evaluation",
        "--experiment",
        "-e",
        help="MLflow experiment name"
    ),
    output_dir: str = typer.Option(
        "reports",
        "--output-dir",
        "-o",
        help="Directory to save JSON and CSV reports"
    )
):
    """Run full evaluation pipeline on a CSV/JSON dataset."""
    from app.evaluation.evaluator import Evaluator
    from app.services.dataset_loader import DatasetLoader
    from app.services.experiment_tracker import ExperimentTracker
    from app.services.report_generator import ReportGenerator

    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset file not found: {dataset_path}")
        raise typer.Exit(code=1)

    typer.echo("Loading dataset...")
    loader = DatasetLoader(dataset_path)
    df = loader.load()
    loader.summary(df)

    typer.echo("Initializing metrics...")
    metrics = get_metrics_list(metrics_list)
    enabled_names = [m.name for m in metrics]
    typer.echo(f"Enabled metrics: {', '.join(enabled_names)}")

    evaluator = Evaluator(metrics)
    typer.echo("Running evaluations (this may take a moment for local models)...")
    results = evaluator.evaluate_dataset(df)

    # Save reports
    report_gen = ReportGenerator(results)
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    json_path = os.path.join(output_dir, "json", f"{dataset_name}_report.json")
    csv_path = os.path.join(output_dir, "csv", f"{dataset_name}_report.csv")

    saved_json = report_gen.to_json(json_path)
    saved_csv = report_gen.to_csv(csv_path)

    # Output to Console
    report_gen.print_summary()

    # Track experiment
    typer.echo("Logging experiment tracking details...")
    tracker = ExperimentTracker(experiment_name=experiment)
    tracker.log_evaluation(
        dataset_name=dataset_name,
        summary=results["summary"],
        enabled_metrics=enabled_names,
        report_path=str(saved_json)
    )

    typer.echo(f"Reports saved successfully to:")
    typer.echo(f" - JSON: {saved_json}")
    typer.echo(f" - CSV: {saved_csv}")


@app.command()
def compare(
    model_a_csv: str,
    model_b_csv: str,
    metric: str = typer.Option(
        "bertscore",
        "--metric",
        "-m",
        help="Metric key to base comparison on (e.g. bleu, rouge, bertscore)"
    ),
    tie_threshold: float = typer.Option(
        0.01,
        "--threshold",
        "-t",
        help="Tie threshold for score differences"
    )
):
    """Compare outputs of Model A and Model B using pairwise comparison."""
    import pandas as pd
    from app.evaluation.metrics.pairwise import (
        PairwiseComparisonMetric,
        WinRateCalculator,
    )
    from rich.console import Console
    from rich.panel import Panel

    if not os.path.exists(model_a_csv) or not os.path.exists(model_b_csv):
        typer.echo("Error: Model CSV dataset files not found.")
        raise typer.Exit(code=1)

    df_a = pd.read_csv(model_a_csv)
    df_b = pd.read_csv(model_b_csv)

    if len(df_a) != len(df_b):
        typer.echo("Error: Datasets must contain the same number of rows.")
        raise typer.Exit(code=1)

    metrics_list = get_metrics_list(metric)
    if not metrics_list:
        typer.echo(f"Error: Unknown comparison metric '{metric}'")
        raise typer.Exit(code=1)

    scorer = metrics_list[0]
    pairwise = PairwiseComparisonMetric(tie_threshold=tie_threshold)

    comparison_results = []
    typer.echo(f"Scoring comparisons using '{scorer.name}'...")

    for idx in range(len(df_a)):
        row_a = df_a.iloc[idx]
        row_b = df_b.iloc[idx]

        res_a = scorer.calculate(
            reference=row_a["reference_answer"],
            prediction=row_a["model_answer"],
            prompt=row_a["prompt"],
            context=row_a.get("context", "")
        )

        res_b = scorer.calculate(
            reference=row_b["reference_answer"],
            prediction=row_b["model_answer"],
            prompt=row_b["prompt"],
            context=row_b.get("context", "")
        )

        def extract_score(res):
            if "f1" in res:
                return res["f1"]
            if "score" in res:
                return res["score"]
            if "rougeL" in res:
                return res["rougeL"]
            return 0.0

        score_a = extract_score(res_a)
        score_b = extract_score(res_b)

        comparison = pairwise.calculate(
            reference=row_a["reference_answer"],
            prediction=row_a["model_answer"],
            model_a_score=score_a,
            model_b_score=score_b
        )
        comparison_results.append(comparison)

    win_rate = WinRateCalculator.calculate(comparison_results)

    console = Console()
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]PAIRWISE COMPARISON SUMMARY ({scorer.name})[/bold cyan]"
    ))
    console.print(f"Total Comparisons : {win_rate['total_comparisons']}")
    console.print(f"Model A Wins      : {win_rate['model_a_wins']}")
    console.print(f"Model B Wins      : {win_rate['model_b_wins']}")
    console.print(f"Ties              : {win_rate['ties']}")
    console.print(
        f"Model A Win Rate  : [green]{win_rate['model_a_win_rate']:.2%}[/green]"
    )
    console.print(
        f"Model B Win Rate  : [green]{win_rate['model_b_win_rate']:.2%}[/green]"
    )
    console.print(
        f"Tie Rate          : [yellow]{win_rate['tie_rate']:.2%}[/yellow]"
    )
    console.print("=" * 60 + "\n")


@app.command()
def human_eval(
    dataset_path: str,
    output_path: str = typer.Option(
        "reports/json/human_evaluations.json",
        "--output",
        "-o",
        help="Path to save evaluation report"
    )
):
    """Run terminal-based interactive human ratings for a dataset."""
    from app.evaluation.metrics.human_eval import (
        HumanEvaluationStore,
        SingleHumanEvaluationMetric,
    )
    from app.services.dataset_loader import DatasetLoader

    if not os.path.exists(dataset_path):
        typer.echo(f"Error: Dataset file not found: {dataset_path}")
        raise typer.Exit(code=1)

    loader = DatasetLoader(dataset_path)
    df = loader.load()
    loader.summary(df)

    metric = SingleHumanEvaluationMetric()
    results = []

    for index, row in df.iterrows():
        typer.echo(f"\nEvaluating Row {index + 1} of {len(df)}:")
        res = metric.calculate(
            reference=row["reference_answer"],
            prediction=row["model_answer"],
            prompt=row["prompt"]
        )
        results.append(res)

    store = HumanEvaluationStore(output_path)
    json_path = store.save(results)
    csv_path = store.save_csv(results)

    typer.echo(f"\n[bold green]Success![/bold green] Human evaluations saved to:")
    typer.echo(f" - JSON: {json_path}")
    typer.echo(f" - CSV: {csv_path}")


@app.command()
def metrics():
    """List all available metrics with CLI keys."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title="Available Metrics",
        show_header=True,
        header_style="bold blue"
    )
    table.add_column("Metric Name", style="cyan")
    table.add_column("CLI Key", style="magenta")
    table.add_column("Description", style="green")

    table.add_row("BLEU", "bleu", "N-gram overlap evaluation")
    table.add_row(
        "ROUGE",
        "rouge",
        "Unigram, bigram, and longest common subsequence recall"
    )
    table.add_row("BERTScore", "bertscore", "Contextual token semantic similarity")
    table.add_row(
        "Faithfulness",
        "faithfulness",
        "Semantic alignment with provided context"
    )
    table.add_row(
        "Hallucination Detection",
        "hallucination",
        "Contradiction detection using NLI models"
    )
    table.add_row("Toxicity Detection", "toxicity", "Harmful language safety filter")
    table.add_row("Bias Detection", "bias", "Linguistic pattern bias audit")
    table.add_row("Latency Measurement", "latency", "Logs execution performance")
    table.add_row("Cost Estimation", "cost", "Estimates API/token cost")
    table.add_row(
        "LLM-as-a-Judge",
        "judge",
        "Local instruction LLM scoring on correctness, relevance, etc."
    )
    table.add_row(
        "Custom Metric Example",
        "custom",
        "Keyword match score example"
    )

    console.print(table)


@app.command()
def version():
    """Show project version."""
    typer.echo("Version: 1.0.0")


if __name__ == "__main__":
    app()