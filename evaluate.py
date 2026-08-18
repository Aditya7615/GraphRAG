"""RAGAS evaluation pipeline for GraphRAG.

Usage:
    from evaluate import RAGASEvaluator
    evaluator = RAGASEvaluator(engine)
    results = evaluator.evaluate(dataset)
"""

import os
import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.run_config import RunConfig

import warnings
warnings.filterwarnings("ignore", message=".*Importing.*from.*ragas.metrics.*is deprecated.*")
from datasets import Dataset

from graphrag import GraphRAG, get_llm

warnings.filterwarnings("ignore")


@dataclass
class EvaluationResult:
    """Container for RAGAS evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: List[Dict[str, Any]]


class RAGASEvaluator:
    """Evaluate GraphRAG pipeline using RAGAS metrics."""

    def __init__(self, engine: GraphRAG):
        self.engine = engine
        self.run_config = RunConfig(timeout=300, max_retries=5, max_workers=4)
        self.llm = LangchainLLMWrapper(get_llm(), run_config=self.run_config)
        self.embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    def prepare_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dataset:
        """Prepare dataset for RAGAS evaluation."""
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, question in enumerate(questions):
            result = self.engine.ask(question)

            dataset_dict["question"].append(question)
            dataset_dict["answer"].append(result["generation"])

            contexts = [
                doc.page_content for doc in result.get("documents", [])[:5]
            ]
            dataset_dict["contexts"].append(contexts)

            if ground_truths and i < len(ground_truths):
                dataset_dict["ground_truth"].append(ground_truths[i])
            else:
                dataset_dict["ground_truth"].append("")

        return Dataset.from_dict(dataset_dict)

    def evaluate_dataset(self, dataset: Dataset) -> EvaluationResult:
        """Run RAGAS evaluation on a prepared dataset."""
        faithfulness.llm = self.llm
        answer_relevancy.llm = self.llm
        context_precision.llm = self.llm
        context_recall.llm = self.llm

        faithfulness.embeddings = self.embeddings
        answer_relevancy.embeddings = self.embeddings
        context_precision.embeddings = self.embeddings
        context_recall.embeddings = self.embeddings

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=self.embeddings,
            run_config=self.run_config,
        )

        details = []
        for i in range(len(dataset)):
            details.append({
                "question": dataset["question"][i],
                "faithfulness": result["faithfulness"][i],
                "answer_relevancy": result["answer_relevancy"][i],
                "context_precision": result["context_precision"][i],
                "context_recall": result["context_recall"][i],
            })

        overall = (
            result["faithfulness"]
            + result["answer_relevancy"]
            + result["context_precision"]
            + result["context_recall"]
        ) / 4

        return EvaluationResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
            overall_score=overall,
            details=details,
        )

    def evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Full evaluation pipeline: prepare dataset and evaluate."""
        dataset = self.prepare_dataset(questions, ground_truths)
        return self.evaluate_dataset(dataset)


def run_evaluation(
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
) -> EvaluationResult:
    """Convenience function to run evaluation with a fresh engine."""
    engine = GraphRAG()
    evaluator = RAGASEvaluator(engine)
    return evaluator.evaluate(questions, ground_truths)
