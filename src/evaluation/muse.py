"""
MUSE: Machine Unlearning Six-way Evaluation for Language Models
Implementation of the six evaluation metrics from the MUSE paper.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.raquel import DeviceLike, resolve_generation_device
from src.training.data.utils import load_json_dataset
from src.training.data.transforms import ANSWER_PREFIX
from src.utils.generation import build_greedy_generation_kwargs
from src.utils.logging import get_logger

logger = get_logger(__name__)

MUSEExample = Dict[str, Any]
MUSECoreResults = Dict[str, Any]


class MUSEEvaluator:
    """
    MUSE Six-way Evaluation Framework for Machine Unlearning

    Metrics:
    1. No Verbatim Memorization
    2. No Knowledge Memorization
    3. No Privacy Leakage
    4. Utility Preservation
    5. Scalability
    6. Sustainability
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[DeviceLike] = None,
        default_max_new_tokens: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = resolve_generation_device(model, device)
        self.default_max_new_tokens = max(int(default_max_new_tokens), 1)
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    @staticmethod
    def _format_prompt(question: str) -> str:
        """Format questions using the repo's canonical QA prompt contract."""
        return f"Question: {question.strip()}{ANSWER_PREFIX}"

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate model response for a given prompt."""
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generation_tokens = (
            self.default_max_new_tokens
            if max_new_tokens is None
            else max(int(max_new_tokens), 1)
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **build_greedy_generation_kwargs(
                    self.model,
                    self.tokenizer,
                    input_length=int(inputs["input_ids"].shape[1]),
                    max_new_tokens=generation_tokens,
                ),
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    @staticmethod
    def _summarize_scores(scores: List[float], prefix: str) -> Dict[str, Any]:
        """Return JSON-serializable summary stats for a list of scores."""
        return {
            f"{prefix}_mean": float(np.mean(scores)) if scores else 0.0,
            f"{prefix}_std": float(np.std(scores)) if scores else 0.0,
            f"{prefix}_scores": [float(score) for score in scores],
        }

    def evaluate_verbatim_memorization(
        self, forget_examples: List[Dict]
    ) -> Dict[str, float]:
        """
        Metric 1: No Verbatim Memorization
        Measures if model can reproduce exact text from forget set.
        Lower ROUGE-L scores indicate better unlearning.
        """
        rouge_scores = []

        for example in tqdm(forget_examples, desc="Evaluating verbatim memorization"):
            question = example.get("question", "")
            ground_truth = example.get("answer", "")

            # Generate response
            response = self.generate_response(question)

            # Calculate ROUGE-L score
            score = self.rouge_scorer.score(ground_truth, response)
            rouge_scores.append(score["rougeL"].fmeasure)

        return self._summarize_scores(rouge_scores, "verbatim_rouge")

    def evaluate_knowledge_memorization(
        self, forget_examples: List[Dict], paraphrased_examples: List[Dict]
    ) -> Dict[str, float]:
        """
        Metric 2: No Knowledge Memorization
        Tests paraphrased versions of forget set questions.
        Lower ROUGE-L scores indicate successful knowledge removal.
        """
        rouge_scores = []
        if len(forget_examples) != len(paraphrased_examples):
            logger.warning(
                "Knowledge memorization eval received mismatched datasets "
                "(forget=%d, paraphrased=%d); truncating to the shorter length.",
                len(forget_examples),
                len(paraphrased_examples),
            )

        for orig_example, para_example in tqdm(
            zip(forget_examples, paraphrased_examples),
            desc="Evaluating knowledge memorization",
        ):
            paraphrased_question = para_example.get("question", "")
            original_answer = orig_example.get("answer", "")

            # Generate response to paraphrased question
            response = self.generate_response(paraphrased_question)

            # Calculate ROUGE-L score against original answer
            score = self.rouge_scorer.score(original_answer, response)
            rouge_scores.append(score["rougeL"].fmeasure)

        return self._summarize_scores(rouge_scores, "knowledge_rouge")

    def evaluate_privacy_leakage(
        self, forget_examples: List[Dict], non_training_examples: List[Dict]
    ) -> Dict[str, float]:
        """
        Metric 3: No Privacy Leakage
        Uses Membership Inference Attack to detect training data.
        Lower MIA success rate indicates better privacy protection.
        """

        def get_confidence_score(example):
            question = str(example.get("question", "")).strip()
            answer = str(example.get("answer", "")).strip()
            prompt = self._format_prompt(question)
            full_text = prompt + answer

            tokenized = self.tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=1024
            )
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            )["input_ids"]
            labels = tokenized["input_ids"].clone()
            prompt_len = min(prompt_ids.shape[1], labels.shape[1])
            labels[:, :prompt_len] = -100

            inputs = {k: v.to(self.device) for k, v in tokenized.items()}
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=labels)
                # Return negative masked loss so higher means more confident on the answer.
                return -float(outputs.loss.item())

        # Get confidence scores
        forget_scores = []
        for example in tqdm(forget_examples, desc="Computing forget set confidence"):
            forget_scores.append(get_confidence_score(example))

        non_training_scores = []
        for example in tqdm(
            non_training_examples, desc="Computing non-training confidence"
        ):
            non_training_scores.append(get_confidence_score(example))

        # MIA: classify based on a threshold calibrated on non-training data only.
        all_scores = forget_scores + non_training_scores
        all_labels = [1] * len(forget_scores) + [0] * len(non_training_scores)

        threshold = np.median(non_training_scores)
        predictions = [1 if score > threshold else 0 for score in all_scores]

        # Calculate accuracy (MIA success rate)
        mia_accuracy = np.mean(
            [pred == label for pred, label in zip(predictions, all_labels)]
        )

        return {
            "mia_accuracy": float(mia_accuracy),
            "forget_confidence_mean": float(np.mean(forget_scores)),
            "non_training_confidence_mean": float(np.mean(non_training_scores)),
            "confidence_threshold": float(threshold),
            "confidence_threshold_source": "non_training_median",
        }

    def evaluate_utility_preservation(
        self,
        retain_examples: List[Dict],
        baseline_responses: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Metric 4: Utility Preservation
        Ensures model performance on retain set remains intact.
        Higher retain set performance indicates better utility preservation.
        """
        if baseline_responses is None:
            # Generate responses for retain set
            responses = []
            for example in tqdm(
                retain_examples, desc="Evaluating utility preservation"
            ):
                question = example.get("question", "")
                response = self.generate_response(question)
                responses.append(response)
        else:
            responses = baseline_responses

        # Calculate ROUGE-L scores against ground truth
        rouge_scores = []
        for example, response in zip(retain_examples, responses):
            ground_truth = example.get("answer", "")
            score = self.rouge_scorer.score(ground_truth, response)
            rouge_scores.append(score["rougeL"].fmeasure)

        return self._summarize_scores(rouge_scores, "utility_rouge")

    def evaluate_scalability(
        self, forget_examples: List[Dict], subset_sizes: List[int] = [10, 50, 100, 200]
    ) -> Dict[str, Dict]:
        """
        Metric 5: Scalability
        Tests unlearning effectiveness across different forget set sizes.
        """
        scalability_results = {}

        for size in subset_sizes:
            if size > len(forget_examples):
                continue

            subset = forget_examples[:size]
            verbatim_results = self.evaluate_verbatim_memorization(subset)

            scalability_results[f"size_{size}"] = {
                "verbatim_rouge_mean": verbatim_results["verbatim_rouge_mean"],
                "subset_size": size,
            }

        return scalability_results

    def evaluate_sustainability(
        self, forget_examples: List[Dict], additional_training_steps: int = 100
    ) -> Dict[str, float]:
        """
        Metric 6: Sustainability
        Tests robustness against continued training after unlearning.
        This is a placeholder - actual implementation would require retraining.
        """
        # For now, just evaluate current state
        # In practice, this would involve:
        # 1. Fine-tune model on general data for additional_training_steps
        # 2. Re-evaluate verbatim memorization
        # 3. Compare with pre-training results

        current_results = self.evaluate_verbatim_memorization(forget_examples)

        return {
            "sustainability_rouge_mean": current_results["verbatim_rouge_mean"],
            "additional_training_steps": additional_training_steps,
            "note": "Placeholder implementation - requires actual retraining",
        }

    def evaluate_all(
        self,
        forget_examples: List[Dict],
        retain_examples: List[Dict],
        paraphrased_examples: Optional[List[Dict]] = None,
        non_training_examples: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Run all six MUSE evaluation metrics.
        """
        results = {}

        # Metric 1: Verbatim Memorization
        logger.info("Evaluating verbatim memorization...")
        results["verbatim"] = self.evaluate_verbatim_memorization(forget_examples)

        # Metric 2: Knowledge Memorization (if paraphrased examples available)
        if paraphrased_examples:
            logger.info("Evaluating knowledge memorization...")
            results["knowledge"] = self.evaluate_knowledge_memorization(
                forget_examples, paraphrased_examples
            )

        # Metric 3: Privacy Leakage (if non-training examples available)
        if non_training_examples:
            logger.info("Evaluating privacy leakage...")
            results["privacy"] = self.evaluate_privacy_leakage(
                forget_examples, non_training_examples
            )

        # Metric 4: Utility Preservation
        logger.info("Evaluating utility preservation...")
        results["utility"] = self.evaluate_utility_preservation(retain_examples)

        # Metric 5: Scalability
        logger.info("Evaluating scalability...")
        results["scalability"] = self.evaluate_scalability(forget_examples)

        # Metric 6: Sustainability
        logger.info("Evaluating sustainability...")
        results["sustainability"] = self.evaluate_sustainability(forget_examples)

        return results


def evaluate_muse_core_metrics(
    evaluator: MUSEEvaluator,
    forget_examples: List[MUSEExample],
    retain_examples: List[MUSEExample],
    paraphrased_examples: Optional[List[MUSEExample]] = None,
    non_training_examples: Optional[List[MUSEExample]] = None,
) -> MUSECoreResults:
    """Run the shared MUSE core metric set used by scripts and callbacks."""
    results: MUSECoreResults = {}

    logger.info("Metric 1: Evaluating Verbatim Memorization...")
    results["verbatim"] = evaluator.evaluate_verbatim_memorization(forget_examples)
    logger.info(
        "  ✓ Verbatim ROUGE-L: %.4f",
        results["verbatim"]["verbatim_rouge_mean"],
    )

    if paraphrased_examples:
        logger.info("Metric 2: Evaluating Knowledge Memorization...")
        results["knowledge"] = evaluator.evaluate_knowledge_memorization(
            forget_examples,
            paraphrased_examples,
        )
        logger.info(
            "  ✓ Knowledge ROUGE-L: %.4f",
            results["knowledge"]["knowledge_rouge_mean"],
        )
    else:
        logger.info("Metric 2: Skipped (no paraphrased data provided)")

    if non_training_examples:
        logger.info("Metric 3: Evaluating Privacy Leakage (MIA)...")
        results["privacy"] = evaluator.evaluate_privacy_leakage(
            forget_examples,
            non_training_examples,
        )
        logger.info(
            "  ✓ MIA Accuracy: %.4f",
            results["privacy"]["mia_accuracy"],
        )
    else:
        logger.info("Metric 3: Skipped (no non-training data provided)")

    logger.info("Metric 4: Evaluating Utility Preservation...")
    results["utility"] = evaluator.evaluate_utility_preservation(retain_examples)
    logger.info(
        "  ✓ Utility ROUGE-L: %.4f",
        results["utility"]["utility_rouge_mean"],
    )

    return results


def build_muse_core_metric_logs(
    results: MUSECoreResults,
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Flatten shared MUSE core metrics for experiment loggers."""
    normalized_prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix
    metrics_to_log: Dict[str, float] = {}

    if "verbatim" in results:
        metrics_to_log[f"{normalized_prefix}verbatim_rouge"] = float(
            results["verbatim"]["verbatim_rouge_mean"]
        )
    if "knowledge" in results:
        metrics_to_log[f"{normalized_prefix}knowledge_rouge"] = float(
            results["knowledge"]["knowledge_rouge_mean"]
        )
    if "privacy" in results:
        metrics_to_log[f"{normalized_prefix}mia_accuracy"] = float(
            results["privacy"]["mia_accuracy"]
        )
    if "utility" in results:
        metrics_to_log[f"{normalized_prefix}utility_rouge"] = float(
            results["utility"]["utility_rouge_mean"]
        )
    return metrics_to_log


def load_evaluation_data(
    forget_path: str,
    retain_path: str,
    paraphrased_path: Optional[str] = None,
    non_training_path: Optional[str] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
]:
    """Load evaluation datasets for MUSE metrics."""
    forget_examples = load_json_dataset(forget_path)
    retain_examples = load_json_dataset(retain_path)

    paraphrased_examples: Optional[List[Dict[str, Any]]] = None
    if paraphrased_path:
        paraphrased_examples = load_json_dataset(paraphrased_path)

    non_training_examples: Optional[List[Dict[str, Any]]] = None
    if non_training_path:
        non_training_examples = load_json_dataset(non_training_path)

    return forget_examples, retain_examples, paraphrased_examples, non_training_examples


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run MUSE evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--forget_data", type=str, required=True, help="Path to forget examples"
    )
    parser.add_argument(
        "--retain_data", type=str, required=True, help="Path to retain examples"
    )
    parser.add_argument(
        "--paraphrased_data", type=str, help="Path to paraphrased examples"
    )
    parser.add_argument(
        "--non_training_data", type=str, help="Path to non-training examples"
    )
    parser.add_argument(
        "--output_path", type=str, default="muse_results.json", help="Output path"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation data
    forget_examples, retain_examples, paraphrased_examples, non_training_examples = (
        load_evaluation_data(
            args.forget_data,
            args.retain_data,
            args.paraphrased_data,
            args.non_training_data,
        )
    )

    # Run evaluation
    evaluator = MUSEEvaluator(model, tokenizer)
    results = evaluator.evaluate_all(
        forget_examples, retain_examples, paraphrased_examples, non_training_examples
    )

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"MUSE evaluation results saved to {args.output_path}")
