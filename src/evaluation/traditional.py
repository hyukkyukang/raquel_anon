"""Shared traditional unlearning evaluation helpers."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy import stats
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.evaluation.raquel import resolve_generation_device
from src.evaluation.traditional_data import (
    EvaluationExample,
    TraditionalEvalConfig,
    load_traditional_examples,
)
from src.training.data.transforms import ANSWER_PREFIX, PROMPT_TEMPLATE
from src.utils.generation import build_greedy_generation_kwargs
from src.utils.logging import get_logger

logger = get_logger(__name__)

DeviceLike = Union[str, torch.device]
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _format_prompt(question: str, answer: str = "") -> str:
    """Format a QA example using the repo's canonical prompt template."""
    return PROMPT_TEMPLATE.format(question=question.strip(), answer=answer.strip())


def _sample_examples(
    examples: Sequence[EvaluationExample],
    sample_num: Optional[int],
    rng: random.Random,
) -> List[EvaluationExample]:
    """Sample a bounded subset of examples while preserving whole-example records."""
    sampled = list(examples)
    if sample_num is None or sample_num <= 0 or sample_num >= len(sampled):
        return sampled
    return rng.sample(sampled, k=sample_num)


def generate_paraphrase(answer: str) -> str:
    """Generate a lightweight placeholder paraphrase for legacy evaluation."""
    return answer.replace(" is ", " was ").replace(" the ", " this ")


def generate_perturbations(
    answer: str,
    num_pert: int = 5,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Generate lightweight placeholder perturbations for legacy evaluation."""
    words = answer.split()
    if not words:
        return ["INCORRECT"] * max(num_pert, 1)

    generator = rng or random
    perturbations: List[str] = []
    for _ in range(max(num_pert, 1)):
        perturbed = words.copy()
        perturbed[generator.randint(0, len(perturbed) - 1)] = "INCORRECT"
        perturbations.append(" ".join(perturbed))
    return perturbations


def compute_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    *,
    device: Optional[DeviceLike] = None,
) -> float:
    """Compute the masked geometric-mean probability of an answer given a question."""
    model_device = resolve_generation_device(model, device)
    full_text = _format_prompt(question, answer)
    prompt_text = f"Question: {question.strip()}{ANSWER_PREFIX}"

    encoded = tokenizer(full_text, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in encoded.items()}
    labels = inputs["input_ids"].clone()

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = min(int(prompt_ids.shape[1]), int(labels.shape[1]))
    labels[:, :prompt_len] = -100

    with torch.inference_mode():
        outputs = model(**inputs, labels=labels)

    return math.exp(-float(outputs.loss.item()))


def compute_truth_ratio(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    *,
    paraphrased_answer: Optional[str] = None,
    perturbed_answers: Optional[List[str]] = None,
    device: Optional[DeviceLike] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Compute the legacy truth-ratio metric for one example."""
    paraphrase = (
        paraphrased_answer
        if paraphrased_answer is not None
        else generate_paraphrase(answer)
    )
    perturbations = (
        perturbed_answers
        if perturbed_answers
        else generate_perturbations(answer, rng=rng)
    )
    p_true = compute_prob(
        model,
        tokenizer,
        question,
        paraphrase,
        device=device,
    )
    p_perts = [
        compute_prob(model, tokenizer, question, pert, device=device)
        for pert in perturbations
    ]
    avg_p_pert = float(np.mean(p_perts)) if p_perts else 0.0
    return p_true / avg_p_pert if avg_p_pert > 0 else float("inf")


def generate_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    *,
    max_new_tokens: int = 64,
    device: Optional[DeviceLike] = None,
) -> str:
    """Generate a deterministic answer for a question."""
    model_device = resolve_generation_device(model, device)
    prompt = f"Question: {question.strip()}{ANSWER_PREFIX}"
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **build_greedy_generation_kwargs(
                model,
                tokenizer,
                input_length=int(inputs["input_ids"].shape[1]),
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
            ),
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def compute_rouge_l_recall(generated: str, reference: str) -> float:
    """Compute ROUGE-L recall between a generated answer and its reference."""
    scores = _ROUGE_SCORER.score(reference, generated)
    return float(scores["rougeL"].recall)


def evaluate_traditional(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_set: Sequence[EvaluationExample],
    retain_set: Sequence[EvaluationExample],
    retain_model: PreTrainedModel,
    retain_tokenizer: PreTrainedTokenizer,
    *,
    config: Optional[TraditionalEvalConfig] = None,
    device: Optional[DeviceLike] = None,
) -> Dict[str, float]:
    """Run the legacy truth-ratio/utility evaluation."""
    eval_config = config or TraditionalEvalConfig()
    rng = random.Random(eval_config.random_seed)

    forget_eval = _sample_examples(forget_set, eval_config.sample_num, rng)
    retain_eval = _sample_examples(retain_set, eval_config.sample_num, rng)

    logger.info(
        "Starting traditional evaluation: forget=%d retain=%d",
        len(forget_eval),
        len(retain_eval),
    )

    forget_truth_ratios: List[float] = []
    baseline_truth_ratios: List[float] = []
    for example in forget_eval:
        forget_truth_ratios.append(
            compute_truth_ratio(
                model,
                tokenizer,
                example["question"],
                example["answer"],
                paraphrased_answer=example.get("paraphrased_answer"),
                perturbed_answers=example.get("perturbed_answer"),
                device=device,
                rng=rng,
            )
        )
        baseline_truth_ratios.append(
            compute_truth_ratio(
                retain_model,
                retain_tokenizer,
                example["question"],
                example["answer"],
                paraphrased_answer=example.get("paraphrased_answer"),
                perturbed_answers=example.get("perturbed_answer"),
                device=device,
                rng=rng,
            )
        )

    if forget_truth_ratios and baseline_truth_ratios:
        ks_stat, p_value = stats.ks_2samp(forget_truth_ratios, baseline_truth_ratios)
        forget_quality_ks_stat = float(ks_stat)
        forget_quality_pvalue = float(p_value)
    else:
        logger.warning(
            "Traditional evaluation received an empty forget split; returning default forget-quality metrics."
        )
        forget_quality_ks_stat = 0.0
        forget_quality_pvalue = 1.0

    utility_scores: List[Tuple[float, float, float]] = []
    for example in retain_eval:
        generated_answer = generate_answer(
            model,
            tokenizer,
            example["question"],
            max_new_tokens=eval_config.generation_max_new_tokens,
            device=device,
        )
        answer_prob = compute_prob(
            model,
            tokenizer,
            example["question"],
            example["answer"],
            device=device,
        )
        rouge_l_recall = compute_rouge_l_recall(generated_answer, example["answer"])
        truth_ratio = compute_truth_ratio(
            model,
            tokenizer,
            example["question"],
            example["answer"],
            paraphrased_answer=example.get("paraphrased_answer"),
            perturbed_answers=example.get("perturbed_answer"),
            device=device,
            rng=rng,
        )
        utility_scores.append((answer_prob, rouge_l_recall, truth_ratio))

    if utility_scores:
        avg_prob, avg_rouge, avg_truth = np.mean(utility_scores, axis=0)
        normalized_scores = [
            min(max(float(score), 0.0), 1.0)
            for score in (avg_prob, avg_rouge, avg_truth)
        ]
        model_utility = (
            0.0
            if any(score == 0.0 for score in normalized_scores)
            else float(statistics.harmonic_mean(normalized_scores))
        )
    else:
        logger.warning(
            "Traditional evaluation received an empty retain split; returning zero utility metrics."
        )
        avg_prob = avg_rouge = avg_truth = 0.0
        model_utility = 0.0

    return {
        "forget_quality_ks_stat": float(forget_quality_ks_stat),
        "forget_quality_pvalue": float(forget_quality_pvalue),
        "model_utility": float(model_utility),
        "avg_prob": float(avg_prob),
        "avg_rouge_l_recall": float(avg_rouge),
        "avg_truth_ratio": float(avg_truth),
        "forget_examples_evaluated": float(len(forget_eval)),
        "retain_examples_evaluated": float(len(retain_eval)),
    }


def evaluate_with_muse(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_set: Sequence[EvaluationExample],
    retain_set: Sequence[EvaluationExample],
    paraphrased_forget_set: Optional[Sequence[EvaluationExample]] = None,
    non_training_set: Optional[Sequence[EvaluationExample]] = None,
    *,
    config: Optional[TraditionalEvalConfig] = None,
    device: Optional[DeviceLike] = None,
) -> Dict[str, Any]:
    """Run MUSE using the same sampling config as the traditional evaluation."""
    from src.evaluation.muse import MUSEEvaluator

    eval_config = config or TraditionalEvalConfig()
    rng = random.Random(eval_config.random_seed)

    if (
        paraphrased_forget_set is not None
        and eval_config.sample_num is not None
        and eval_config.sample_num > 0
        and len(forget_set) == len(paraphrased_forget_set)
        and eval_config.sample_num < len(forget_set)
    ):
        indices = rng.sample(range(len(forget_set)), k=eval_config.sample_num)
        forget_eval = [forget_set[idx] for idx in indices]
        paraphrased_eval: Optional[List[EvaluationExample]] = [
            paraphrased_forget_set[idx] for idx in indices
        ]
    else:
        forget_eval = _sample_examples(forget_set, eval_config.sample_num, rng)
        paraphrased_eval = (
            _sample_examples(paraphrased_forget_set, eval_config.sample_num, rng)
            if paraphrased_forget_set is not None
            else None
        )
        if (
            paraphrased_forget_set is not None
            and eval_config.sample_num is not None
            and eval_config.sample_num > 0
            and len(forget_set) != len(paraphrased_forget_set)
        ):
            logger.warning(
                "Paraphrased forget set length (%d) does not match forget set length (%d); sampling independently.",
                len(paraphrased_forget_set),
                len(forget_set),
            )

    retain_eval = _sample_examples(retain_set, eval_config.sample_num, rng)
    non_training_eval = (
        _sample_examples(non_training_set, eval_config.sample_num, rng)
        if non_training_set is not None
        else None
    )

    evaluator = MUSEEvaluator(
        model,
        tokenizer,
        device=resolve_generation_device(model, device),
    )
    return evaluator.evaluate_all(
        forget_eval,
        retain_eval,
        paraphrased_eval,
        non_training_eval,
    )


def load_dataset(file_path: str) -> List[EvaluationExample]:
    """Backward-compatible alias for traditional example loading."""
    return load_traditional_examples(file_path)


def compute_rouge_l(generated: str, reference: str) -> float:
    """Backward-compatible alias for ROUGE-L recall."""
    return compute_rouge_l_recall(generated, reference)


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_set: Sequence[EvaluationExample],
    retain_set: Sequence[EvaluationExample],
    retain_model: PreTrainedModel,
    retain_tokenizer: PreTrainedTokenizer,
    sample_num: Optional[int] = None,
) -> Dict[str, float]:
    """Backward-compatible alias for the traditional evaluation."""
    return evaluate_traditional(
        model,
        tokenizer,
        forget_set,
        retain_set,
        retain_model,
        retain_tokenizer,
        config=TraditionalEvalConfig(sample_num=sample_num),
    )
