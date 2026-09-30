from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    from .config import cfg, ensure_dir, resolve_path
except ImportError:
    from config import cfg, ensure_dir, resolve_path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_score(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meteor_lite_score(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    matches = sum(min(pred_counter[token], ref_counter[token]) for token in pred_counter)
    if matches == 0:
        return 0.0

    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10 * precision * recall) / max(1e-8, recall + 9 * precision)

    ref_positions = defaultdict(list)
    for idx, token in enumerate(ref_tokens):
        ref_positions[token].append(idx)

    matched_positions: List[int] = []
    used = set()
    for token in pred_tokens:
        candidates = ref_positions.get(token, [])
        pos = next((candidate for candidate in candidates if candidate not in used), None)
        if pos is not None:
            matched_positions.append(pos)
            used.add(pos)

    if not matched_positions:
        return 0.0

    chunks = 1
    for idx in range(1, len(matched_positions)):
        if matched_positions[idx] != matched_positions[idx - 1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return (1 - penalty) * f_mean


def _cosine_similarity(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for key, value in vec_a.items():
        dot += value * vec_b.get(key, 0.0)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _document_frequency(references: Sequence[Sequence[str]]) -> Dict[int, Counter]:
    doc_freq: Dict[int, Counter] = {n: Counter() for n in range(1, 5)}
    for refs in references:
        for ref in refs:
            ref_tokens = tokenize(ref)
            for n in range(1, 5):
                doc_freq[n].update(set(ngrams(ref_tokens, n)))
    return doc_freq


def _tfidf_vector(tokens: Sequence[str], n: int, doc_freq: Dict[int, Counter], num_docs: int) -> Dict[Tuple[str, ...], float]:
    grams = ngrams(tokens, n)
    counts = Counter(grams)
    total = sum(counts.values())
    if total == 0:
        return {}
    vector: Dict[Tuple[str, ...], float] = {}
    for gram, count in counts.items():
        idf = math.log((num_docs + 1.0) / (1.0 + doc_freq[n].get(gram, 0)))
        vector[gram] = (count / total) * idf
    return vector


def cider_lite_score(pred_tokens: Sequence[str], ref_token_lists: Sequence[Sequence[str]], doc_freq: Dict[int, Counter], num_docs: int) -> float:
    if not pred_tokens:
        return 0.0
    score = 0.0
    for n in range(1, 5):
        pred_vec = _tfidf_vector(pred_tokens, n, doc_freq, num_docs)
        ref_scores = []
        for ref_tokens in ref_token_lists:
            ref_vec = _tfidf_vector(ref_tokens, n, doc_freq, num_docs)
            ref_scores.append(_cosine_similarity(pred_vec, ref_vec))
        score += sum(ref_scores) / max(1, len(ref_scores))
    return 10.0 * score / 4.0


def compute_caption_metrics(predictions: Sequence[str], references: Sequence[Sequence[str]]) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    smoothie = SmoothingFunction().method1
    doc_freq = _document_frequency(references)
    num_docs = sum(len(refs) for refs in references)

    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    cider_scores = []
    for prediction, refs in zip(predictions, references):
        pred_tokens = tokenize(prediction)
        ref_token_lists = [tokenize(ref) for ref in refs if ref]
        if not ref_token_lists:
            ref_token_lists = [[]]

        bleu_scores.append(
            sentence_bleu(
                ref_token_lists,
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie,
            )
        )
        meteor_scores.append(max(meteor_lite_score(pred_tokens, ref_tokens) for ref_tokens in ref_token_lists))
        rouge_scores.append(max(rouge_l_score(pred_tokens, ref_tokens) for ref_tokens in ref_token_lists))
        cider_scores.append(cider_lite_score(pred_tokens, ref_token_lists, doc_freq, num_docs))

    return {
        "BLEU-4": float(sum(bleu_scores) / max(1, len(bleu_scores))),
        "METEOR": float(sum(meteor_scores) / max(1, len(meteor_scores))),
        "ROUGE-L": float(sum(rouge_scores) / max(1, len(rouge_scores))),
        "CIDEr": float(sum(cider_scores) / max(1, len(cider_scores))),
    }


def load_predictions_file(path: Path) -> Tuple[List[str], List[List[str]]]:
    payload = json.loads(path.read_text())
    predictions = [item["prediction"] for item in payload]
    references = [item["references"] for item in payload]
    return predictions, references


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated captions.")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    predictions, references = load_predictions_file(pred_path)
    metrics = compute_caption_metrics(predictions, references)

    print(json.dumps(metrics, indent=2))
    out_json = Path(args.out_json) if args.out_json else ensure_dir(cfg.METRICS_DIR) / f"{pred_path.stem}_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[Save] metrics -> {out_json}")


if __name__ == "__main__":
    main()
