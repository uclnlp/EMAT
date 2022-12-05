# coding=utf-8
import re
import string
import unicodedata
from collections import Counter

import datasets

_CITATION = """\
@inproceedings{rajpurkar-etal-2016-squad,
    title = "{SQ}u{AD}: 100,000+ Questions for Machine Comprehension of Text",
    author = "Rajpurkar, Pranav  and
      Zhang, Jian  and
      Lopyrev, Konstantin  and
      Liang, Percy",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D16-1264",
    doi = "10.18653/v1/D16-1264",
    pages = "2383--2392",
}
@inproceedings{lee-etal-2019-latent,
    title = "Latent Retrieval for Weakly Supervised Open Domain Question Answering",
    author = "Lee, Kenton  and
      Chang, Ming-Wei  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1612",
    doi = "10.18653/v1/P19-1612",
    pages = "6086--6096",
}
"""

_DESCRIPTION = """\
Exact match score for Open-domain Question Answering. 
This metric measures the percentage of predictions that match any one of the ground truth answers exactly.
"""

_KWARGS_DESCRIPTION = """
Calculates the percentage of predictions that match any one of the ground truth answers exactly.
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a list of strings with tokens separated by spaces.
Returns:
    em: description of the first score,
Examples:
    >>> em_metric = datasets.load_metric("exact_match")
    >>> results = em_metric.compute(references=[["apple", "orange"], ["banana"]], predictions=["apple", "pear"])
    >>> print(results)
    {'em': 0.5}
"""


def normalize_answer(s):
    """Normalize answer."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, ground_truth):
    try:
        regex = re.compile(ground_truth, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
        return regex.match(prediction) is not None
    except re.error:
        return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ExactMatch(datasets.Metric):
    """Exact match (EM) metric for Open-domain Question Answering."""

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Sequence(datasets.Value('string')),
            }),
            homepage="https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html",
            codebase_urls=[
                "https://github.com/google-research/language/blob/58f5dc33a99d168a71586d64ffb7648a0f33b49a/language/orqa/utils/eval_utils.py#L23"],
            reference_urls=["https://arxiv.org/pdf/1606.05250.pdf"]
        )

    def _compute(self, predictions, references, is_regex=False):
        match_fn = regex_match_score if is_regex else exact_match_score
        em_score = sum(metric_max_over_ground_truths(match_fn, i, j) for i, j in zip(predictions, references)) / len(
            predictions)

        return {"em": em_score}
