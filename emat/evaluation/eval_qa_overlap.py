"""Evaluation script to get prediction scores for overlapped QA pairs in ODQA datasets"""
import argparse
import os
import string

from emat.evaluation.exact_match import exact_match_score, metric_max_over_ground_truths, f1_score
from emat.utils import load_jsonl

ANNOTATIONS = [
    'total',
    'question_overlap',
    'no_question_overlap',
    'answer_overlap',
    'no_answer_overlap',
    'answer_overlap_only',
    'no_overlap',
]

DIRNAME = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATHS = {
    'triviaqa': 'triviaqa-test.qa.csv',
    'naturalquestions': 'nq-test.qa.csv',
    'webquestions': 'webquestions-test.qa.csv',
}
ANNOTATION_PATHS = {
    'triviaqa': 'triviaqa-annotations.jsonl',
    'naturalquestions': 'nq-annotations.jsonl',
    'webquestions': 'webquestions-annotations.jsonl',
}


def preprocess(text: str) -> str:
    exclude = set(string.punctuation)
    exclude.add(" ")
    exclude.add("’")
    return "".join(ch for ch in text if ch not in exclude)


def read_references(fi, sep='\t'):
    def parse_pandas_answer(a_string):
        # to avoid a pandas dependency, deserialize these manually
        try:
            parsed_answers = eval(a_string) if a_string.startswith('[') else eval(a_string.replace('""', '"')[1:-1])
        except:
            parsed_answers = eval(a_string.replace('""', '"').replace('""', '"').replace('""', '"')[1:-1])
        return parsed_answers

    questions, references = [], []
    for i, line in enumerate(open(fi)):
        q, answer_str = line.strip('\n').split(sep)
        questions.append(q)
        refs = parse_pandas_answer(answer_str)
        references.append({'references': refs, 'id': i})
    return questions, references


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f]


def read_predictions(path):
    if path.endswith('json') or path.endswith('.jsonl'):
        return load_jsonl(path)
    else:
        return [{'id': i, 'prediction': pred} for i, pred in enumerate(read_lines(path))]


def _get_scores(answers, refs, fn):
    return [metric_max_over_ground_truths(fn, pred, rs) for pred, rs in zip(answers, refs)]


def get_scores(predictions, references, annotations, annotation_labels=None):
    predictions_map = {p['id']: p for p in predictions}
    references_map = {r['id']: r for r in references}
    annotations_map = {a['id']: a for a in annotations}
    assert predictions_map.keys() == references_map.keys(), 'predictions file doesnt match the gold references file '
    assert predictions_map.keys() == annotations_map.keys(), 'prediction file doesnt match the annotation file '
    assert annotations_map.keys() == references_map.keys(), 'annotations file doesnt match the gold references file '

    annotation_labels = ANNOTATIONS if annotation_labels is None else annotation_labels

    results = {}
    for annotation_label in annotation_labels:
        if annotation_label == 'no_overlap':
            annotation_ids = [
                annotation['id'] for annotation in annotations if
                all(label in annotation['labels'] for label in ['no_question_overlap', 'no_answer_overlap'])
            ]
        else:
            annotation_ids = [
                annotation['id'] for annotation in annotations if annotation_label in annotation['labels']
            ]

        preds = [predictions_map[idd]['prediction'] for idd in annotation_ids]
        refs = [references_map[idd]['references'] for idd in annotation_ids]
        em = _get_scores(preds, refs, exact_match_score)
        f1 = _get_scores(preds, refs, f1_score)
        results[annotation_label] = {
            'exact_match': 100 * sum(em) / len(em),
            'f1_score': 100 * sum(f1) / len(f1),
            'n_examples': len(annotation_ids),
        }

    return results


def _print_score(label, results_dict):
    print('-' * 50)
    print('Label       :', label)
    print('N examples  : ', results_dict['n_examples'])
    print('Exact Match : ', results_dict['exact_match'])
    # print('F1 score    : ', results_dict['f1_score'])


def main(predictions_path, dataset_name, data_dir):
    references_path = os.path.join(data_dir, REFERENCE_PATHS[dataset_name])
    annotations_path = os.path.join(data_dir, ANNOTATION_PATHS[dataset_name])
    if not os.path.exists(references_path):
        raise Exception(' References expected at ' + references_path
                        + ' not found, please download them using the download script (see readme)')
    if not os.path.exists(annotations_path):
        raise Exception(' Annotations expected at ' + annotations_path
                        + ' not found, please download them usiing the download script (see readme)')

    questions, references = read_references(references_path)
    annotations = load_jsonl(annotations_path)

    predictions = read_predictions(predictions_path)
    assert len(predictions) == len(references) == len(annotations)

    # Align the predictions with the references using the questions
    questions = [preprocess(q) for q in questions]
    question_to_id = {q.strip(): qid for qid, q in enumerate(questions)}
    id_to_prediction = {}
    for pred in predictions:
        q = preprocess(pred["question"].strip())
        qid = question_to_id[q]
        id_to_prediction[qid] = {"id": qid, "prediction": pred["prediction"]}
    assert len(id_to_prediction) == len(references)
    aligned_predictions = [id_to_prediction[qid] for qid in range(len(references))]

    scores = get_scores(aligned_predictions, references, annotations)
    for label in ANNOTATIONS:
        _print_score(label, scores[label])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",
                        help="path to predictions txt file, one answer per line. "
                             "Answer order should follow the order in data/{dataset}-test.qa.csv", type=str)
    parser.add_argument('--dataset_name', choices=['naturalquestions', 'triviaqa', 'webquestions'], type=str,
                        help='name of datset to evaluate on')
    parser.add_argument('--data_dir', default="data/qa-overlap", type=str, help='directory of the annotated data')

    args = parser.parse_args()
    main(args.predictions, args.dataset_name, args.data_dir)
