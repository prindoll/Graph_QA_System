import pytest

from evaluate_fusarium import (
    bleu_score,
    exact_match_score,
    evidence_hit,
    load_fusarium_qa,
    rouge_l_score,
    token_precision_recall_f1,
)


def test_load_fusarium_qa_reads_jsonl_format(tmp_path):
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        '{"id":"QA001","question":"What?","expected_answer":"Answer.","question_type":"fact","difficulty":"easy"}\n',
        encoding="utf-8",
    )

    rows = load_fusarium_qa(qa_path)

    assert rows == [
        {
            "id": "QA001",
            "question": "What?",
            "expected_answer": "Answer.",
            "question_type": "fact",
            "difficulty": "easy",
        }
    ]


def test_load_fusarium_qa_rejects_missing_expected_answer(tmp_path):
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"question":"What?"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected_answer"):
        load_fusarium_qa(qa_path)


def test_answer_metrics_normalize_articles_and_punctuation():
    prediction = "Only six genes support the F1 node."
    expected = "Only 6 of the 19 genes fully support the F1 node."

    assert exact_match_score("The Fusarium", "Fusarium!") == 1.0
    prf = token_precision_recall_f1(prediction, expected)
    assert 0.0 < prf["precision"] <= 1.0
    assert 0.0 < prf["recall"] <= 1.0
    assert 0.0 < prf["f1"] <= 1.0
    assert 0.0 < rouge_l_score(prediction, expected) <= 1.0
    assert 0.0 < bleu_score(prediction, expected) <= 1.0


def test_evidence_hit_matches_section_or_text_preview():
    sources = [
        {
            "title": "data.md",
            "section_title": "Introduction",
            "heading_path": "data.md > Introduction",
            "text_preview": "The abstract discusses F1 and F3 support.",
        }
    ]

    assert evidence_hit("Abstract; RESULTS", sources) == 1.0
    assert evidence_hit("MATERIALS AND METHODS", sources) == 0.0
