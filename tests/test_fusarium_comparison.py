from compare_fusarium_rag_graphrag import _approaches, _comparison_row, _slug


def test_comparison_approaches_use_plain_rag_and_graph_rag():
    approaches = _approaches(rag_mode="basic", graph_mode="auto")

    assert approaches == [
        {"name": "rag", "retrieval_mode": "basic", "use_graph": False},
        {"name": "graphrag", "retrieval_mode": "auto", "use_graph": True},
    ]


def test_comparison_row_flattens_eval_summary():
    report = {
        "summary": {
            "count": 2,
            "errors": 0,
            "overall": {
                "exact_match": 0.5,
                "token_precision": 0.8,
                "token_recall": 0.7,
                "f1_score": 0.74,
                "rouge_l": 0.6,
                "bleu": 0.4,
                "evidence_hit": 1.0,
            },
        },
        "csv_path": "details.csv",
        "json_path": "details.json",
    }

    row = _comparison_row(
        model="gemma2.5",
        approach="graphrag",
        retrieval_mode="auto",
        use_graph=True,
        report=report,
    )

    assert row["model"] == "gemma2.5"
    assert row["approach"] == "graphrag"
    assert row["f1_score"] == 0.74
    assert row["detail_csv"] == "details.csv"


def test_model_slug_is_filesystem_friendly():
    assert _slug("qwen 3.5:latest") == "qwen_3.5_latest"
