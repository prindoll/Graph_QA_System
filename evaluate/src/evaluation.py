"""Comprehensive RAG evaluation on HotPotQA.

Sections:
  A. Text-based metrics  (EM, F1, ROUGE, BLEU)
  B. Retrieval metrics   (Recall, Precision, MRR, Hit Rate)
  C. LLM-based metrics   (Correctness, Faithfulness, Relevancy)
  D. Analysis & reporting (by type, by level, error analysis, save)
"""

from __future__ import annotations

import json
import logging
import math
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

from config.settings import (
    DATA_DIR,
    EVAL_SAMPLE_SIZE,
    LLM_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


# ── A. Text-based metrics ────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _get_tokens(text: str) -> list[str]:
    """Tokenise normalised text."""
    return _normalize_answer(text).split()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalised strings match exactly, else 0.0."""
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def token_precision_recall_f1(
    prediction: str, ground_truth: str,
) -> dict[str, float]:
    """Compute token-level precision, recall, and F1."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (convenience wrapper)."""
    return token_precision_recall_f1(prediction, ground_truth)["f1"]


_rouge_scorer_instance = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
)


def rouge_scores(prediction: str, ground_truth: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F-measure."""
    scores = _rouge_scorer_instance.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F-measure (convenience wrapper)."""
    return rouge_scores(prediction, ground_truth)["rougeL"]


def _compute_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with +1 smoothing (Chen & Cherry 2014)."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return 0.0

    bp = min(1.0, math.exp(1 - len(gold_tokens) / len(pred_tokens))) if pred_tokens else 0.0
    log_avg = 0.0
    effective_n = 0

    for n in range(1, max_n + 1):
        pred_ngrams = _compute_ngrams(pred_tokens, n)
        gold_ngrams = _compute_ngrams(gold_tokens, n)
        if not pred_ngrams:
            continue
        clipped = sum(min(pred_ngrams[ng], gold_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        precision_n = (clipped + 1) / (total + 1)
        log_avg += math.log(precision_n)
        effective_n += 1

    if effective_n == 0:
        return 0.0
    return bp * math.exp(log_avg / effective_n)


# ── B. Retrieval metrics ─────────────────────────────────────────

def _get_retrieved_sources(docs) -> list[str]:
    """Extract source titles from retrieved documents (preserving order)."""
    return [doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")]


def retrieval_recall(retrieved_docs, supporting_facts_titles: list[str]) -> float:
    """|SF ∩ Retrieved| / |SF|"""
    if not supporting_facts_titles:
        return 0.0
    unique_sf = set(supporting_facts_titles)
    return len(unique_sf & set(_get_retrieved_sources(retrieved_docs))) / len(unique_sf)


def retrieval_precision(retrieved_docs, supporting_facts_titles: list[str]) -> float:
    """|SF ∩ Retrieved| / |Retrieved|"""
    if not supporting_facts_titles:
        return 0.0
    unique_sf = set(supporting_facts_titles)
    sources = _get_retrieved_sources(retrieved_docs)
    if not sources:
        return 0.0
    return sum(1 for s in sources if s in unique_sf) / len(sources)


def retrieval_f1(retrieved_docs, supporting_facts_titles: list[str]) -> float:
    """Harmonic mean of retrieval precision and recall."""
    prec = retrieval_precision(retrieved_docs, supporting_facts_titles)
    rec = retrieval_recall(retrieved_docs, supporting_facts_titles)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mean_reciprocal_rank(retrieved_docs, supporting_facts_titles: list[str]) -> float:
    """1 / rank of the first relevant document."""
    if not supporting_facts_titles:
        return 0.0
    unique_sf = set(supporting_facts_titles)
    for rank, source in enumerate(_get_retrieved_sources(retrieved_docs), start=1):
        if source in unique_sf:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_docs, supporting_facts_titles: list[str], k: int = 5) -> float:
    """1.0 if at least one supporting fact appears in the top-K results."""
    if not supporting_facts_titles:
        return 0.0
    unique_sf = set(supporting_facts_titles)
    return float(any(s in unique_sf for s in _get_retrieved_sources(retrieved_docs)[:k]))


# ── C. LLM-based metrics ──────────────────────────────────────────

def _get_eval_llm():
    """Return a ChatOpenAI instance for evaluation."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL, temperature=0, max_tokens=256, openai_api_key=OPENAI_API_KEY,
    )


def llm_answer_correctness(
    question: str, prediction: str, ground_truth: str, llm=None,
) -> dict[str, Any]:
    """LLM judge: is the prediction correct? Returns {"score": 0-5, "reasoning": "..."}."""
    if llm is None:
        llm = _get_eval_llm()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator. Given a question, a reference answer, "
         "and a predicted answer, evaluate the correctness of the prediction.\n\n"
         "Score from 0 to 5:\n"
         "  5 = Perfectly correct and complete\n"
         "  4 = Mostly correct, minor details missing\n"
         "  3 = Partially correct\n"
         "  2 = Contains some correct info but largely wrong\n"
         "  1 = Mostly incorrect\n"
         "  0 = Completely wrong or irrelevant\n\n"
         'Respond in JSON format: {{"score": <int>, "reasoning": "<brief explanation>"}}'),
        ("human",
         "Question: {question}\n"
         "Reference Answer: {ground_truth}\n"
         "Predicted Answer: {prediction}"),
    ])
    response = (prompt | llm).invoke({
        "question": question, "ground_truth": ground_truth, "prediction": prediction,
    })
    try:
        result = json.loads(response.content)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "reasoning": response.content}


def llm_faithfulness(
    question: str, prediction: str, context: str, llm=None,
) -> dict[str, Any]:
    """LLM judge: is the prediction faithful to the context? Returns {"score": 0-5, ...}."""
    if llm is None:
        llm = _get_eval_llm()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator checking FAITHFULNESS. "
         "Determine if the predicted answer is fully supported by the given context. "
         "A faithful answer only contains claims that can be verified from the context.\n\n"
         "Score from 0 to 5:\n"
         "  5 = Completely faithful, all claims supported by context\n"
         "  4 = Mostly faithful, one minor unsupported detail\n"
         "  3 = Partially faithful, some claims not in context\n"
         "  2 = Mostly unfaithful, significant hallucinations\n"
         "  1 = Almost entirely hallucinated\n"
         "  0 = Completely hallucinated or contradicts context\n\n"
         'Respond in JSON format: {{"score": <int>, "reasoning": "<brief explanation>"}}'),
        ("human",
         "Question: {question}\nContext: {context}\nPredicted Answer: {prediction}"),
    ])
    response = (prompt | llm).invoke({
        "question": question, "context": context[:3000], "prediction": prediction,
    })
    try:
        result = json.loads(response.content)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "reasoning": response.content}


def llm_answer_relevancy(
    question: str, prediction: str, llm=None,
) -> dict[str, Any]:
    """LLM judge: is the prediction relevant to the question? Returns {"score": 0-5, ...}."""
    if llm is None:
        llm = _get_eval_llm()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator checking ANSWER RELEVANCY. "
         "Determine if the predicted answer is relevant to the question asked.\n\n"
         "Score from 0 to 5:\n"
         "  5 = Directly and completely answers the question\n"
         "  4 = Mostly relevant, slight tangent\n"
         "  3 = Partially relevant\n"
         "  2 = Marginally relevant\n"
         "  1 = Mostly irrelevant\n"
         "  0 = Completely irrelevant\n\n"
         'Respond in JSON format: {{"score": <int>, "reasoning": "<brief explanation>"}}'),
        ("human", "Question: {question}\nPredicted Answer: {prediction}"),
    ])
    response = (prompt | llm).invoke({"question": question, "prediction": prediction})
    try:
        result = json.loads(response.content)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "reasoning": response.content}


def llm_context_relevancy(
    question: str, context: str, llm=None,
) -> dict[str, Any]:
    """LLM judge: is the retrieved context relevant to the question? Returns {"score": 0-5, ...}."""
    if llm is None:
        llm = _get_eval_llm()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator checking CONTEXT RELEVANCY. "
         "Determine if the retrieved context passages are relevant to "
         "answering the question.\n\n"
         "Score from 0 to 5:\n"
         "  5 = Highly relevant, contains all needed information\n"
         "  4 = Mostly relevant, most info present\n"
         "  3 = Partially relevant\n"
         "  2 = Marginally relevant, mostly noise\n"
         "  1 = Almost irrelevant\n"
         "  0 = Completely irrelevant\n\n"
         'Respond in JSON format: {{"score": <int>, "reasoning": "<brief explanation>"}}'),
        ("human", "Question: {question}\nRetrieved Context: {context}"),
    ])
    response = (prompt | llm).invoke({"question": question, "context": context[:3000]})
    try:
        result = json.loads(response.content)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "reasoning": response.content}


def llm_answer_similarity(
    prediction: str, ground_truth: str, llm=None,
) -> dict[str, Any]:
    """LLM judge: semantic similarity between predicted and reference answers (0-5)."""
    if llm is None:
        llm = _get_eval_llm()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert evaluator checking SEMANTIC ANSWER SIMILARITY. "
         "Compare a predicted answer with a reference answer. "
         "Focus on meaning equivalence, not exact wording.\n\n"
         "Score from 0 to 5:\n"
         "  5 = Semantically equivalent (same meaning)\n"
         "  4 = Very close meaning, minor detail differences\n"
         "  3 = Partial overlap in meaning\n"
         "  2 = Limited overlap\n"
         "  1 = Almost different meaning\n"
         "  0 = Completely different or contradictory\n\n"
         'Respond in JSON format: {{"score": <int>, "reasoning": "<brief explanation>"}}'),
        ("human",
         "Reference Answer: {ground_truth}\n"
         "Predicted Answer: {prediction}"),
    ])
    response = (prompt | llm).invoke({
        "ground_truth": ground_truth,
        "prediction": prediction,
    })
    try:
        result = json.loads(response.content)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": -1, "reasoning": response.content}


# ── D. Evaluation orchestration & reporting ──────────────────────

def _invoke_rag_chain(rag_chain, retriever, question: str) -> dict[str, Any]:
    """Invoke the RAG chain and return a normalised result dict (including scored sources)."""
    try:
        response = rag_chain.invoke({"input": question})
        if isinstance(response, dict):
            prediction = response.get("answer", "")
            context_docs = response.get("context", [])
        else:
            prediction = str(response)
            context_docs = retriever.invoke(question)
    except Exception:
        prediction = str(rag_chain.invoke(question))
        context_docs = retriever.invoke(question)

    context_text = "\n\n---\n\n".join(
        doc.page_content for doc in context_docs
    ) if context_docs else ""

    # Retrieve docs with similarity scores for the sources output
    scored_sources: list[dict[str, Any]] = []
    try:
        vs = getattr(retriever, "vectorstore", None)
        if vs is not None:
            scored_pairs = vs.similarity_search_with_relevance_scores(question, k=5)
            for doc, score in scored_pairs:
                scored_sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": round(float(score), 6),
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "original_content"},
                })
    except Exception:
        # Fallback: use context_docs without scores
        for doc in context_docs:
            scored_sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": None,
                "metadata": {k: v for k, v in doc.metadata.items() if k != "original_content"},
            })

    return {
        "prediction": prediction,
        "context_docs": context_docs,
        "context_text": context_text,
        "scored_sources": scored_sources,
    }


def _normalize_level(level: Any) -> str:
    """Normalize difficulty labels to: easy / medium / hard / unknown."""
    text = str(level or "").strip().lower()
    if not text:
        return "unknown"

    easy_labels = {"easy", "de"}
    medium_labels = {"medium", "trungbinh", "trung_binh", "trung binh", "average", "normal"}
    hard_labels = {"hard", "kho", "difficult"}

    if text in easy_labels:
        return "easy"
    if text in medium_labels:
        return "medium"
    if text in hard_labels:
        return "hard"
    return text


def evaluate_rag(
    rag_chain,
    retriever,
    qa_pairs: list[dict[str, Any]],
    sample_size: int = EVAL_SAMPLE_SIZE,
    use_llm_eval: bool = False,
) -> pd.DataFrame:
    """Run text + retrieval (and optionally LLM) evaluation over *sample_size* QA pairs."""
    n = min(sample_size, len(qa_pairs))
    eval_pairs = qa_pairs[:n]
    eval_llm = _get_eval_llm() if use_llm_eval else None

    results = []
    for pair in tqdm(eval_pairs, desc="Evaluating RAG"):
        question = pair["question"]
        ground_truth = pair["answer"]
        sf_titles = pair.get("supporting_facts_titles", [])

        rag_result = _invoke_rag_chain(rag_chain, retriever, question)
        prediction = rag_result["prediction"]
        context_docs = rag_result["context_docs"]
        context_text = rag_result["context_text"]

        # Text metrics
        em = exact_match_score(prediction, ground_truth)
        prf = token_precision_recall_f1(prediction, ground_truth)
        rouges = rouge_scores(prediction, ground_truth)
        bleu = bleu_score(prediction, ground_truth)

        # Retrieval metrics
        ret_recall = retrieval_recall(context_docs, sf_titles)
        ret_prec = retrieval_precision(context_docs, sf_titles)
        ret_f1_val = retrieval_f1(context_docs, sf_titles)
        mrr = mean_reciprocal_rank(context_docs, sf_titles)
        hit = hit_rate_at_k(context_docs, sf_titles)

        row: dict[str, Any] = {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "type": pair.get("type", ""),
            "level": _normalize_level(pair.get("level", "")),
            "exact_match": em,
            "token_precision": prf["precision"],
            "token_recall": prf["recall"],
            "f1_score": prf["f1"],
            "rouge1": rouges["rouge1"],
            "rouge2": rouges["rouge2"],
            "rougeL": rouges["rougeL"],
            "bleu": bleu,
            "documents_retrieved": len(context_docs),
            "documents_used": len(context_docs),
            "sources": rag_result.get("scored_sources", []),
            "retrieval_recall": ret_recall,
            "retrieval_precision": ret_prec,
            "retrieval_f1": ret_f1_val,
            "mrr": mrr,
            "hit_rate_at_5": hit,
        }

        # Optional LLM metrics
        if use_llm_eval and eval_llm is not None:
            correctness = llm_answer_correctness(question, prediction, ground_truth, eval_llm)
            faith = llm_faithfulness(question, prediction, context_text, eval_llm)
            relevancy = llm_answer_relevancy(question, prediction, eval_llm)
            ctx_rel = llm_context_relevancy(question, context_text, eval_llm)
            sim = llm_answer_similarity(prediction, ground_truth, eval_llm)
            row.update({
                "llm_correctness": correctness["score"],
                "llm_correctness_reason": correctness["reasoning"],
                "llm_faithfulness": faith["score"],
                "llm_faithfulness_reason": faith["reasoning"],
                "llm_answer_relevancy": relevancy["score"],
                "llm_answer_relevancy_reason": relevancy["reasoning"],
                "llm_context_relevancy": ctx_rel["score"],
                "llm_context_relevancy_reason": ctx_rel["reasoning"],
                "llm_answer_similarity": sim["score"],
                "llm_answer_similarity_reason": sim["reasoning"],
            })

        results.append(row)

    df = pd.DataFrame(results)
    _log_summary(df, use_llm_eval)
    return df


def _log_summary(df: pd.DataFrame, use_llm_eval: bool = False):
    """Log a compact evaluation summary."""
    logger.info("=== RAG Evaluation Summary (%d samples) ===", len(df))
    for label, col in [
        ("EM", "exact_match"), ("F1", "f1_score"),
        ("ROUGE-L", "rougeL"), ("BLEU", "bleu"),
        ("Ret. Recall", "retrieval_recall"), ("MRR", "mrr"),
        ("Hit@5", "hit_rate_at_5"),
    ]:
        if col in df.columns:
            logger.info("  %-15s: %.4f", label, df[col].mean())
    if use_llm_eval:
        for label, col in [
            ("Correct.", "llm_correctness"), ("Faithful.", "llm_faithfulness"),
            ("Ans. Rel.", "llm_answer_relevancy"), ("Ctx. Rel.", "llm_context_relevancy"),
            ("Ans. Sim.", "llm_answer_similarity"),
        ]:
            if col in df.columns:
                valid = df[df[col] >= 0][col]
                if not valid.empty:
                    logger.info("  %-15s: %.2f / 5", label, valid.mean())


def analyze_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics grouped by question type (bridge / comparison)."""
    if "type" not in df.columns:
        return pd.DataFrame()

    df_group = df.copy()
    df_group["type"] = df_group["type"].fillna("").astype(str).str.strip().replace("", "unknown")

    _exclude = ("question", "ground_truth", "prediction", "type", "level",
                "sources", "documents_retrieved", "documents_used")
    metric_cols = [
        c for c in df.columns
        if c not in _exclude
        and not c.endswith("_reason")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df_group.groupby("type")[metric_cols].mean()
    grouped["count"] = df_group.groupby("type").size()
    return grouped


def analyze_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics grouped by difficulty level (easy / medium / hard)."""
    if "level" not in df.columns:
        return pd.DataFrame()

    df_group = df.copy()
    df_group["level"] = df_group["level"].apply(_normalize_level)

    _exclude = ("question", "ground_truth", "prediction", "type", "level",
                "sources", "documents_retrieved", "documents_used")
    metric_cols = [
        c for c in df.columns
        if c not in _exclude
        and not c.endswith("_reason")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df_group.groupby("level")[metric_cols].mean()
    grouped["count"] = df_group.groupby("level").size()
    return grouped


def _grouped_summary_records(df: pd.DataFrame, grouped_by: str) -> list[dict[str, Any]]:
    """Return grouped summaries as JSON-serializable records."""
    if grouped_by == "level":
        grouped = analyze_by_level(df)
    elif grouped_by == "type":
        grouped = analyze_by_type(df)
    else:
        return []

    if grouped.empty:
        return []

    records: list[dict[str, Any]] = []
    metric_columns = [
        "exact_match", "f1_score", "rougeL", "bleu",
        "retrieval_recall", "retrieval_precision", "retrieval_f1", "mrr", "hit_rate_at_5",
        "llm_correctness", "llm_faithfulness", "llm_answer_relevancy", "llm_context_relevancy",
        "llm_answer_similarity",
    ]

    for group_name in grouped.index:
        row = grouped.loc[group_name]
        record: dict[str, Any] = {
            grouped_by: str(group_name),
            "count": int(row.get("count", 0)),
        }
        for col in metric_columns:
            if col in grouped.columns:
                value = float(row.get(col, 0.0))
                if col.startswith("llm_"):
                    record[col] = round(value, 2)
                else:
                    record[col] = round(value, 4)
        records.append(record)

    if grouped_by == "level":
        order = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}
        records.sort(key=lambda item: order.get(item.get("level", ""), 99))
    return records


def error_analysis(df: pd.DataFrame, threshold_f1: float = 0.3) -> pd.DataFrame:
    """Return rows with F1 < threshold and classify error types."""
    errors = df[df["f1_score"] < threshold_f1].copy()

    def _classify(row):
        pred = row["prediction"].lower().strip()
        if not pred or pred in ("i don't know", "i cannot answer"):
            return "no_answer"
        if row.get("retrieval_recall", 0) == 0:
            return "retrieval_miss"
        if row.get("retrieval_recall", 0) < 0.5:
            return "partial_retrieval"
        if row["ground_truth"].lower().strip() in pred:
            return "verbose_answer"
        return "wrong_answer"

    if not errors.empty:
        errors["error_type"] = errors.apply(_classify, axis=1)
    return errors


def save_results(
    df: pd.DataFrame,
    output_dir: Path | str = DATA_DIR,
    prefix: str = "eval",
) -> dict[str, Path]:
    """Persist evaluation results as a single JSON file with per-question detail + sources."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}

    # Build per-question result list (including top-5 sources)
    result_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "prediction": row["prediction"],
            "type": row.get("type", ""),
            "level": _normalize_level(row.get("level", "")),
            "exact_match": row.get("exact_match", 0.0),
            "f1": row.get("f1_score", 0.0),
            "precision": row.get("token_precision", 0.0),
            "recall": row.get("token_recall", 0.0),
            "rouge1": row.get("rouge1", 0.0),
            "rouge2": row.get("rouge2", 0.0),
            "rougeL": row.get("rougeL", 0.0),
            "bleu": row.get("bleu", 0.0),
            "retrieval_recall": row.get("retrieval_recall", 0.0),
            "retrieval_precision": row.get("retrieval_precision", 0.0),
            "retrieval_f1": row.get("retrieval_f1", 0.0),
            "mrr": row.get("mrr", 0.0),
            "hit_rate_at_5": row.get("hit_rate_at_5", 0.0),
            "documents_retrieved": row.get("documents_retrieved", 0),
            "documents_used": row.get("documents_used", 0),
            "sources": row.get("sources", []),
        }
        # Include LLM metrics if present
        for llm_col in ("llm_correctness", "llm_faithfulness",
                        "llm_answer_relevancy", "llm_context_relevancy", "llm_answer_similarity"):
            if llm_col in row and row[llm_col] is not None:
                record[llm_col] = row[llm_col]
                reason_col = f"{llm_col}_reason"
                if reason_col in row and row[reason_col] is not None:
                    record[reason_col] = str(row[reason_col])
        result_records.append(record)

    # Compute overall averages
    metric_cols = [
        c for c in df.columns
        if c not in ("question", "ground_truth", "prediction", "type", "level",
                     "sources", "documents_retrieved", "documents_used")
        and not c.endswith("_reason")
    ]
    averages = {col: round(float(df[col].mean()), 4) for col in metric_cols if col in df.columns}

    # For LLM metrics, ignore failed judge outputs (-1).
    for llm_col in ("llm_correctness", "llm_faithfulness",
                    "llm_answer_relevancy", "llm_context_relevancy", "llm_answer_similarity"):
        if llm_col in df.columns:
            valid = df[df[llm_col] >= 0][llm_col]
            averages[llm_col] = round(float(valid.mean()), 2) if not valid.empty else None

    averages["num_samples"] = len(df)

    has_llm = all(
        col in df.columns for col in (
            "llm_correctness", "llm_faithfulness", "llm_answer_relevancy", "llm_context_relevancy",
            "llm_answer_similarity",
        )
    )

    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": averages,
        "by_level": _grouped_summary_records(df, grouped_by="level"),
        "by_type": _grouped_summary_records(df, grouped_by="type"),
        "llm_evaluation_enabled": has_llm,
        "results": result_records,
    }

    result_path = output_dir / f"{prefix}_results_{ts}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    paths["results"] = result_path

    logger.info("Saved evaluation results: %s", result_path)
    return paths
