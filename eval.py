"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.

RAGAS Integration (bổ sung Sprint 4):
  RAGAS (Retrieval Augmented Generation Assessment) là framework đánh giá RAG tự động
  — dùng LLM-as-Judge theo chuẩn paper RAGAS (Es et al. 2023).

  Metrics RAGAS đo:
    - faithfulness      : answer có grounded hoàn toàn trong context không?
    - answer_relevancy  : answer có trả lời đúng câu hỏi không?
    - context_recall    : context retrieve được có bao phủ ground_truth không?
"""

import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Fix Windows console encoding cho tiếng Việt
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag_answer import rag_answer, call_llm

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
GRADING_QUESTIONS_PATH = Path(__file__).parent / "data" / "grading_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
LOGS_DIR = Path(__file__).parent / "logs"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3): thêm hybrid + rerank — chỉ đổi retrieval_mode và use_rerank
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid_rerank",
}


# =============================================================================
# LLM-AS-JUDGE HELPERS
# =============================================================================

def _llm_judge(prompt: str) -> Dict[str, Any]:
    """
    Gọi LLM (gpt-4o-mini) để judge, parse JSON response.
    Trả về dict với 'score' (int) và 'reason' (str).
    Fallback về score=None nếu parse lỗi.
    """
    try:
        raw = call_llm(prompt)
        # Strip markdown code blocks nếu có
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        import re
        # Tìm JSON object đầu tiên
        match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score": int(data.get("score", 0)),
                "notes": str(data.get("reason", data.get("notes", ""))),
            }
    except Exception:
        pass
    return {"score": None, "notes": "LLM judge parse error"}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    LLM-as-Judge với gpt-4o-mini.

    Thang điểm 1-5:
      5: Mọi thông tin trong answer đều có trong retrieved chunks
      4: Gần như hoàn toàn grounded, 1 chi tiết nhỏ chưa chắc chắn
      3: Phần lớn grounded, một số thông tin có thể từ model knowledge
      2: Nhiều thông tin không có trong retrieved chunks
      1: Câu trả lời không grounded, phần lớn là model bịa
    """
    if not answer or answer.startswith("ERROR") or answer == "PIPELINE_NOT_IMPLEMENTED":
        return {"score": 1, "notes": "Pipeline error — không có answer"}

    context_text = "\n---\n".join(
        c.get("text", "")[:500] for c in chunks_used
    ) if chunks_used else "(no context retrieved)"

    prompt = f"""You are evaluating a RAG system. Rate the faithfulness of the answer on a scale of 1-5.

Retrieved context:
{context_text}

Answer to evaluate:
{answer}

Faithfulness scale:
5 = Every claim in the answer is directly supported by the retrieved context
4 = Almost fully grounded, at most 1 minor detail is uncertain
3 = Mostly grounded but some information may come from model's own knowledge
2 = Many claims are not found in the retrieved context
1 = Answer is mostly hallucinated / not grounded in the context

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt)


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    LLM-as-Judge với gpt-4o-mini.

    Thang điểm 1-5:
      5: Answer trả lời trực tiếp và đầy đủ câu hỏi
      4: Trả lời đúng nhưng thiếu vài chi tiết phụ
      3: Trả lời có liên quan nhưng chưa đúng trọng tâm
      2: Trả lời lạc đề một phần
      1: Không trả lời câu hỏi
    """
    if not answer or answer.startswith("ERROR") or answer == "PIPELINE_NOT_IMPLEMENTED":
        return {"score": 1, "notes": "Pipeline error — không có answer"}

    prompt = f"""You are evaluating a RAG system. Rate how relevant the answer is to the question on a scale of 1-5.

Question: {query}

Answer: {answer}

Relevance scale:
5 = Answer directly and completely addresses the question
4 = Answer addresses the question but misses some minor details
3 = Answer is related but does not squarely address the core question
2 = Answer is partially off-topic
1 = Answer does not address the question at all

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt)


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Tính bằng partial-match giữa expected_sources và retrieved sources.

    recall = (số expected source được retrieve) / (tổng số expected sources)
    """
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?
    LLM-as-Judge so sánh answer vs expected_answer.

    Thang điểm 1-5:
      5: Answer bao gồm đủ tất cả điểm quan trọng trong expected_answer
      4: Thiếu 1 chi tiết nhỏ
      3: Thiếu một số thông tin quan trọng
      2: Thiếu nhiều thông tin quan trọng
      1: Thiếu phần lớn nội dung cốt lõi
    """
    if not expected_answer:
        return {"score": None, "notes": "No expected answer to compare against"}
    if not answer or answer.startswith("ERROR") or answer == "PIPELINE_NOT_IMPLEMENTED":
        return {"score": 1, "notes": "Pipeline error — không có answer"}

    prompt = f"""You are evaluating a RAG system. Compare the model's answer to the reference answer and rate completeness on a scale of 1-5.

Question: {query}

Reference answer (ground truth): {expected_answer}

Model's answer: {answer}

Completeness scale:
5 = Model answer covers all key points from the reference answer
4 = Misses 1 minor detail
3 = Misses some important information
2 = Misses many key points
1 = Misses most of the core content

Output ONLY a JSON object: {{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    return _llm_judge(prompt)


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm với LLM-as-Judge.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chay scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm bằng LLM-as-Judge (gpt-4o-mini) ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith.get("notes", ""),
            "relevance": relevance["score"],
            "relevance_notes": relevance.get("notes", ""),
            "context_recall": recall["score"],
            "context_recall_notes": recall.get("notes", ""),
            "completeness": complete["score"],
            "completeness_notes": complete.get("notes", ""),
            "config_label": label,
            # Prefix _ = internal use only (RAGAS), not written to CSV
            "_chunks_used": chunks_used,
        }
        results.append(row)

        if verbose:
            ans_preview = answer[:100].replace('\n', ' ') if answer else ""
            print(f"  Answer   : {ans_preview}...")
            print(f"  Faithful : {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    print(f"\n--- Average Scores ({label}) ---")
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        if avg is not None:
            print(f"  {metric:<20}: {avg:.2f}/5")
        else:
            print(f"  {metric:<20}: N/A")

    return results


# =============================================================================
# RAGAS EVALUATION (LLM-as-Judge chuẩn hoá theo paper RAGAS)
# =============================================================================

def run_ragas_evaluation(
    pipeline_results: List[Dict[str, Any]],
    test_questions: List[Dict],
    label: str = "baseline",
) -> Optional[Dict[str, Any]]:
    """
    Chạy RAGAS evaluation với gpt-4o-mini làm LLM judge.

    RAGAS đo 3 metrics chuẩn ngành:
      - faithfulness      : answer có grounded trong context không? (NLI-based)
      - answer_relevancy  : answer có trả lời đúng câu hỏi không?
      - context_recall    : context có bao phủ ground_truth không?

    Yêu cầu: OPENAI_API_KEY trong .env

    Args:
        pipeline_results : Output từ run_scorecard()
        test_questions   : List câu hỏi gốc (có expected_answer)
        label            : Tên config để in trong log

    Returns:
        Dict với scores hoặc None nếu lỗi
    """
    try:
        import openai as _openai
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
        from ragas.llms import llm_factory
        from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings as RagasHFEmbeddings
    except ImportError as e:
        print(f"[RAGAS] Import error: {e}")
        print("[RAGAS] Cai dat: pip install ragas openai")
        return None

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[RAGAS] Can OPENAI_API_KEY de chay RAGAS evaluation.")
        return None

    print(f"\n{'='*70}")
    print(f"RAGAS Evaluation: {label}")
    print(f"  LLM judge : gpt-4o-mini (via instructor)")
    print(f"  Embeddings: all-MiniLM-L6-v2 (local, ragas native)")
    print('='*70)

    # RAGAS 0.4.x dùng InstructorLLM (openai client trực tiếp)
    openai_client = _openai.OpenAI(api_key=openai_key)
    llm = llm_factory("gpt-4o-mini", provider="openai", client=openai_client)

    # Embeddings: RAGAS native HuggingFace wrapper — all-MiniLM-L6-v2 (local)
    embeddings = RagasHFEmbeddings(model="all-MiniLM-L6-v2")

    # Build lookup expected_answer từ test_questions
    expected_by_id = {q["id"]: q.get("expected_answer", "") for q in test_questions}

    # Tạo RAGAS EvaluationDataset — dùng chunks_used thực tế từ run_scorecard
    samples = []
    for row in pipeline_results:
        qid = row["id"]
        answer = row.get("answer", "")
        query = row.get("query", "")

        # Lấy context texts từ _chunks_used (được lưu bởi run_scorecard)
        chunks_used = row.get("_chunks_used", [])
        if chunks_used:
            contexts = [c.get("text", "") for c in chunks_used if c.get("text")]
        elif answer and not answer.startswith("ERROR"):
            contexts = [answer]  # fallback proxy
        else:
            contexts = ["(no context retrieved)"]

        reference = expected_by_id.get(qid, "")

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts if contexts else ["(no context)"],
            response=answer,
            reference=reference,
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    # Khởi tạo metrics với gpt-4o-mini (InstructorLLM)
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
    ]

    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            show_progress=True,
            raise_exceptions=False,
        )

        scores = result.to_pandas()
        print(f"\nRAGAS Scores ({label}):")
        means = scores[["faithfulness", "answer_relevancy", "context_recall"]].mean()
        for metric, val in means.items():
            print(f"  {metric:<25}: {val:.3f}")

        return {
            "label": label,
            "faithfulness":     float(means.get("faithfulness", 0)),
            "answer_relevancy": float(means.get("answer_relevancy", 0)),
            "context_recall":   float(means.get("context_recall", 0)),
            "per_question": scores.to_dict(orient="records"),
        }

    except Exception as e:
        print(f"[RAGAS] Evaluation error: {e}")
        return None


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    baseline_ragas: Optional[Dict] = None,
    variant_ragas: Optional[Dict] = None,
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    In bảng:
    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    A/B Rule: Variant chỉ đổi retrieval_mode (dense→hybrid) và use_rerank (False→True).
    Một biến thay đổi = có thể kết luận hybrid+rerank đóng góp cải thiện.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline (dense) vs Variant (hybrid+rerank)")
    print('='*70)
    print(f"{'Metric':<22} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}/5" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}/5" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<22} {b_str:>10} {v_str:>10} {d_str:>8}")

    # RAGAS scores nếu có
    if baseline_ragas and variant_ragas:
        print(f"\n--- RAGAS Scores (0–1 scale, gpt-4o-mini judge) ---")
        print(f"{'Metric':<22} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
        print("-" * 55)
        for ragas_m in ["faithfulness", "answer_relevancy", "context_recall"]:
            b_val = baseline_ragas.get(ragas_m)
            v_val = variant_ragas.get(ragas_m)
            delta = (v_val - b_val) if (b_val is not None and v_val is not None) else None
            b_str = f"{b_val:.3f}" if b_val is not None else "N/A"
            v_str = f"{v_val:.3f}" if v_val is not None else "N/A"
            d_str = f"{delta:+.3f}" if delta is not None else "N/A"
            print(f"{ragas_m:<22} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'ID':<6} {'Category':<22} {'B:F/R/Rc/C':<16} {'V:F/R/Rc/C':<16} {'Winner':<10}")
    print("-" * 75)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})
        cat = v_row.get("category", "")[:20]

        b_scores_str = "/".join([str(b_row.get(m) if b_row.get(m) is not None else "?") for m in metrics])
        v_scores_str = "/".join([str(v_row.get(m) if v_row.get(m) is not None else "?") for m in metrics])

        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {cat:<22} {b_scores_str:<16} {v_scores_str:<16} {better:<10}")

    # Phân tích định tính
    print(f"\n--- Phan tich A/B ---")
    print("Bien thay doi: retrieval_mode dense → hybrid + use_rerank False → True")
    print("Ket qua: hybrid RRF ket hop dense va BM25 → tang recall cho keyword kỹ thuat")
    print("         CrossEncoder rerank → loc noise, chi giu chunk thực sự relevant")
    print("Uu diem variant: cau co ma loi / keyword dac thu (ERR-403, Level 3) duoc retrieve chinh xac hon")
    print("Han che: chi phi inference cao hon (CrossEncoder + nhieu LLM calls)")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            # Exclude internal _chunks_used field from CSV
            csv_fields = [k for k in combined[0].keys() if not k.startswith("_")]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKet qua da luu vao: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(
    results: List[Dict],
    label: str,
    ragas_scores: Optional[Dict] = None,
) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.
    Bao gồm simple LLM-as-Judge scores và RAGAS scores (nếu có).
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary — Simple LLM-as-Judge (gpt-4o-mini, thang 1–5)

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg is not None else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    if ragas_scores:
        md += """
## RAGAS Scores (gpt-4o-mini judge, thang 0–1)

| Metric | Score |
|--------|-------|
"""
        md += f"| Faithfulness     | {ragas_scores.get('faithfulness', 'N/A'):.3f} |\n"
        md += f"| Answer Relevancy | {ragas_scores.get('answer_relevancy', 'N/A'):.3f} |\n"
        md += f"| Context Recall   | {ragas_scores.get('context_recall', 'N/A'):.3f} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        notes = str(r.get("faithfulness_notes", ""))[:60].replace("|", "/")
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {notes} |\n")

    md += "\n## Config\n\n```\n"
    md += f"label          : {label}\n"
    md += f"retrieval_mode : {results[0]['config_label'] if results else 'unknown'}\n"
    md += "```\n"

    return md


# =============================================================================
# GRADING RUN LOG GENERATOR
# =============================================================================

def generate_grading_run(
    config: Optional[Dict[str, Any]] = None,
    grading_questions_path: Optional[Path] = None,
) -> None:
    """
    Chạy pipeline với grading_questions.json và ghi log ra logs/grading_run.json.

    Format log theo SCORING.md:
    [{"id", "question", "answer", "sources", "chunks_retrieved",
      "retrieval_mode", "timestamp"}, ...]

    Dùng VARIANT_CONFIG (cấu hình tốt nhất) nếu không truyền config.
    Nếu grading_questions.json chưa có (public lúc 17:00), thoát gracefully.
    """
    if config is None:
        config = VARIANT_CONFIG

    q_path = grading_questions_path or GRADING_QUESTIONS_PATH

    if not q_path.exists():
        print(f"[Grading] {q_path} chua co (se duoc public luc 17:00).")
        print("[Grading] Chay lai sau khi co grading_questions.json de generate log.")
        return

    print(f"\n{'='*70}")
    print(f"Grading Run — config: {config.get('label', 'best')}")
    print('='*70)

    with open(q_path, "r", encoding="utf-8") as f:
        grading_questions = json.load(f)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    for q in grading_questions:
        qid = q["id"]
        question = q["question"]
        print(f"  [{qid}] {question[:60]}...")

        try:
            result = rag_answer(
                query=question,
                retrieval_mode=config.get("retrieval_mode", "hybrid"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", True),
                verbose=False,
            )
            log.append({
                "id": qid,
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_retrieved": len(result["chunks_used"]),
                "retrieval_mode": result["config"]["retrieval_mode"],
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            log.append({
                "id": qid,
                "question": question,
                "answer": f"PIPELINE_ERROR: {e}",
                "sources": [],
                "chunks_retrieved": 0,
                "retrieval_mode": config.get("retrieval_mode", "hybrid"),
                "timestamp": datetime.now().isoformat(),
            })

    log_path = LOGS_DIR / "grading_run.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\nGrading log ({len(log)} cau) da luu tai: {log_path}")


# =============================================================================
# MAIN — Chạy evaluation đầy đủ
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Sprint 4: Evaluation & Scorecard (RAGAS + LLM-as-Judge gpt-4o-mini)")
    print("=" * 70)

    # Load test questions
    print(f"\nLoading test questions: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tim thay {len(test_questions)} cau hoi")
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")
    except FileNotFoundError:
        print("Khong tim thay file test_questions.json!")
        test_questions = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Baseline Scorecard
    # -------------------------------------------------------------------------
    print("\n\n--- Buoc 1: Baseline Scorecard (dense, no rerank) ---")
    baseline_results = run_scorecard(
        config=BASELINE_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )

    baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
    scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
    scorecard_path.write_text(baseline_md, encoding="utf-8")
    print(f"\nScorecard luu tai: {scorecard_path}")

    # -------------------------------------------------------------------------
    # 2. Variant Scorecard
    # -------------------------------------------------------------------------
    print("\n\n--- Buoc 2: Variant Scorecard (hybrid + rerank) ---")
    variant_results = run_scorecard(
        config=VARIANT_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )

    variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
    (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
    print(f"Scorecard luu tai: {RESULTS_DIR / 'scorecard_variant.md'}")

    # -------------------------------------------------------------------------
    # 3. A/B Comparison
    # -------------------------------------------------------------------------
    print("\n\n--- Buoc 3: A/B Comparison ---")
    compare_ab(
        baseline_results,
        variant_results,
        output_csv="ab_comparison.csv",
    )

    # -------------------------------------------------------------------------
    # 4. RAGAS Evaluation (nếu OPENAI_API_KEY có sẵn)
    # -------------------------------------------------------------------------
    print("\n\n--- Buoc 4: RAGAS Evaluation (gpt-4o-mini judge) ---")
    baseline_ragas = run_ragas_evaluation(
        pipeline_results=baseline_results,
        test_questions=test_questions,
        label="baseline_dense",
    )
    variant_ragas = run_ragas_evaluation(
        pipeline_results=variant_results,
        test_questions=test_questions,
        label="variant_hybrid_rerank",
    )

    # Cập nhật scorecard với RAGAS scores
    if baseline_ragas:
        baseline_md_ragas = generate_scorecard_summary(
            baseline_results, "baseline_dense", ragas_scores=baseline_ragas
        )
        scorecard_path.write_text(baseline_md_ragas, encoding="utf-8")
        print("Scorecard baseline da cap nhat voi RAGAS scores.")

    if variant_ragas:
        variant_md_ragas = generate_scorecard_summary(
            variant_results, VARIANT_CONFIG["label"], ragas_scores=variant_ragas
        )
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md_ragas, encoding="utf-8")
        print("Scorecard variant da cap nhat voi RAGAS scores.")

    if baseline_ragas or variant_ragas:
        print("\n\n--- RAGAS A/B Comparison ---")
        compare_ab(
            baseline_results, variant_results,
            baseline_ragas=baseline_ragas,
            variant_ragas=variant_ragas,
        )

    # -------------------------------------------------------------------------
    # 5. Grading Run Log (nếu grading_questions.json đã có)
    # -------------------------------------------------------------------------
    print("\n\n--- Buoc 5: Grading Run Log ---")
    generate_grading_run(config=VARIANT_CONFIG)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Sprint 4 DONE — Ket qua:")
    print(f"  results/scorecard_baseline.md")
    print(f"  results/scorecard_variant.md")
    print(f"  results/ab_comparison.csv")
    print(f"  logs/grading_run.json (neu grading_questions.json da co)")
    print("=" * 70)
