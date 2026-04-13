# python -X utf8 rag_answer.py
"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Đảm bảo stdout UTF-8 trên Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

# Nếu chunk tốt nhất có score thấp hơn ngưỡng này → abstain, không gọi LLM
# Dense/sparse: [0,1] — 0.3 là ngưỡng thực nghiệm hợp lý
# Hybrid RRF:   giá trị nhỏ hơn (~0.01), dùng ABSTAIN_THRESHOLD_HYBRID (mặc định 0.0)
# Rerank:       logit score, thường [-5, 5], dùng ABSTAIN_THRESHOLD_RERANK (mặc định 0.0)
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.3"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity score

    TODO Sprint 2:
    1. Embed query bằng cùng model đã dùng khi index (xem index.py)
    2. Query ChromaDB với embedding đó
    3. Trả về kết quả kèm score

    Gợi ý:
        import chromadb
        from index import get_embedding, CHROMA_DB_DIR

        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")

        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Lưu ý: distances trong ChromaDB cosine = 1 - similarity
        # Score = 1 - distance
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        raise RuntimeError(
            "Collection 'rag_lab' trống. Hãy chạy 'python index.py' trước để build index."
        )

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": 1 - dist,  # cosine distance → similarity
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa
    """
    import chromadb
    from rank_bm25 import BM25Okapi
    from index import CHROMA_DB_DIR

    # Load toàn bộ corpus từ ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        raise RuntimeError(
            "Collection 'rag_lab' trống. Hãy chạy 'python index.py' trước để build index."
        )

    all_data = collection.get(include=["documents", "metadatas"])

    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    # Tokenize corpus (lowercase, split)
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Query
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k index theo score giảm dần, chỉ giữ chunk có raw score > 0
    top_indices = sorted(
        [i for i in range(len(scores)) if scores[i] > 0],
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    # Normalize về [0, 1] dựa trên max raw score của TOÀN corpus
    # để score phản ánh mức độ match tuyệt đối, không phải tương đối trong top-k
    max_possible = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

    chunks = []
    for idx in top_indices:
        chunks.append({
            "text": all_docs[idx],
            "metadata": all_metas[idx],
            "score": float(scores[idx]) / max_possible,  # [0,1] so sánh được với dense
        })

    return chunks


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản

    Args:
        dense_weight: Trọng số cho dense score (0-1)
        sparse_weight: Trọng số cho sparse score (0-1)

    TODO Sprint 3 (nếu chọn hybrid):
    1. Chạy retrieve_dense() → dense_results
    2. Chạy retrieve_sparse() → sparse_results
    3. Merge bằng RRF:
       RRF_score(doc) = dense_weight * (1 / (60 + dense_rank)) +
                        sparse_weight * (1 / (60 + sparse_rank))
       60 là hằng số RRF tiêu chuẩn
    4. Sort theo RRF score giảm dần, trả về top_k

    Khi nào dùng hybrid (từ slide):
    - Corpus có cả câu tự nhiên VÀ tên riêng, mã lỗi, điều khoản
    - Query như "Approval Matrix" khi doc đổi tên thành "Access Control SOP"
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Xây dựng map: text → chunk (dùng text làm key để dedup)
    chunk_map: Dict[str, Dict[str, Any]] = {}
    for chunk in dense_results + sparse_results:
        key = chunk["text"]
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Tính rank từ mỗi danh sách (rank bắt đầu từ 1)
    dense_rank: Dict[str, int] = {c["text"]: i + 1 for i, c in enumerate(dense_results)}
    sparse_rank: Dict[str, int] = {c["text"]: i + 1 for i, c in enumerate(sparse_results)}

    # RRF score: dense_weight * 1/(60+rank_d) + sparse_weight * 1/(60+rank_s)
    # Nếu không xuất hiện trong một danh sách → dùng rank = top_k (penalty)
    rrf_scores: Dict[str, float] = {}
    for text in chunk_map:
        rd = dense_rank.get(text, top_k)
        rs = sparse_rank.get(text, top_k)
        rrf_scores[text] = (
            dense_weight * (1.0 / (60 + rd)) +
            sparse_weight * (1.0 / (60 + rs))
        )

    # Sort theo RRF score giảm dần
    sorted_texts = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

    results = []
    for text in sorted_texts[:top_k]:
        chunk = chunk_map[text].copy()
        chunk["score"] = rrf_scores[text]  # ghi đè score bằng RRF score
        results.append(chunk)

    return results


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.

    Cross-encoder: chấm lại "chunk nào thực sự trả lời câu hỏi này?"
    MMR (Maximal Marginal Relevance): giữ relevance nhưng giảm trùng lặp

    Funnel logic (từ slide):
      Search rộng (top-20) → Rerank (top-6) → Select (top-3)

    TODO Sprint 3 (nếu chọn rerank):
    Option A — Cross-encoder:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]

    Option B — Rerank bằng LLM (đơn giản hơn nhưng tốn token):
        Gửi list chunks cho LLM, yêu cầu chọn top_k relevant nhất

    Khi nào dùng rerank:
    - Dense/hybrid trả về nhiều chunk nhưng có noise
    - Muốn chắc chắn chỉ 3-5 chunk tốt nhất vào prompt
    """
    from sentence_transformers import CrossEncoder

    # Lazy-load model (chỉ tải lần đầu, cache trong module scope)
    if not hasattr(rerank, "_model"):
        rerank._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    model: CrossEncoder = rerank._model

    # Tạo pairs [query, chunk_text] để cross-encoder chấm điểm
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)  # numpy array

    # Gắn rerank score và sort giảm dần
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for chunk, score in ranked[:top_k]:
        chunk = chunk.copy()
        chunk["score"] = float(score)  # ghi đè bằng cross-encoder score
        results.append(chunk)

    return results


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành 2-3 sub-queries
      - "hyde": Sinh câu trả lời giả (hypothetical document) để embed thay query

    TODO Sprint 3 (nếu chọn query transformation):
    Gọi LLM với prompt phù hợp với từng strategy.

    Ví dụ expansion prompt:
        "Given the query: '{query}'
         Generate 2-3 alternative phrasings or related terms in Vietnamese.
         Output as JSON array of strings."

    Ví dụ decomposition:
        "Break down this complex query into 2-3 simpler sub-queries: '{query}'
         Output as JSON array."

    Khi nào dùng:
    - Expansion: query dùng alias/tên cũ (ví dụ: "Approval Matrix" → "Access Control SOP")
    - Decomposition: query hỏi nhiều thứ một lúc
    - HyDE: query mơ hồ, search theo nghĩa không hiệu quả
    """
    import json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if strategy == "expansion":
        system_msg = (
            "You are a search query assistant. "
            "Given a user query, generate 2-3 alternative phrasings or closely related terms "
            "that help retrieve the same information. "
            "Output ONLY a JSON array of strings, no explanation."
        )
        user_msg = f"Query: {query}"

    elif strategy == "decomposition":
        system_msg = (
            "You are a search query assistant. "
            "Break down the following complex query into 2-3 simpler, focused sub-queries. "
            "Each sub-query should target a distinct aspect of the original question. "
            "Output ONLY a JSON array of strings, no explanation."
        )
        user_msg = f"Query: {query}"

    elif strategy == "hyde":
        # HyDE: sinh câu trả lời giả → embed câu trả lời thay vì query gốc
        system_msg = (
            "You are a knowledgeable assistant. "
            "Write a concise, factual answer (2-4 sentences) to the following question "
            "as if you were answering from an internal policy document. "
            "Output ONLY the hypothetical answer text, no preamble."
        )
        user_msg = f"Question: {query}"
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=256,
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        # Trả về query gốc + hypothetical document để retrieve cả hai
        return [query, hypothetical_doc]

    else:
        raise ValueError(f"strategy không hợp lệ: {strategy}. Chọn: expansion | decomposition | hyde")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()

    try:
        alternatives = json.loads(raw)
        if not isinstance(alternatives, list):
            raise ValueError("Expected JSON array")
    except (json.JSONDecodeError, ValueError):
        # Fallback: trả về query gốc nếu parse thất bại
        return [query]

    # Luôn bao gồm query gốc để không bỏ sót
    return [query] + [q for q in alternatives if q != query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    Trả về JSON gồm:
      - "answer": câu trả lời với citation [N]
      - "grounded_spans": list các câu/cụm từ NGUYÊN VĂN từ context đã dùng
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source (in brackets like [1]) when using information from a chunk.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

You MUST return a JSON object with exactly these two fields:
{{
  "answer": "<your grounded answer with [N] citations>",
  "grounded_spans": ["<exact phrase or sentence copied verbatim from the context that you relied on>", ...]
}}

Question: {query}

Context:
{context_block}"""
    return prompt


def call_llm(prompt: str) -> Tuple[str, List[str]]:
    """
    Gọi LLM, parse JSON response.

    Returns:
        (answer, grounded_spans)
          - answer: câu trả lời với citation [N]
          - grounded_spans: list câu/cụm từ nguyên văn từ context đã dùng
    """
    import json
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=768,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
        answer = data.get("answer", raw)
        grounded_spans = data.get("grounded_spans", [])
        if not isinstance(grounded_spans, list):
            grounded_spans = []
        # Lọc span rỗng
        grounded_spans = [s for s in grounded_spans if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError):
        answer = raw
        grounded_spans = []
    return answer, grounded_spans


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng

    TODO Sprint 2 — Implement pipeline cơ bản:
    1. Chọn retrieval function dựa theo retrieval_mode
    2. Gọi rerank() nếu use_rerank=True
    3. Truncate về top_k_select chunks
    4. Build context block và grounded prompt
    5. Gọi call_llm() để sinh câu trả lời
    6. Trả về kết quả kèm metadata

    TODO Sprint 3 — Thử các variant:
    - Variant A: đổi retrieval_mode="hybrid"
    - Variant B: bật use_rerank=True
    - Variant C: thêm query transformation trước khi retrieve
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 2.5: Abstain nếu chunk tốt nhất dưới ngưỡng ---
    # Ngưỡng cần phụ thuộc thang điểm của score theo từng mode.
    # - Dense/sparse: cosine/BM25 normalized ~ [0,1]
    # - Hybrid RRF: score rất nhỏ (~0.01)
    # - Rerank: cross-encoder logit có thể âm/dương
    if use_rerank:
        abstain_threshold = float(os.getenv("ABSTAIN_THRESHOLD_RERANK", "0.0"))
    elif retrieval_mode == "hybrid":
        abstain_threshold = float(os.getenv("ABSTAIN_THRESHOLD_HYBRID", "0.0"))
    else:
        abstain_threshold = ABSTAIN_THRESHOLD

    best_score = candidates[0].get("score", 1.0) if candidates else 0.0
    if not candidates or best_score < abstain_threshold:
        if verbose:
            print(
                f"[RAG] Abstain — best_score={best_score:.3f} < threshold={abstain_threshold} "
                f"(mode={retrieval_mode}, rerank={use_rerank})"
            )
        return {
            "query": query,
            "answer": "Không đủ dữ liệu để trả lời câu hỏi này.",
            "sources": [],
            "chunks_used": [],
            "config": config,
            "abstained": True,
            "best_score": best_score,
        }

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer, grounded_spans = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "grounded_spans": grounded_spans,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
        "abstained": False,
        "best_score": best_score,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "hybrid"]  # Thêm "sparse" sau khi implement

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# GROUNDING HIGHLIGHT HELPERS
# =============================================================================

def _highlight_chunk_html(
    chunk_idx: int,
    chunk: Dict[str, Any],
    grounded_spans: List[str],
) -> str:
    """
    Render một chunk thành HTML, tô màu vàng những câu/cụm từ
    có trong grounded_spans (nguyên văn từ LLM).

    Màu vàng (#ffe066) = câu đóng góp trực tiếp vào câu trả lời.
    """
    import html as _html

    meta = chunk.get("metadata", {})
    score = chunk.get("score", 0)
    src = meta.get("source", "?")
    sec = meta.get("section", "")
    text = chunk.get("text", "")

    header = f"<b>[{chunk_idx}]</b> <code>{_html.escape(src)}</code>"
    if sec:
        header += f" | <i>{_html.escape(sec)}</i>"
    header += f" | score={score:.3f}"

    # Escape toàn bộ text, sau đó tô màu từng span
    escaped = _html.escape(text)
    highlighted_count = 0
    for span in grounded_spans:
        span = span.strip()
        if not span:
            continue
        escaped_span = _html.escape(span)
        if escaped_span in escaped:
            escaped = escaped.replace(
                escaped_span,
                f'<mark style="background:#ffe066;padding:1px 3px;border-radius:3px">'
                f'{escaped_span}</mark>',
                1,
            )
            highlighted_count += 1

    # Dấu hiệu chunk có đóng góp
    border_color = "#f5a623" if highlighted_count > 0 else "#ddd"
    body = escaped.replace("\n", "<br>")

    return (
        f'<div style="margin-bottom:14px;padding:10px 12px;'
        f'border-left:4px solid {border_color};border-radius:4px;'
        f'background:#fafafa">'
        f'<p style="margin:0 0 6px;font-size:0.85em;color:#555">{header}</p>'
        f'<p style="margin:0;font-size:0.88em;line-height:1.6">{body}</p>'
        f'</div>'
    )


# =============================================================================
# MAIN — Chatbot UI (Gradio)
# =============================================================================

# Câu hỏi mẫu từ data/test_questions.json
_EXAMPLE_QUERIES = [
    "SLA xử lý ticket P1 là bao lâu?",
    "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
    "Ai phải phê duyệt để cấp quyền Level 3?",
    "Sản phẩm kỹ thuật số có được hoàn tiền không?",
    "Tài khoản bị khóa sau bao nhiêu lần đăng nhập sai?",
    "Escalation trong sự cố P1 diễn ra như thế nào?",
    "Nhân viên được làm remote tối đa mấy ngày mỗi tuần?",
    "ERR-403-AUTH là lỗi gì và cách xử lý?",
]


def _chat_fn(
    query: str,
    retrieval_mode: str,
    top_k_search: int,
    top_k_select: int,
    use_rerank: bool,
    history: list,
) -> Tuple[list, str, str]:
    """
    Hàm xử lý mỗi lượt chat, trả về:
      - history mới (cho Chatbot component)
      - sources markdown
      - chunks HTML (có highlight câu đóng góp)
    """
    if not query.strip():
        return history, "", ""

    try:
        result = rag_answer(
            query=query.strip(),
            retrieval_mode=retrieval_mode,
            top_k_search=top_k_search,
            top_k_select=top_k_select,
            use_rerank=use_rerank,
            verbose=False,
        )
        answer = result["answer"]
        sources = result["sources"]
        chunks = result["chunks_used"]
        grounded_spans = result.get("grounded_spans", [])

        # --- Format sources ---
        if sources:
            src_md = "\n".join(f"- `{s}`" for s in sorted(sources))
        else:
            src_md = "_Không có source_"

        # --- Format chunks với highlight ---
        if chunks:
            chunk_html_parts = [
                _highlight_chunk_html(i, c, grounded_spans)
                for i, c in enumerate(chunks, 1)
            ]
            legend = (
                '<p style="font-size:0.8em;color:#888;margin:0 0 10px">'
                '<mark style="background:#ffe066;padding:1px 4px;border-radius:3px">▌</mark>'
                " = câu trực tiếp đóng góp vào câu trả lời</p>"
            )
            chunks_html = legend + "".join(chunk_html_parts)
        else:
            chunks_html = "<i>Không có chunk</i>"

    except Exception as e:
        answer = f"Lỗi: {e}"
        src_md = ""
        chunks_html = ""

    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    return history, src_md, chunks_html


def launch_chatbot() -> None:
    import gradio as gr

    with gr.Blocks(title="RAG Chatbot — Day 08 Lab") as demo:
        gr.Markdown(
            "# RAG Chatbot\n"
            "Hỏi về SLA, chính sách hoàn tiền, quyền truy cập, HR policy…\n"
            "Câu trả lời được grounded từ tài liệu nội bộ, kèm citation."
        )

        with gr.Row():
            # ── Left column: chat ────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Hội thoại",
                    height=480,
                )
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Nhập câu hỏi của bạn…",
                        show_label=False,
                        scale=5,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Gửi", variant="primary", scale=1)
                clear_btn = gr.Button("Xóa lịch sử", size="sm")

                gr.Markdown("**Câu hỏi mẫu** — click để điền vào ô nhập:")
                example_btns = []
                for i in range(0, len(_EXAMPLE_QUERIES), 2):
                    with gr.Row():
                        for q in _EXAMPLE_QUERIES[i:i+2]:
                            btn = gr.Button(q, size="sm", variant="secondary")
                            btn.click(fn=lambda v=q: v, outputs=query_box)
                            example_btns.append(btn)

            # ── Right column: settings + debug ───────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Cấu hình pipeline")
                retrieval_mode = gr.Radio(
                    choices=["dense", "hybrid", "sparse"],
                    value="dense",
                    label="Retrieval mode",
                    info="dense=vector search | hybrid=dense+BM25 | sparse=BM25 only",
                )
                top_k_search = gr.Slider(
                    minimum=3, maximum=20, step=1, value=TOP_K_SEARCH,
                    label=f"Top-K search (search rộng)",
                )
                top_k_select = gr.Slider(
                    minimum=1, maximum=10, step=1, value=TOP_K_SELECT,
                    label="Top-K select (đưa vào prompt)",
                )
                use_rerank = gr.Checkbox(
                    value=False,
                    label="Rerank (cross-encoder) — Sprint 3",
                )

                gr.Markdown("### Sources")
                sources_box = gr.Markdown(value="_Chưa có câu hỏi_")

                with gr.Accordion("Chunks debug (câu highlight = đóng góp vào trả lời)", open=False):
                    chunks_box = gr.HTML(value="")

        # ── State ─────────────────────────────────────────────────────────────
        state_history = gr.State([])

        # ── Submit handlers ───────────────────────────────────────────────────
        submit_inputs = [
            query_box, retrieval_mode, top_k_search, top_k_select,
            use_rerank, state_history,
        ]

        def _submit(q, mode, k_s, k_sel, rerank, hist):
            new_hist, src, cks = _chat_fn(q, mode, k_s, k_sel, rerank, hist)
            return new_hist, new_hist, src, cks, ""

        for trigger in [send_btn.click, query_box.submit]:
            trigger(
                fn=_submit,
                inputs=submit_inputs,
                outputs=[state_history, chatbot, sources_box, chunks_box, query_box],
            )

        clear_btn.click(
            fn=lambda: ([], [], "_Chưa có câu hỏi_", ""),
            outputs=[state_history, chatbot, sources_box, chunks_box],
        )

    demo.launch(inbrowser=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_chatbot()



