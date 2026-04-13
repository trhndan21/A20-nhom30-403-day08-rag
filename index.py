import os
import re
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# CẤU HÌNH
DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 400       # Ước lượng tokens
CHUNK_OVERLAP = 80     # Độ gối đầu

# Khởi tạo model Local - Chạy nhanh và không tốn phí API
print("⏳ Đang khởi tạo model all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 1: PREPROCESS - Trích xuất Metadata & Clean Text
def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": Path(filepath).name,
        "section": "General",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            # Regex bắt cặp Key: Value ở đầu file
            match = re.match(r"^(Source|Department|Effective Date|Access):\s*(.*)$", line, re.IGNORECASE)
            if match:
                key = match.group(1).lower().replace(" ", "_")
                value = match.group(2).strip()
                metadata[key] = value
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text) 

    return {"text": cleaned_text, "metadata": metadata}

# STEP 2: CHUNK - Cắt văn bản theo Paragraph & Heading
def _split_by_size(text: str, base_metadata: Dict, section: str) -> List[Dict[str, Any]]:
    chunk_chars = CHUNK_SIZE * 4
    overlap_chars = CHUNK_OVERLAP * 4
    
    # Chiến thuật: Cắt theo đoạn văn để giữ trọn ý nghĩa
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts = []
    current_len = 0
    overlap_tail = ""

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_chars and current_parts:
            chunk_text = overlap_tail + "\n\n".join(current_parts)
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**base_metadata, "section": section}
            })
            # Lấy đoạn đuôi làm gối đầu cho chunk sau
            overlap_tail = chunk_text[-overlap_chars:] + "\n\n" if len(chunk_text) > overlap_chars else ""
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        chunks.append({
            "text": (overlap_tail + "\n\n".join(current_parts)).strip(),
            "metadata": {**base_metadata, "section": section}
        })
    return chunks

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Cắt theo Heading === Section ===
    parts = re.split(r"(===.*?===)", text)
    current_section = "General"
    
    for part in parts:
        part = part.strip()
        if not part: continue
        if re.match(r"===.*?===", part):
            current_section = part.replace("=", "").strip()
        else:
            section_chunks = _split_by_size(part, base_metadata, current_section)
            chunks.extend(section_chunks)
    return chunks

# STEP 3: EMBED + STORE
def get_embedding(text: str) -> List[float]:
    from sentence_transformers import SentenceTransformer
    if not hasattr(get_embedding, "_model"):
        get_embedding._model = SentenceTransformer("all-MiniLM-L6-v2")
    return get_embedding._model.encode(text, normalize_embeddings=True).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    print(f"🚀 Bắt đầu Indexing tài liệu vào: {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"}
    )

    doc_files = list(docs_dir.glob("*.txt"))
    total_chunks = 0

    for filepath in doc_files:
        print(f"📄 Đang đọc: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)
        
        ids, embs, docs, metas = [], [], [], []
        for i, ck in enumerate(chunks):
            ids.append(f"{filepath.stem}_{i}")
            embs.append(get_embedding(ck["text"]))
            docs.append(ck["text"])
            metas.append(ck["metadata"])
            
        if ids:
            collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
            total_chunks += len(ids)

    print(f"\n✅ Thành công! Đã lưu {total_chunks} chunks.")

# STEP 4: INSPECT - Kiểm tra chất lượng (Phục vụ DoD)
def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 3):
    print("\n" + "="*50 + "\nKIỂM TRA CHUNK MẪU\n" + "="*50)
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        res = collection.get(limit=n, include=["documents", "metadatas"])
        for i, (d, m) in enumerate(zip(res["documents"], res["metadatas"])):
            print(f"Chunk {i+1} | Nguồn: {m['source']} | Mục: {m['section']}")
            print(f"Ngày hiệu lực: {m.get('effective_date')}")
            print(f"Nội dung: {d[:150]}...\n")
    except Exception as e:
        print(f"Lỗi đọc DB: {e}")

def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR):
    print("--- THỐNG KÊ METADATA ---")
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        res = collection.get(include=["metadatas"])
        depts = {}
        for m in res["metadatas"]:
            d = m.get("department", "unknown")
            depts[d] = depts.get(d, 0) + 1
        for d, count in depts.items():
            print(f"- {d}: {count} chunks")
    except Exception as e:
        print(f"Lỗi: {e}")

# RUN
if __name__ == "__main__":
    build_index()
    list_chunks()
    inspect_metadata_coverage()