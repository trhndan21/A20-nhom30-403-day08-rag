# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Trịnh Đức An
**Vai trò trong nhóm:** Retrieval Owner (Phụ trách Sprint 2 & Sprint 3)
**Ngày nộp:** 2026-04-13
**Độ dài:** ~660 từ

---

## 1. Tôi đã làm gì trong lab này? (Đóng góp cụ thể)

Trong Lab 08, tôi đảm nhận vai trò **Retrieval Owner**, chịu trách nhiệm trực tiếp xây dựng và tối ưu hoá lớp truy hồi (Retrieval Layer) của pipeline thông qua Sprint 2 và Sprint 3. Nhiệm vụ cốt lõi của tôi là đảm bảo hệ thống cung cấp ngữ cảnh (context) chính xác, bao phủ đủ thông tin để LLM sinh câu trả lời đúng và không bị hallucinate.

Các task tôi đã tự tay implement trong cấu trúc `rag_answer.py`:
- **Sprint 2 (Baseline Dense Retrieval):** Tôi thiết kế hàm `retrieve_dense()`, khởi tạo môi trường ChromaDB để kết nối với Index Database từ Sprint 1. Sử dụng kỹ thuật Vector Cosine Similarity để trích xuất ra top-k chunks có ý nghĩa ngữ nghĩa (semantic) gần nhất với query. Tôi đồng thời xử lý lỗi API bị rate-limit bên phía Gemini bằng cách tích hợp fallback qua model OpenAI để pipeline không gục ngã.
- **Sprint 3 (Tối ưu Retrieval bằng Hybrid Search):** Phân tích corpus cho thấy Dense Retrieval yếu lật mặt trong việc bắt các keyword kỹ thuật và mã số cứng ngắc (*ERR-403, P1, VPN*), tôi chọn variant **Hybrid Search** cho Sprint 3. Tôi phát triển độc lập hàm `retrieve_sparse()` thông qua cài đặt thư viện `rank_bm25` để đếm tần suất từ vựng (exact-match). Điểm nhấn là tôi đã xây dựng thành công hàm `retrieve_hybrid()`, dùng thuật toán **Reciprocal Rank Fusion (RRF)** ($K=60$) để scale và dung hòa mượt mà Dense Score lẫn Sparse Score.

---

## 2. Phân tích một câu hỏi trong Grading Questions

**Câu hỏi được phân tích:** `gq07` - *"Approval Matrix để hệ thống cấp quyền là tài liệu nào?"* (Phân loại: Access Control)  

**Phân tích lỗi (Failure Mode & Root Cause):**  
Trong bộ dữ liệu cung cấp (corpus), tài liệu thực tế tên là **"Access Control SOP"** - hệ thống đã cập nhật tên quy định mà không dùng "Approval Matrix" nữa. 
Khi vận hành thủ công query bằng **Dense Retrieval** (baseline), hệ thống ngay lập tức gặp tình trạng semantic blindness (mù ngữ nghĩa tên gọi lóng). Thuật toán embedding không tìm thấy sự hội tụ vector gần gũi giữa "Approval Matrix" và "Access Control", đẩy chunk quan trọng nhất bị tuột khỏi nhóm top-k. Khi Generation (LLM) đối diện chùm context nhiễu này, nó đành ngậm miệng (abstain) hoặc sinh đáp án sai. 

**Giải pháp triệt để với Hybrid RRF:**  
Bằng hàm `retrieve_hybrid` tôi viết, luồng `retrieve_sparse` (BM25) khi phân rã query này đã bắt được đích xác chữ *"Approval"* (vô tình còn sót lại trong một đoạn text con của file). BM25 kích rank của văn bản này lên kịch kim ($1.0$). Khi dồn luồng ngang cấp tại bộ **RRF**, chunk ngay lập tức được boost kéo vút lên Rank-1. LLM nhận đầy đủ context chuẩn và trả về điểm 100% trong Scorecard Variant.
*Bài học làm nghề:* Với tập tài liệu hành chính dính nhiều alias (tên gọi tiếng lóng) khác biệt hẳn mặt chữ, Hybrid Search không phải tính năng "Nice-to-have", mà là thành phần **bắt buộc**.

---

## 3. Điều tôi ngạc nhiên và rút kinh nghiệm

Trong quá trình implement mô hình Sparse (BM25), rào cản gieo khó khăn thực tế nhất lại đến từ vấn đề **Tokenization**. 
Ban đầu tôi chia tách từ khóa đơn sơ qua hàm `.lower().split()`. Đáng ngạc nhiên là đối với những truy vấn mã lỗi cứng như `"ERR-403-AUTH"`, Python split chừa nó thành một khối y nguyên. Khi đối chiếu (matching), chỉ cần User nhập dư dấu cách hay sót dấu gạch nối, BM25 lập tức rớt bảng và trả về 0 chunks. 
Lúc này tôi ghi nhận một bài học đau đớn: **Sparse Search là con dao hai lưỡi**. Dù khả năng bắt Keyword bá đạo hơn Embeddings, nhưng lại cực kì mong manh trước tokenizer. Cách chia token quá hời hợt hoặc chỉ đơn thuần dùng space cho một ngôn ngữ đặc thù (tiếng Việt mã IT) sẽ bóp nát tỷ lệ recall của hệ thống. 

---

## 4. Đề xuất cải tiến hệ thống

Xuyên suốt quá trình chạy A/B Testing dẫn đến file `scorecard_variant.md`, dù mức Context Recall của phương pháp Hybrid đạt mốc cực chuẩn ($5.0/5$), tôi xin đề nghị 2 điểm kiến trúc chiến lược nhằm nâng cấp pipeline lên tiêu chuẩn Production-ready:

1. **Khảm Custom Tokenizer cho Nhúng Tiếng Việt/IT:** Thay đổi hoàn toàn cơ chế `.split()` trong BM25 bằng các thư viện Word Segmentation thực thụ (ví dụ `pyvi` hoặc `underthesea`). Điều này giúp phân tách từ ghép "hệ thống", "hoàn tiền" thành một token nguyên vẹn làm Sparse Score tụ hội chính xác hơn.
2. **Quy trình Query Expansion trước Retrieval:** Đối với việc User nhập từ khóa cổ (như "Approval Matrix"), để không phó mặc sự may rủi vào việc vô tình file vẫn chứa chữ "Approval", ta nên gài một bước LLM rewrite query nhanh. Ví dụ khi User hỏi "Approval Matrix", LLM sinh keyword đồng nghĩa: `Approval Matrix OR Access Control SOP`. Gom toàn bộ chuỗi này đẩy vào kho tìm kiếm Hybrid, khả năng bị Fail Keyword sẽ rớt về mốc $0%$.
