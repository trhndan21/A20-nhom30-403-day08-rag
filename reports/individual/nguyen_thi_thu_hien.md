# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Thị Thu Hiền

**Mã học viên:** 2A202600212

**Nhóm:** 30  

**Vai trò trong nhóm:** Retrieval Owner (Phụ trách phần Indexing & Embedding)

**Ngày nộp:** 13/04/2026 

**Độ dài:** ~750 từ

---

## 1. Tôi đã làm gì trong lab này?

Trong buổi lab này, tôi chịu trách nhiệm chính về "nền móng" dữ liệu của toàn bộ pipeline, cụ thể là xây dựng file `index.py` xuyên suốt Sprint 1 và tối ưu hóa nó cho Sprint 3. Tôi đã trực tiếp thiết kế và lập trình bộ lọc tiền xử lý (Preprocess) sử dụng Regular Expressions để bóc tách chính xác các trường Metadata quan trọng như *Department, Effective Date, Access* từ phần header của tài liệu. 

Về phần chiến thuật xử lý văn bản, tôi đã quyết định triển khai mô hình **Paragraph-based Chunking** thay vì cắt theo số lượng ký tự cứng nhắc, kết hợp với cơ chế **Overlap (gối đầu)** 80 tokens để bảo toàn ngữ cảnh giữa các đoạn. Đặc biệt, tôi đã thực hiện một bước chuyển đổi quan trọng từ mô hình thương mại OpenAI sang mô hình mã nguồn mở **all-MiniLM-L6-v2**. Quyết định này giúp nhóm có thể chạy pipeline hoàn toàn offline trên máy cá nhân, đảm bảo tính riêng tư và tăng tốc độ xử lý vector hóa. Công việc của tôi cung cấp cơ sở dữ liệu Vector (ChromaDB) hoàn chỉnh để Tech Lead thực hiện các bước truy xuất và tạo câu trả lời ở các Sprint tiếp theo.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Sau buổi lab này, tôi đã thực sự hiểu rõ tầm quan trọng của **Metadata-driven Retrieval**. Trước đây, tôi chỉ hình dung đơn giản RAG là việc biến chữ thành số rồi đi so sánh độ tương đồng. Nhưng khi trực tiếp gán các nhãn như `effective_date` hay `department` vào từng chunk dữ liệu, tôi hiểu rằng đây là chìa khóa để giải quyết bài toán "trợ lý nội bộ" trong doanh nghiệp. Nó cho phép hệ thống không chỉ tìm kiếm theo ngữ nghĩa mà còn có thể lọc được những thông tin lỗi thời hoặc giới hạn quyền truy cập dựa trên phòng ban.

Bên cạnh đó, việc thực hành **Chunking strategy** giúp tôi nhận ra rằng chất lượng của một hệ thống RAG phụ thuộc rất nhiều vào "mẩu" dữ liệu đầu vào. Nếu khâu indexing làm không tốt (ví dụ: cắt ngang xương một điều khoản SLA), LLM dù thông minh đến đâu cũng sẽ bị dẫn dắt tới hiện tượng ảo giác (hallucination) do dữ liệu đầu vào bị mất ngữ cảnh.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Điều khiến tôi ngạc nhiên nhất chính là sự nhạy cảm của **Vector Dimensions** (kích thước vector). Khi chuyển đổi từ OpenAI (1536 chiều) sang MiniLM (384 chiều), tôi đã gặp lỗi xung đột nghiêm trọng vì ChromaDB không thể ghi đè dữ liệu mới vào collection cũ có cấu trúc khác biệt. Việc debug lỗi này mất khá nhiều thời gian cho đến khi tôi nhận ra mình phải xóa sạch thư mục `chroma_db` vật lý để khởi tạo lại toàn bộ index.

Một khó khăn khác nằm ở việc xử lý dữ liệu thô từ file `.txt`. Các ký tự xuống dòng thừa và định dạng không đồng nhất khiến hàm chunking ban đầu hoạt động sai lệch, tạo ra các chunk quá ngắn hoặc quá dài. Tôi đã phải sử dụng Regex để chuẩn hóa văn bản (normalize), đảm bảo mỗi đoạn văn được phân tách bởi đúng hai ký tự xuống dòng trước khi đưa vào bộ cắt đoạn. Điều này dạy cho tôi bài học rằng: trong AI, việc làm sạch dữ liệu chiếm 80% thành công của mô hình.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** "SLA xử lý ticket P1 là bao lâu?"

**Phân tích:**
Đây là câu hỏi mà nhóm tôi dùng để kiểm chứng hiệu quả của việc Indexing và Retrieval. 

* **Ở bản Baseline (Sprint 2):** Hệ thống chỉ đạt điểm Context Recall trung bình (khoảng 3/5). Nguyên nhân là do từ khóa "P1" là một mã định danh đặc thù. Mô hình Dense Retrieval đôi khi ưu tiên các đoạn văn nói về "SLA chung" của công ty thay vì tìm trúng đoạn văn có chứa mã "P1". Kết quả là AI trả lời mơ hồ hoặc bị lẫn lộn giữa các mức độ ưu tiên khác nhau.
* **Lỗi nằm ở:** **Retrieval**. Việc tìm kiếm theo ngữ nghĩa (Semantic Search) thuần túy đôi khi bỏ lỡ các từ khóa kỹ thuật quan trọng. 
* **Ở bản Variant (Sprint 3):** Sau khi tôi tư vấn cho nhóm sử dụng Metadata lọc theo `source` và kết hợp với việc tinh chỉnh lại Chunking, điểm số đã cải thiện rõ rệt lên 5/5. Nhờ việc gán metadata nguồn chính xác, hệ thống đã truy xuất đúng file `sla_p1_2026.txt`. Điều này chứng minh rằng việc tổ chức Metadata tốt ngay từ khâu Indexing giúp LLM trích dẫn nguồn một cách tự tin và chính xác tuyệt đối.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi muốn thử nghiệm kỹ thuật **Recursive Character Text Splitter** thay vì chia theo paragraph đơn giản. Kết quả đánh giá (evaluation) cho thấy một số đoạn văn trong tài liệu chính sách rất dài và chứa nhiều danh sách liệt kê (bullet points). Việc cắt theo đoạn văn khiến các chunk này bị "loãng" thông tin. Tôi muốn áp dụng cách cắt phân cấp (Heading -> Paragraph -> Sentence) để mỗi chunk chứa một ý duy nhất, từ đó nâng cao độ chính xác của câu trả lời cuối cùng.
