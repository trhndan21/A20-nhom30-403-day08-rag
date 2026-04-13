# Nhật Ký Tinh Chỉnh Pipeline RAG

## 1. Cấu hình baseline

Ngày chạy: 2026-04-13

```text
retrieval_mode = dense
chunk_size = 400
overlap = 80
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = gpt-4o-mini
```

Điểm trung bình baseline:

| Chỉ số | Điểm |
|---|---|
| Faithfulness | 4.70/5 |
| Relevance | 4.80/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.20/5 |

Các câu cần cải thiện trong baseline:

1. q10: đúng hướng từ chối do thiếu dữ liệu, nhưng thiếu nêu rõ quy trình chuẩn liên quan.
2. q04: trả lời đúng chính sách chính, nhưng thiếu chi tiết diễn giải đầy đủ về ngoại lệ.
3. q07: chưa nêu rõ thông tin đổi tên tài liệu trong phần trả lời.

## 2. Cấu hình variant

Ngày chạy: 2026-04-13

```text
retrieval_mode = hybrid
chunk_size = 400
overlap = 80
top_k_search = 10
top_k_select = 3
use_rerank = True
llm_model = gpt-4o-mini
```

Điểm trung bình variant:

| Chỉ số | Baseline | Variant | Chênh lệch |
|---|---:|---:|---:|
| Faithfulness | 4.70/5 | 4.70/5 | +0.00 |
| Relevance | 4.80/5 | 4.50/5 | -0.30 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 4.20/5 | 4.20/5 | +0.00 |

Quan sát chính:

1. Variant cải thiện cục bộ ở q06.
2. Variant giảm chất lượng ở q09 và q10.
3. Tập câu hỏi hiện tại cho thấy baseline ổn định hơn về tổng thể.

## 3. Kết luận A/B

Kết quả thí nghiệm chưa cho thấy variant vượt baseline ở mức trung bình toàn tập. Baseline phù hợp làm cấu hình chính để trình diễn và nộp kết quả ổn định. Variant vẫn có giá trị tham khảo cho các truy vấn kỹ thuật, nhưng cần tinh chỉnh thêm trước khi dùng làm cấu hình mặc định. Ở lần chạy này, nhóm thay đổi đồng thời `retrieval_mode` (dense -> hybrid) và `use_rerank` (False -> True), vì vậy bước tiếp theo sẽ tách biến để đo tác động rõ ràng hơn của từng thành phần.

## 4. Hướng cải tiến tiếp theo

1. Tinh chỉnh prompt cho nhóm câu hỏi thiếu ngữ cảnh để tăng độ nhất quán khi từ chối trả lời.
2. Thử tách thí nghiệm từng biến độc lập để xác định tác động thực sự của rerank.
3. Bổ sung kiểm soát completeness theo mẫu trả lời chuẩn cho các câu hỏi chính sách có nhiều điều kiện.
4. Chạy lại A/B chuẩn một biến theo hai bước: (a) dense + rerank=False so với hybrid + rerank=False, (b) hybrid + rerank=False so với hybrid + rerank=True.
