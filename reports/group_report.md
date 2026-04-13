# Báo Cáo Nhóm — Lab Day 08 RAG Pipeline

## 1. Mục tiêu và phạm vi

Nhóm xây dựng hệ thống RAG cho trợ lý nội bộ nhằm trả lời các câu hỏi liên quan đến SLA, chính sách hoàn tiền, kiểm soát truy cập và quy định nhân sự. Hệ thống phải bảo đảm câu trả lời bám sát tài liệu nguồn, có trích dẫn, và ưu tiên từ chối trả lời khi dữ liệu không đủ thay vì suy diễn.

## 2. Kiến trúc đã triển khai

Pipeline gồm bốn lớp chính:

1. Lập chỉ mục: tiền xử lý tài liệu, chia đoạn, sinh embedding, lưu vào ChromaDB.
2. Truy hồi: triển khai baseline dense retrieval và phương án variant hybrid kết hợp rerank.
3. Sinh câu trả lời: dùng grounded prompt, giới hạn nhiệt độ bằng 0 để tăng tính ổn định.
4. Đánh giá: chấm điểm theo bốn chỉ số và so sánh A/B trên tập 10 câu hỏi.

## 3. Kết quả thực nghiệm

Điểm trung bình hiện tại:

- Baseline: Faithfulness 4.70, Relevance 4.80, Context Recall 5.00, Completeness 4.20.
- Variant: Faithfulness 4.70, Relevance 4.50, Context Recall 5.00, Completeness 4.20.

Nhận xét:

1. Baseline ổn định hơn về chất lượng tổng thể.
2. Variant cải thiện cục bộ ở một số câu hỏi nhưng chưa bền vững trên toàn bộ tập đánh giá.
3. Context Recall cao ở cả hai cấu hình cho thấy khả năng truy hồi nguồn tốt; phần cần tối ưu thêm nằm ở lớp sinh câu trả lời và độ đầy đủ nội dung.
4. Trong lần chạy hiện tại, nhóm thử đồng thời hybrid và rerank; vì vậy cần thêm một vòng tách biến để kết luận rõ hơn mức đóng góp của từng thành phần.

## 4. Phân tích và bài học

1. Nguyên tắc A/B một biến thay đổi là cần thiết để đọc đúng tác động của từng quyết định kỹ thuật.
2. Truy hồi tốt chưa đồng nghĩa câu trả lời tốt; cần kiểm soát chặt tính đầy đủ và mức độ bám ngữ cảnh khi sinh đáp án.
3. Với nhóm câu hỏi thiếu dữ liệu, quy tắc từ chối trả lời cần rõ ràng và nhất quán hơn để tránh giảm điểm relevance/completeness.
4. Cần bổ sung một vòng thí nghiệm tách biến để xác nhận rõ đóng góp riêng của hybrid và rerank.

## 5. Định hướng cải tiến

1. Tinh chỉnh prompt cho nhóm câu hỏi thiếu ngữ cảnh để cải thiện tính nhất quán khi từ chối trả lời.
2. Tách thử nghiệm theo từng biến độc lập, đặc biệt với rerank, để tránh nhiễu khi phân tích.
3. Bổ sung tiêu chí kiểm tra completeness theo mẫu câu trả lời chuẩn ở các câu hỏi chính sách nhiều điều kiện.

## 6. Tài liệu liên quan

- docs/architecture.md
- docs/tuning-log.md
