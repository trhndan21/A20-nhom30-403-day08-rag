# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Quang Hiển  
**Vai trò trong nhóm:** Documentation Owner (Sprint Lead 4)  
**Ngày nộp:** 2026-04-13

## 1. Tôi đã làm gì trong lab này?

Trong buổi lab này, tôi phụ trách Documentation Owner và đồng thời làm Sprint Lead 4. Phần việc chính của tôi là hoàn thiện bộ tài liệu nộp bài của nhóm gồm `docs/architecture.md`, `docs/tuning-log.md` và phần tổng hợp trong `reports/group_report.md`.

Tôi thực hiện theo ba bước. Một là thu thập đầu vào kỹ thuật từ các thành viên phụ trách code: thông số chunking, cấu hình baseline/variant, và tham số mô hình sinh câu trả lời. Hai là đối chiếu số liệu giữa `results/scorecard_baseline.md`, `results/scorecard_variant.md` và `results/ab_comparison.csv` để tránh ghi nhận sai so với lần chạy thực tế. Ba là chuẩn hóa cách trình bày để người chấm có thể lần theo logic từ code đến kết quả rồi đến kết luận.

Với vai trò Sprint Lead 4, tôi cũng chốt định dạng deliverables để bám rubric: architecture mô tả rõ pipeline và thông số chính, tuning-log ghi đúng cấu hình đã thử, report nhóm phản ánh đúng năng lực hiện tại của hệ thống.

## 2. Điều tôi hiểu rõ hơn sau lab này

Điều tôi hiểu rõ nhất là retrieval tốt chưa đủ để tạo câu trả lời tốt. Trong scorecard hiện tại, baseline và variant đều có Context Recall 5.00/5 nhưng Completeness chỉ 4.20/5. Điều này cho thấy hệ thống đã lấy đúng ngữ cảnh, nhưng phần generation vẫn có thể thiếu ý hoặc trả lời chưa đủ hữu ích.

Tôi cũng hiểu rõ hơn về kỷ luật A/B testing: chỉ đổi một biến mỗi lần. Nếu thay nhiều biến đồng thời, gần như không thể xác định nguyên nhân thay đổi điểm. Vì vậy ở tuning-log, tôi luôn ghi rõ retrieval mode, rerank bật/tắt, top-k và model trước khi đưa ra kết luận.

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Khó khăn lớn nhất với vai trò Documentation Owner là cân bằng giữa văn phong dễ đọc và tính chính xác theo bằng chứng. Khi tổng hợp từ nhiều nguồn, rất dễ diễn giải “hợp lý” nhưng vượt quá dữ liệu. Trường hợp điển hình là variant: có cải thiện cục bộ ở q06, nhưng điểm Relevance trung bình lại giảm.

Điều tôi ngạc nhiên là câu trả lời dạng abstain không phải lúc nào cũng được chấm relevance cao. Mô hình có thể trung thực (faithfulness tốt) nhưng vẫn mất điểm nếu trả lời quá ngắn hoặc không nêu thông tin chuẩn liên quan gần nhất.

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi được chọn:** q10 — “Nếu cần hoàn tiền khẩn cấp cho khách hàng VIP, quy trình có khác không?”

Đây là câu hỏi kiểm tra tốt khả năng abstain có kiểm soát. Ở baseline, hệ thống trả lời không có thông tin quy trình VIP trong ngữ cảnh; điểm là Faithfulness 5, Relevance 3, Completeness 3. Ở variant, câu trả lời vẫn từ chối tương tự; Faithfulness giữ 5 nhưng Relevance giảm xuống 1.

Kết quả này cho thấy lỗi chính không nằm ở retrieval mà ở generation policy. Câu trả lời hiện tại đúng về mặt dữ liệu nhưng chưa đủ hữu ích. Expected answer yêu cầu đồng thời hai ý: không có quy trình VIP riêng và vẫn áp dụng quy trình chuẩn 3-5 ngày làm việc. Hệ thống mới nêu ý thứ nhất nên bị trừ relevance/completeness.

Theo tôi, nguyên nhân gốc là prompt abstain chưa ép mô hình bổ sung “thông tin chuẩn gần nhất” khi thiếu dữ liệu đặc thù. Nếu sửa theo hướng này, hệ thống vẫn giữ grounding nhưng trả lời đầy đủ hơn.

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi sẽ làm hai việc. Thứ nhất, tinh chỉnh prompt cho nhóm câu hỏi thiếu ngữ cảnh đặc thù theo mẫu: xác nhận thiếu dữ liệu trực tiếp, sau đó nêu quy trình chuẩn liên quan nếu có trong tài liệu. Mục tiêu là tăng Relevance ở các câu như q10 mà không đánh đổi tính trung thực.

Thứ hai, chạy thêm một vòng A/B chỉ thay riêng yếu tố rerank (giữ nguyên retrieval mode) để đo tác động thật của từng biến. Cách này giúp kết luận trong tuning-log chắc hơn và tránh nhiễu khi phân tích.
