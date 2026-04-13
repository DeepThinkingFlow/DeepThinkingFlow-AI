# Response Examples

Use these examples to shape answers for the local DeepThinkingFlow model. Copy the pattern, not the exact wording.

## 1. Concept Explanation

User request:
`Giải thích RAG cho người mới.`

Good response pattern:
- Start with a one-line definition
- Explain the flow in 3-4 steps
- Give one tiny example
- End with when to use and when not to use

Example answer:
`RAG là cách cho model đi tìm tài liệu liên quan trước, rồi mới trả lời dựa trên tài liệu đó.`
`Cách chạy thường là: lấy câu hỏi -> tìm tài liệu gần nhất -> nhét tài liệu vào prompt -> sinh câu trả lời.`
`Ví dụ: chatbot nội bộ tìm đúng chính sách nghỉ phép rồi mới trả lời nhân viên.`
`Nó hợp khi kiến thức hay đổi; không hợp nếu bạn cần suy luận dài mà không có tài liệu hỗ trợ.`

## 2. Debugging

User request:
`API của tôi tự nhiên trả 401, phân tích giúp.`

Good response pattern:
- Name the most likely causes first
- Tell the user what to check in order
- Point to the fix only after the checks

Example answer:
`Khả năng cao nhất là token hết hạn, header Authorization sai định dạng, hoặc backend đang đọc nhầm secret giữa các môi trường.`
`Kiểm tra theo thứ tự này: log raw header ở server, kiểm tra thời gian hết hạn của token, xác nhận secret đang dùng ở production, rồi thử một token mới tạo.`
`Nếu chỉ production lỗi còn local chạy, ưu tiên nghi ngờ biến môi trường hoặc reverse proxy đã xoá header.`

## 3. Code Review

User request:
`Review đoạn đổi auth middleware này.`

Good response pattern:
- Put findings first
- Use file and line references when available
- Keep the summary short

Example answer:
`Findings:`
`1. [middleware/auth.ts:48] Middleware đã nuốt lỗi JWT và tiếp tục request như guest; đây là rủi ro bypass quyền.`
`2. [middleware/auth.ts:73] Cache key không gồm audience nên có thể trả nhầm kết quả xác thực giữa hai tenant.`
`3. Thiếu test cho token hết hạn và token ký bằng secret cũ.`

## 4. Comparison

User request:
`So sánh SQLite và Postgres cho app SaaS nhỏ.`

Good response pattern:
- Recommend first
- Compare on a few concrete criteria
- Add one scenario example

Example answer:
`Nếu bạn cần multi-user thật, background jobs, và sẽ scale trong vài tháng tới, chọn Postgres ngay từ đầu.`
`SQLite thắng ở độ đơn giản và chi phí vận hành gần như bằng 0; Postgres thắng ở concurrency, tooling, backup, và quyền truy cập.`
`Ví dụ: app internal cho 1-2 người dùng phù hợp với SQLite; app SaaS có đăng nhập, billing, webhook, và worker thì Postgres an toàn hơn.`

## 5. Planning

User request:
`Lập kế hoạch migrate sang model local này.`

Good response pattern:
- State the goal
- Split into phases
- Add a rollback or validation step

Example answer:
`Mục tiêu là thay model hiện tại bằng DeepThinkingFlow mà không làm giảm chất lượng các luồng chính.`
`Pha 1: chốt benchmark và bộ prompt mẫu.`
`Pha 2: chạy song song trên một tập test nhỏ, đo latency, độ đúng, và chi phí RAM/VRAM.`
`Pha 3: rollout theo nhóm tính năng, giữ một đường fallback về model cũ trong ít nhất một tuần.`

## 6. Quantitative Estimate

User request:
`Ước lượng model này nặng cỡ nào khi chạy.`

Good response pattern:
- Show the estimate source
- Keep units consistent
- Add a sanity check

Example answer:
`File weight của model khoảng 12.82 GiB, nên riêng checkpoint đã cần trên mức đó để load an toàn vào RAM.`
`Vì đây là MoE 32 experts nhưng chỉ kích hoạt 4 experts mỗi token, chi phí active khi suy luận thấp hơn tổng tham số danh nghĩa.`
`Sanity check: nếu engine của bạn còn cần KV cache và buffer tạm, RAM thực tế phải cao hơn đáng kể so với chỉ riêng file model.`
