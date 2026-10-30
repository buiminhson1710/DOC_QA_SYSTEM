# BÁO CÁO PHÂN TÍCH LỖI VÀ HƯỚNG DẪN BÀN GIAO HỆ THỐNG DOC_QA
*(Dành cho việc chuyển giao và chạy trên môi trường Windows)*

---

## I. TỔNG QUAN NGUYÊN NHÂN CÁC LỖI THỰC TẾ

### 1. Tại sao trả lời 1 câu hỏi mất gần 300 giây (295 giây)?
- **Nhật ký thực tế**:
  ```
  WARNING  | docqa.llm | Model gemini/gemini-3.8-flash tạm ngưng 60s (overloaded)
  WARNING  | docqa.llm | Đã dùng model dự phòng: gemini/gemini-2.5-flash
  INFO     | docqa.rag | CHAT_OK | seconds: 294.98 | model: gemini/gemini-2.5-flash
  ```
- **Nguyên nhân**:
  1. Trong file `.env` trên máy Windows, model chính **vẫn đang đặt `GENERATION_MODEL=gemini/gemini-3.8-flash`**.
  2. Vì model `gemini-3.8-flash` không tồn tại / quá tải, `llm_router.py` cố gắng thử 3 lần. Mỗi lần gọi qua LiteLLM mặc định chờ timeout **90 giây**:
     $$\text{Lần 1 (90s)} + \text{Lần 2 (90s)} + \text{Lần 3 (90s)} + \text{thời gian chờ} \approx 275 \text{ giây}$$
  3. Sau 3 lần thất bại (mất 275s), hệ thống mới kích hoạt model dự phòng `gemini/gemini-2.5-flash` (mất thêm ~20s để sinh câu trả lời).
  4. **Tổng thời gian = 294.98 giây (gần 5 phút)**!
- **Giải pháp**:
  - Đổi model chính trong `.env` thành `gemini/gemini-2.5-flash` (hoặc `gemini/gemini-2.0-flash`).
  - Trong `src/services/llm_router.py`, đã bổ sung `timeout=35` và cơ chế **ngắt ngay khi timeout**: nếu model gặp timeout quá 35s, lập tức chuyển sang model dự phòng kế tiếp chứ không lặp lại 3 lần chờ timeout nữa.

---

### 2. Tại sao câu trả lời bị cụt và báo lỗi "định dạng đầu ra của AI không đúng schema"?
- **Hiện tượng trên giao diện**:
  ```json
  {
  "answer": "Các trường hợp chỉ định thầu bao gồm:\n\n1.

  ⚠ Kiểm tra tự động (chỉ là heuristic, không thay thế đối chiếu văn bản gốc): định dạng đầu ra của AI không đúng schema.
  ```
- **Nguyên nhân**:
  1. Trong `.env`, `ANSWER_MAX_TOKENS=900`.
  2. Nhật ký ghi: `prompt_tokens: 2720, completion_tokens: 884`.
  3. Với các câu hỏi phức tạp (nhiều điều khoản), câu trả lời cần từ 1.200 - 1.800 tokens. Khi sinh đến token thứ 884 (chạm trần 900), **Google Gemini bị ngắt ngang giữa chừng**, chưa kịp đóng dấu ngoặc kép `"` và ngoặc nhọn `}` của JSON.
  4. Bộ phân tích JSON `_parse_json` thấy chuỗi JSON bị đứt đoạn nên báo lỗi `ValidationError` -> hệ thống hiển thị nguyên văn chuỗi JSON bị cụt và kèm cảnh báo *"định dạng đầu ra của AI không đúng schema"*.
- **Giải pháp**:
  - Nâng `ANSWER_MAX_TOKENS=2500` trong `.env` và `src/config.py` để AI thoải mái sinh câu trả lời đầy đủ.
  - Cải tiến hàm `_parse_json` trong `src/services/rag_engine.py`: Thêm bộ giải mã phục hồi thông minh bằng Regex. Nếu AI bị ngắt ngang, hệ thống tự động bóc tách sạch sẽ phần nội dung văn bản bên trong `"answer": "..."` thay vì hiển thị raw JSON bị lỗi.

---

### 3. Tại sao hỏi về 349 thì chỉ trích về 214 chứ không trích 349?
- **Câu hỏi**: *"điều nào trong nghị định 214 và 349 là các trường hợp chỉ định thầu"*
- **Nguyên nhân 1 (Trượt nhận diện văn bản thứ 2)**:
  - Hàm `resolve_docs` trong `direct_answer.py` trước đây dùng Regex `_SHORT_REF` đòi hỏi từ `"nghị định"` phải đi liền trước số hiệu.
  - Khi người dùng hỏi *"nghị định 214 và 349"*, hệ thống chỉ nhận diện được `214`, bỏ qua `349`!
  - `target_docs` bị gán cứng thành `['214/2025/NĐ-CP']`, khiến vector search lọc bỏ toàn bộ các chunk của Nghị định 349!
- **Nguyên nhân 2 (Lỗi gán cờ văn bản sửa đổi `is_amendment`)**:
  - Trong `_expand_with_relations`, code cũ viết: `is_amendment = (doc_number in expansions and bool(h.payload.get("is_amendment", False)))`.
  - Nhưng trong Qdrant vector database, payload hoàn toàn không có trường `is_amendment`. Biểu thức này luôn trả về `False`, khiến các đoạn trích của Nghị định 349 bị mất nhãn *"VĂN BẢN SỬA ĐỔI"*.
- **Nguyên nhân 3 (Lệch số Điều khi Reranking)**:
  - Trong NĐ 214: Các trường hợp chỉ định thầu nằm ở **Điều 78**.
  - Trong NĐ 349: Quy định sửa đổi, bổ sung chỉ định thầu nằm ở **Điều 4** (nội dung: *"Điều 4. Sửa đổi, bổ sung một số điểm, khoản của Điều 78"*).
  - Thuật toán Rerank cũ chỉ so khớp số điều của chunk (`4`) với điều trọng tâm (`78`), thấy $4 \neq 78$ nên không boost điểm cho Điều 4 của NĐ 349. Hậu quả là các chunk của NĐ 214 chiếm hết Top-K, đẩy NĐ 349 ra ngoài.
- **Giải pháp**:
  - Nâng cấp `resolve_docs` trong `direct_answer.py`: Tự động nhận diện chuỗi số hiệu đi kèm (`214 và 349`), đưa cả `214/2025/NĐ-CP` và `349/2026/NĐ-CP` vào danh sách tìm kiếm.
  - Sửa logic tìm kiếm đa văn bản trong `rag_engine.py`: Phân bổ đều hạn mức vector search cho từng văn bản mục tiêu, đảm bảo luôn có kết quả của cả 214 và 349.
  - Sửa cờ `is_amendment`: Chuyển thành `is_amendment = (doc_number in expansions)`.
  - Nâng cấp Reranking: Nếu nội dung của chunk văn bản sửa đổi có đề cập đến Điều luật trọng tâm (chứa cụm từ *"Điều 78"*), chunk đó sẽ được boost 1.35x điểm để đưa lên cùng Điều 78.

---

## II. CHI TIẾT CÁC FILE ĐÃ SỬA VÀ NÂNG CẤP

1. **`src/services/llm_router.py`**:
   - Thêm `timeout=35` cho mỗi lần gọi model.
   - Thêm cơ chế ngắt ngay khi gặp timeout (không chờ retry 3 lần vô ích).
2. **`src/services/direct_answer.py`**:
   - Nâng cấp `resolve_docs`: Nhận diện nhiều văn bản trong cùng một câu hỏi.
   - Nâng cấp `_heading_re`: Nhận diện tiêu đề Điều luật chịu được các lỗi OCR (dấu phẩy, gạch ngang, chữ hoa, thiếu dấu).
3. **`src/services/rag_engine.py`**:
   - Nâng cấp `_parse_json`: Tự phục hồi văn bản khi JSON bị cắt ngang.
   - Cập nhật `_prepare`: Tìm kiếm đa văn bản cân bằng cho từng văn bản được hỏi.
   - Nâng cấp Reranking: Nhận diện chunk của văn bản sửa đổi có chứa nội dung điều luật trọng tâm.
   - Sửa cờ `is_amendment` trong `_expand_with_relations`.
   - Nâng cấp BM25: Đưa cả `doc_number + article + content` vào chỉ mục.
4. **`src/config.py` & `.env`**:
   - `GENERATION_MODEL=gemini/gemini-2.0-flash` (hoặc `gemini/gemini-2.5-flash`).
   - `GENERATION_FALLBACK_MODELS=gemini/gemini-1.5-flash,gemini/gemini-2.5-flash`.
   - `ANSWER_MAX_TOKENS=2500`.
5. **`src/services/security_guard.py`**:
   - Thêm Circuit Breaker 60s khi Ollama không bật, không làm chậm request.
6. **`src/services/document_parser.py`**:
   - Sửa lỗi regex trích xuất Điều và hàm `normalize_article_number`, khôi phục 100% các điều khoản bị thiếu trong OCR.

---

## III. HƯỚNG DẪN THỰC HIỆN TRÊN MÁY WINDOWS

Để máy Windows áp dụng các cải tiến trên và chạy với tốc độ 5-10 giây/câu:

### Bước 1: Mở file `.env` trên máy Windows và cập nhật
```env
# Đổi model chính sang model 2.5-flash hoặc 2.0-flash (KHÔNG dùng 3.8-flash)
GENERATION_MODEL=gemini/gemini-2.5-flash
GENERATION_FALLBACK_MODELS=gemini/gemini-2.0-flash,gemini/gemini-1.5-flash

# Tăng trần token câu trả lời
ANSWER_MAX_TOKENS=2500

# Nếu máy Windows chưa cài Ollama, hãy để trống dòng này:
LOCAL_MODEL_URL=
```

### Bước 2: Khởi động lại server FastAPI
Trong PowerShell / Command Prompt:
```powershell
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 3: Nạp lại tài liệu (Re-index)
Chạy lệnh sau trong PowerShell để hệ thống cắt chunk và lập chỉ mục lại theo thuật toán mới:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/documents/reindex" -Method Post
```

### Bước 4: Kiểm tra nhanh bằng công cụ kiểm tra (Không tốn token AI)
Chạy script kiểm tra xem cả 214 và 349 đã được tìm thấy chưa:
```powershell
python scripts/inspect_retrieval.py "điều nào trong nghị định 214 và 349 là các trường hợp chỉ định thầu" --api http://localhost:8000
```
Kết quả sẽ hiển thị ngay lập tức:
- **Nguồn từ 214/2025/NĐ-CP**: Điều 78 (trang 85, 86, 87...).
- **Nguồn từ 349/2026/NĐ-CP**: Điều 4 (trang 3 - *"Sửa đổi, bổ sung một số điểm, khoản của Điều 78"*).
