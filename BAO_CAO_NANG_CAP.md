# Báo cáo nâng cấp

> Phạm vi: nâng cấp độ chính xác, khả năng đo lường và kiểm chứng. Mọi số liệu dưới đây là số thật đo được trong lần bàn giao này,
> trên **dữ liệu tổng hợp với embedding/LLM giả lập** (môi trường bàn giao không có khoá API). Chưa có số đo trên 3 PDF thật của bạn.

## 1. Chạy trong VS Code
1. Mở thư mục `doc_qa_system` bằng VS Code (File → Open Folder).
2. Terminal (PowerShell): `python -m venv .venv` → `.venv\Scripts\Activate.ps1` → `python -m pip install -r requirements-dev.txt`
3. Chép `.env.example` thành `.env` rồi điền khoá/model **đúng như `.env` cũ của bạn** (xem mục 6). Chạy server: `F5` (cấu hình "Chạy server") hoặc `python -m uvicorn src.main:app --reload`.
4. Test: `python -m pytest tests` (hoặc biểu tượng Testing của VS Code). Đánh giá RAG: `python scripts/eval_rag.py inprocess`.
5. Sau khi chạy: `http://localhost:8000/api/health` (cấu hình/OCR/model tạm ngưng), `/api/diagnostics/documents` (độ đầy đủ từng văn bản), `python scripts/analyze_trace.py` (tóm tắt log).
6. Lần đầu sau khi nâng cấp: gọi `POST /api/documents/resync` một lần (cập nhật trạng thái `active` cũ → `unverified` trong Qdrant; chỉ metadata, không đụng vector).

## 2. Giữ nguyên
Model embedding, LLM, OCR; số chiều vector; collection Qdrant (`legal_docs_v3`). Không xoá dữ liệu, không re-embedding, không migration. Ghi chunk giờ là upsert rồi mới xoá chunk thừa (ID xác định → không trùng, không có khoảng mất dữ liệu).

## 3. File đã sửa/thêm và lý do (so với bản bạn đang chạy)
- `.env.example` — các biến mới
- `README.md` — hướng dẫn chẩn đoán/đo lường
- `pytest.ini  (MỚI)` — mới
- `requirements-dev.txt  (MỚI)` — mới
- `src/config.py` — thêm cấu hình: DATA_DIR, embedding (batch/timeout/retry/cache), hybrid, ghim điều khoản, OCR, trace, công tắc tính năng; cảnh báo cấu hình sai
- `src/evaluation.py  (MỚI)` — mới
- `src/main.py` — endpoint chẩn đoán (/api/diagnostics/*, /api/documents/resync), health hiển thị OCR/model tạm ngưng/cảnh báo cấu hình
- `src/services/direct_answer.py` — sửa đổi
- `src/services/document_parser.py` — OCR (Tesseract local / AI theo lô, cache từng trang, tiếp tục), độ tin cậy, phát hiện thiếu 'Điều', phạm vi quan hệ, sửa số hiệu theo tên file, nhãn điều của chunk đầu
- `src/services/lexical.py  (MỚI)` — mới
- `src/services/llm_router.py` — sửa đổi
- `src/services/qdrant_service.py` — embedding: phân loại lỗi, retry có giới hạn (backoff+jitter), cache sqlite để tiếp tục, kiểm tra dimension; upsert-rồi-xoá-thừa an toàn
- `src/services/rag_engine.py` — pipeline chính: ingest có ingest_id + kiểm chứng + hoàn tác; truy xuất lai BM25+RRF; ghim điều khoản; prompt v3; kiểm tra schema/citation; trace; trả lời trực tiếp/cache/trích đoạn
- `src/services/relations_service.py` — trạng thái 'unverified', nối số hiệu chịu lỗi OCR, phạm vi quan hệ, lịch sử, hoàn tác, resync
- `src/services/security_guard.py` — chấm điểm theo tổ hợp khái niệm + chuẩn hoá chống né + lọc đầu ra (canary, bỏ qua câu từ chối hợp lệ)
- `src/services/trace.py  (MỚI)` — mới
- `static/index.html` — hiển thị 'chưa xác minh hiệu lực', link văn bản sửa đổi, URL bấm được
- `tests/`, `scripts/`, `eval/`, `pytest.ini`, `requirements-dev.txt`, `.vscode/` — **MỚI**: bộ test, script đánh giá/phân tích log, báo cáo, cấu hình VS Code.

## 4. Tính năng đã thực sự tích hợp vào pipeline
- Ingestion: thống kê trang/OCR lỗi/rỗng/độ tin cậy, chunk, vector, dimension, số điểm Qdrant thực tế, thời gian từng bước, retry, lỗi; `ingest_id`; trạng thái `complete | complete_with_warnings | incomplete | failed`; **tiếp tục được** (cache OCR từng trang + cache embedding) mà không tạo trùng; hoàn tác registry nếu lưu Qdrant lỗi.
- Embedding: phân loại 429 theo ngày (không retry) / 429 ngắn hạn / sai khoá / mạng / đầu vào; retry có giới hạn với exponential backoff + jitter; batch/timeout cấu hình được; kiểm tra số lượng, dimension, vector hỏng.
- Retrieval: hybrid vector + BM25 + RRF; mở rộng sang văn bản sửa đổi/thay thế/bãi bỏ; ghim nguyên văn điều khoản khi câu hỏi nêu "Điều N". Trace ghi top-k, ID, hạng, điểm từng nhánh.
- Trace: `request_id` mỗi câu hỏi, `ingest_id` mỗi lần nạp; model, phiên bản prompt, token thật (nếu nhà cung cấp trả) **tách khỏi** token ước tính, thời gian từng bước, số lần thử, lỗi; che khoá/token; tuỳ chọn ẩn nội dung (`TRACE_LOG_TEXT=false`).
- Prompt v3 + kiểm tra đầu ra: schema (pydantic), citation `[n]`, đối chiếu số liệu theo cặp (số, đơn vị), số hiệu/điều/trang, khẳng định hiệu lực. **Đây là heuristic phát hiện tham chiếu không có trong nguồn, KHÔNG phải bằng chứng câu trả lời đúng về pháp lý.**
- Quan hệ/hiệu lực: phân biệt sửa đổi một phần / bãi bỏ một phần / thay thế toàn bộ / bãi bỏ; trạng thái mặc định **`unverified` (chưa xác minh)** — không suy ra "còn hiệu lực" chỉ vì chưa thấy văn bản bãi bỏ; lưu lịch sử nạp.

## 5. Kết quả kiểm thử (chạy thật)
```
SKIPPED tests/test_live.py:17: đặt RUN_LIVE_TESTS=1 và khoá API thật trong .env để chạy
SKIPPED tests/test_live.py:26: đặt RUN_LIVE_TESTS=1 và khoá API thật trong .env để chạy
SKIPPED tests/test_live.py:39: đặt RUN_LIVE_TESTS=1 và khoá API thật trong .env để chạy
SKIPPED tests/test_live.py:51: đặt thêm RUN_LIVE_E2E=1 để chạy đầu-cuối (tốn token)
SKIPPED tests/test_parser_ingestion.py:152: Tesseract hoặc dữ liệu ngôn ngữ 'vie' chưa cài (đặt TESSDATA_PREFIX nếu vie nằm ở thư mục riêng)
162 passed, 5 skipped, 8 xfailed in 14.77s
```
- **0 fail.** `skipped`: test dùng dịch vụ thật (cần `RUN_LIVE_TESTS=1` + khoá API thật), và test Tesseract khi máy chưa cài dữ liệu tiếng Việt (đã chạy đạt khi có `vie`).
- `xfailed` (đã biết, ghi rõ trong test): 7 mẫu tấn công mà bộ lọc heuristic không bắt (cần Ollama) và 1 câu đánh giá M6 (nối gián tiếp "miễn báo cáo" → "bãi bỏ khoản 3 Điều 9").
- Test bắt được và đã sửa 4 lỗi thật: ghi trace có thể âm thầm không ra file; câu từ chối hợp lệ của AI bị chặn nhầm là "lộ prompt"; kiểm tra số liệu bỏ lọt giá trị bị đổi ("10 ngày" vs "15 ngày"); chunk đầu văn bản mất nhãn điều.

## 6. Đánh giá RAG (kho tổng hợp, embedding giả lập, so với baseline)
> Embedding GIẢ LẬP (băm n-gram ký tự) + kho tổng hợp: đo cơ chế pipeline, không dự đoán chất lượng embedding thật.

| Cấu hình | R@1 | R@3 | R@5 | MRR | Phủ bằng chứng (ngữ cảnh) | Chỉ có giá trị cũ | Mới được đánh dấu sửa đổi | Token ngữ cảnh TB | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| vector_only | 0.5 | 0.672 | 0.759 | 0.629 | 0.793 | 0.25 | 0.0 | 1938 | 2.8 |
| baseline_prev | 0.5 | 0.672 | 0.759 | 0.629 | 0.862 | 0.0 | 1.0 | 2479 | 2.3 |
| hybrid | 0.569 | 0.793 | 0.879 | 0.722 | 0.897 | 0.0 | 1.0 | 2679 | 2.3 |
| full | 0.569 | 0.793 | 0.879 | 0.722 | 0.897 | 0.0 | 1.0 | 2703 | 2.1 |

**Theo nhóm — Recall@3 (truy xuất) / Phủ bằng chứng (ngữ cảnh)**

| Nhóm | vector_only | baseline_prev | hybrid | full |
|---|---|---|---|---|
| amended | 0.5 / 0.667 | 0.5 / 0.833 | 0.667 / 0.833 | 0.667 / 0.833 |
| amended_long | 0.333 / 0.333 | 0.333 / 0.667 | 0.333 / 1.0 | 0.333 / 1.0 |
| lookup | 0.75 / 1.0 | 0.75 / 1.0 | 0.875 / 1.0 | 0.875 / 1.0 |
| multi_doc | 0.5 / 0.333 | 0.5 / 0.333 | 0.667 / 0.667 | 0.667 / 0.667 |
| ocr_noisy | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| paraphrase | 0.667 / 1.0 | 0.667 / 1.0 | 1.0 / 0.667 | 1.0 / 0.667 |
| relation | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

Cách đọc: `baseline_prev` ≈ hành vi trước nâng cấp (chỉ vector + mở rộng sang văn bản sửa đổi); `vector_only` là mức tối thiểu; `hybrid` thêm BM25+RRF; `full` thêm ghim điều khoản. R@K/MRR đo bước truy xuất; "Phủ bằng chứng" đo những gì thực sự gửi cho AI.
- **Có số đo cải thiện** (hybrid so với baseline_prev): Recall@3 0.672 → 0.793, MRR 0.629 → 0.722, phủ bằng chứng trong ngữ cảnh 0.862 → 0.897. Mạnh nhất ở nhóm văn bản sửa đổi dài (`amended_long`: 0.667 → 1.0) và nhiều văn bản (`multi_doc`: 0.333 → 0.667).
- **Mở rộng sang văn bản sửa đổi** đưa tỉ lệ "ngữ cảnh chỉ có giá trị cũ" ở nhóm `amended` từ 0.25 (vector_only) về 0.0.
- **Chưa có bằng chứng lợi ích của "ghim điều khoản"** trên bộ này (full cho chỉ số y hệt hybrid, chỉ tốn thêm ~24 token); bật/tắt bằng `PIN_ARTICLES`.
- **Đổi lại:** ngữ cảnh gửi AI dài hơn khoảng 8–9% so với baseline_prev (2479 → 2679–2703 token ước tính); và nhóm `paraphrase` bị **giảm** phủ bằng chứng ở hybrid (1.0 → 0.667) vì giới hạn ngân sách ngữ cảnh cắt bớt nguồn — cần theo dõi khi có embedding thật (có thể tăng `MAX_CONTEXT_CHARS`).
- **Không đo được ở đây** (cần LLM + embedding thật): độ đúng câu trả lời, groundedness, khả năng từ chối của AI, token thật. Điểm tương đồng cũng **không tách được** câu "không có dữ liệu" khỏi câu có đáp án (nên không dùng ngưỡng điểm để từ chối). Chạy bộ đo thật: `python scripts/eval_rag.py live --url http://localhost:8000 --golden eval/golden_real_template.json` (điền mẫu bằng nguyên văn lấy từ văn bản của bạn và tự kiểm chứng).

## 7. Mẫu log
`eval/sample_logs/trace.jsonl`, `ingest.jsonl` (sinh từ LLM/embedding giả lập, chỉ để minh hoạ định dạng) và `python scripts/analyze_trace.py --logs eval/sample_logs`.

## 8. Vấn đề còn tồn tại
- Bộ chặn tấn công: riêng heuristic chỉ bắt ~55% mẫu chưa từng thấy; cần chạy Ollama (`ollama pull qwen2.5:3b`) để có lớp ngữ nghĩa — chất lượng lớp này chưa được đo ở đây.
- OCR Tesseract trên bản scan thật có thể kém Gemini; hệ thống cảnh báo trang độ tin cậy thấp và thiếu "Điều" nhưng không tự sửa chữ sai. Hãy đọc lại các điều quan trọng với bản gốc.
- Hiệu lực văn bản: mới có trạng thái chưa xác minh + lịch sử nạp; chưa có luồng xác minh với nguồn chính thức.
- Câu hỏi cần nối gián tiếp giữa điều gốc và mục sửa đổi (M6) chưa giải quyết được.
- Cache câu trả lời/ngữ cảnh nằm trong bộ nhớ tiến trình (nhiều worker cần Redis).

## 9. Rollback
- Toàn bộ: giải nén `rollback/BASELINE_truoc_nang_cap.zip` (bản bạn đang chạy trước nâng cấp). Dữ liệu Qdrant không bị đổi nên dùng lại được.
- Từng phần bằng `.env`: `HYBRID_SEARCH=false`, `PIN_ARTICLES=false`, `EXPAND_AMENDMENTS=false`, `EMBED_CACHE=false`, `DIRECT_ANSWERS=false`.
