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