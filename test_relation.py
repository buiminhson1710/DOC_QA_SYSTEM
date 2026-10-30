from src.services.document_parser import extract_relations

text = """Chương I: Sửa đổi bổ sung một số điều của Nghị định số 214/2025/NĐ-CP ngày 04 tháng 8 năm 2025 của Chính phủ quy định chi tiết một số điều và biện pháp thi hành Luật Đấu thầu về lựa chọn nhà thầu (được sửa đổi bổ sung bởi Nghị định số 165/2026/NĐ-CP ngày 15 tháng 5 năm 2026 của Chính phủ và Nghị định số 170/2026/NĐ-CP ngày 15 tháng 5 năm 2026 của Chính phủ)."""

relations = extract_relations(text, "349/2026/NĐ-CP")

for r in relations:
    print(f"Loại: {r.type} | Văn bản: {r.target} | Phạm vi: {r.scope}")
