from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)

print("Số hiệu văn bản:", parsed.doc_number)
print("Các quan hệ được trích xuất:")
for r in parsed.relations:
    print(f"- {r.type} | {r.target} | {r.scope} | {r.evidence}")
