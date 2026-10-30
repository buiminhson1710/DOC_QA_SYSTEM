from pathlib import Path
import re
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)
text = parsed.full_text

print("Tổng ký tự văn bản:", len(text))
matches = list(re.finditer(r"214", text, re.IGNORECASE))
print("Số lần xuất hiện của chuỗi 214:", len(matches))

for m in matches:
    start = max(0, m.start() - 180)
    end = min(len(text), m.end() + 350)
    print("\n--- Vị trí", m.start(), "---")
    print(text[start:end].replace("\n", " "))
