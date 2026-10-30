from pathlib import Path
import re
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text

m = re.search(r"214\s*/\s*2025", text, re.IGNORECASE)
if m:
    print("Vị trí xuất hiện đầu tiên của 214/2025:")
    print(text[max(0, m.start()-500):min(len(text), m.end()+300)])
else:
    print("Không tìm thấy mẫu 214/2025 trong văn bản OCR.")
