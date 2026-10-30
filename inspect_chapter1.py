from pathlib import Path
import re
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text

for m in re.finditer(r"chương\s+i\b", text, re.IGNORECASE):
    print("\n===== ĐOẠN OCR TỪ CHƯƠNG I =====")
    print(text[m.start():m.start() + 1800])
    break
else:
    print("Không tìm thấy cụm Chương I trong văn bản OCR.")
