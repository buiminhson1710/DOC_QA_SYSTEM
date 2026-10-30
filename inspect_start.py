from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)

print("===== 6000 KÝ TỰ ĐẦU VĂN BẢN OCR =====")
print(repr(parsed.full_text[:6000]))
