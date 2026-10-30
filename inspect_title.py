from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)

print("===== 4000 KÝ TỰ ĐẦU VĂN BẢN =====")
print(parsed.full_text[:4000])
