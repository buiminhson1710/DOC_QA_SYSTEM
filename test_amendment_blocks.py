from pathlib import Path
from src.services.document_parser import parse_file
from src.services import direct_answer as DA

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)

blocks = DA.extract_amendment_blocks(parsed.full_text, "78")

print("Số đoạn tìm được:", len(blocks))

for i, (pos, block) in enumerate(blocks, 1):
    print(f"\n--- Kết quả {i} | vị trí {pos} ---")
    print(block[:1200])
