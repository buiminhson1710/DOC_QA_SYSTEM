from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)
text = parsed.full_text

for i, line in enumerate(text.splitlines()):
    if "214" in line or "Chương I" in line or "Chuong I" in line:
        print(f"\n--- Dòng {i + 1} ---")
        lines = text.splitlines()
        print("\n".join(lines[max(0, i-2):min(len(lines), i+4)]))
