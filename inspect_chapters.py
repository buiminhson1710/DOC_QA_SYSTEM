from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text
lines = text.splitlines()

for i, line in enumerate(lines):
    if line.strip().startswith("Chương") or line.strip().startswith("CHƯƠNG"):
        print(f"\n--- Dòng {i+1} ---")
        print("\n".join(lines[i:i+6]))
