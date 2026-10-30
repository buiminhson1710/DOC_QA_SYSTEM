import re
from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text

pattern = re.compile(r"Điều\s+(\d+[a-z]?)\s*[.:]\s*([^\n]{0,180})", re.IGNORECASE)
for m in pattern.finditer(text):
    print(f"Điều {m.group(1)}: {m.group(2).strip()}")
