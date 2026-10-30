import re
from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)
text = parsed.full_text

matches = list(re.finditer(r"60\s*/\s*2021", text, re.IGNORECASE))
print("Matches for 60/2021:", len(matches))

for m in matches:
    start = max(0, m.start() - 220)
    end = min(len(text), m.end() + 300)
    print("\n--- Context ---")
    print(text[start:end].replace("\n", " "))

print("\nExtracted relations:")
for r in parsed.relations:
    print(r.type, "|", r.target, "|", r.scope)
