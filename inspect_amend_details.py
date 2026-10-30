import re
from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text

pattern = re.compile(r"(?im)^[ \t]*Điều\s+(\d+)\s*\.")
matches = list(pattern.finditer(text))
wanted = {"2", "8", "11", "19"}

for i, m in enumerate(matches):
    number = m.group(1)
    if number not in wanted:
        continue

    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    print(f"\n{'=' * 25} ĐIỀU {number} {'=' * 25}")
    print(text[m.start():min(end, m.start() + 1800)])
