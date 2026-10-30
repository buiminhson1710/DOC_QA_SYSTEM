import re
import unicodedata
from pathlib import Path
from src.services.document_parser import parse_file

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
text = parse_file(pdf).full_text
flat = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text))

chapters = list(re.finditer(r"(?i)\bch\u01b0\u01a1ng\s+\S+", flat))
print("Số tiêu đề Chương thực sự tìm thấy:", len(chapters))

num_pattern = re.compile(
    r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[a-zđ]{1,6}\d{0,2}(?:-[a-zđ]{1,8})*",
    re.IGNORECASE,
)

for i, chapter in enumerate(chapters):
    rest = flat[chapter.end():]
    next_article = re.search(
        r"(?i)\b\u0111i\u1ec1u\s+\d+[a-z]?\s*[.:]",
        rest,
    )
    heading_end = (
        chapter.end() + next_article.start()
        if next_article
        else min(len(flat), chapter.start() + 1800)
    )
    heading = flat[chapter.start():heading_end]

    normalized = unicodedata.normalize("NFD", heading).casefold()
    normalized = "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    ).replace("đ", "d")

    target = num_pattern.search(heading)

    print(f"\n--- Chương {i + 1} ---")
    print("Tiêu đề:", heading[:300])
    print("Có 'sua doi':", "sua doi" in normalized)
    print("Có 'bo sung':", "bo sung" in normalized)
    print("Số hiệu đầu tiên:", target.group(0) if target else None)
