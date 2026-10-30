import re
import unicodedata
from pathlib import Path
from src.services.document_parser import parse_file, DOC_NUM_RE, extract_relations

pdf = Path(r".\storage\docs\Nghi dinh 349_2026_ND-CP pdf.pdf")
parsed = parse_file(pdf)
text = parsed.full_text

chapter = re.search(r"(?i)\bchương\s+\S+", text)
print("1. Tìm thấy tiêu đề chương:", bool(chapter))

if chapter:
    remaining = text[chapter.start():]
    article = re.search(r"(?i)\bđiều\s+1\s*[.:]", remaining)
    print("2. Tìm thấy Điều 1:", bool(article))

    end = chapter.start() + article.start() if article else min(len(text), chapter.start() + 1800)
    heading = text[chapter.start():end]
    normalized = unicodedata.normalize("NFD", heading).casefold()
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn").replace("đ", "d")

    print("3. Nhận diện 'sua doi':", "sua doi" in normalized)
    print("4. Nhận diện 'bo sung':", "bo sung" in normalized)
    m = DOC_NUM_RE.search(heading)
    print("5. Số hiệu đầu tiên trong tiêu đề:", repr(m.group(1) if m else None))
    print("6. Tiêu đề đã chuẩn hóa:", normalized[:250])

print("7. Quan hệ cuối cùng:")
for r in extract_relations(text, parsed.doc_number):
    print(r.type, "|", r.target, "|", r.scope)
