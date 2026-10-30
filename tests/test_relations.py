"""Unit test relations_service: không nhầm sửa đổi/thay thế/bãi bỏ; trạng thái 'chưa xác minh'; chịu lỗi OCR; hoàn tác; lịch sử."""
from __future__ import annotations

from src.services.document_parser import Relation
from src.services.relations_service import RelationsService


class DummyQdrant:
    def __init__(self):
        self.payloads: dict[str, dict] = {}

    def set_doc_payload(self, doc, payload):
        self.payloads[doc] = payload


def add(svc, number, relations=(), filename=None):
    return svc.upsert_document(doc_number=number, filename=filename or f"{number.replace('/', '_')}.pdf", file_url=f"http://x/{number}",
                               title=None, doc_type="Nghị định", chunk_count=3, relations=list(relations), pages_read=2, pages_total=2)


def make(cfg):
    q = DummyQdrant()
    return RelationsService(q, cfg), q


def test_default_status_is_unverified_not_active(cfg):
    svc, _ = make(cfg)
    add(svc, "101/2030/NĐ-CP")
    p = svc.payload_for("101/2030/NĐ-CP")
    assert p["status"] == "unverified" and p["amended_by"] == [] and p["repealed_by"] == []


def test_amend_replace_repeal_set_distinct_status_and_fields(cfg):
    svc, _ = make(cfg)
    for n in ("1/2020/NĐ-CP", "2/2020/NĐ-CP", "3/2020/NĐ-CP", "4/2021/NĐ-CP", "5/2021/NĐ-CP", "6/2021/NĐ-CP"):
        add(svc, n)
    add(svc, "4/2021/NĐ-CP", [Relation("amends", "1/2020/NĐ-CP", "ev", "partial")])
    add(svc, "5/2021/NĐ-CP", [Relation("replaces", "2/2020/NĐ-CP", "ev", "full")])
    add(svc, "6/2021/NĐ-CP", [Relation("repeals", "3/2020/NĐ-CP", "ev", "full")])
    assert svc.payload_for("1/2020/NĐ-CP")["status"] == "amended" and svc.payload_for("1/2020/NĐ-CP")["amended_by"] == ["4/2021/NĐ-CP"]
    assert svc.payload_for("2/2020/NĐ-CP")["status"] == "replaced" and svc.payload_for("2/2020/NĐ-CP")["replaced_by"] == ["5/2021/NĐ-CP"]
    assert svc.payload_for("3/2020/NĐ-CP")["status"] == "repealed" and svc.payload_for("3/2020/NĐ-CP")["repealed_by"] == ["6/2021/NĐ-CP"]
    assert svc.payload_for("1/2020/NĐ-CP")["replaced_by"] == [] and svc.payload_for("1/2020/NĐ-CP")["repealed_by"] == []


def test_partial_repeal_does_not_mark_whole_document_repealed(cfg):
    svc, _ = make(cfg)
    add(svc, "101/2030/NĐ-CP")
    add(svc, "202/2031/NĐ-CP", [Relation("amends", "101/2030/NĐ-CP", "bãi bỏ khoản 3 Điều 9", "partial_repeal")])
    p = svc.payload_for("101/2030/NĐ-CP")
    assert p["status"] == "amended" and p["repealed_by"] == []


def test_status_priority_repealed_over_replaced_over_amended(cfg):
    svc, _ = make(cfg)
    add(svc, "1/2020/NĐ-CP")
    add(svc, "2/2021/NĐ-CP", [Relation("amends", "1/2020/NĐ-CP", "", "partial")])
    assert svc.payload_for("1/2020/NĐ-CP")["status"] == "amended"
    add(svc, "3/2021/NĐ-CP", [Relation("replaces", "1/2020/NĐ-CP", "", "full")])
    assert svc.payload_for("1/2020/NĐ-CP")["status"] == "replaced"
    add(svc, "4/2021/NĐ-CP", [Relation("repeals", "1/2020/NĐ-CP", "", "full")])
    assert svc.payload_for("1/2020/NĐ-CP")["status"] == "repealed"


def test_reverse_ingest_order_gives_same_result(cfg):
    a, _ = make(cfg)
    add(a, "349/2026/NĐ-CP", [Relation("amends", "214/2025/NĐ-CP", "", "partial")])
    add(a, "214/2025/NĐ-CP")
    assert a.payload_for("214/2025/NĐ-CP")["amended_by"] == ["349/2026/NĐ-CP"]


def test_target_referenced_but_not_ingested_still_has_incoming_relations(cfg):
    svc, _ = make(cfg)
    add(svc, "359/2026/NĐ-CP", [Relation("repeals", "53/2013/NĐ-CP", "ev", "full")])
    p = svc.payload_for("53/2013/NĐ-CP")
    assert p["repealed_by"] == ["359/2026/NĐ-CP"] and p["status"] == "repealed"
    assert svc.file_url_of("53/2013/NĐ-CP") is None                       # chưa nạp => không có link


def test_ocr_garbled_suffix_is_matched_but_other_document_types_are_not(cfg):
    svc, _ = make(cfg)
    add(svc, "214/2025/NĐ-CP")
    add(svc, "349/2026/NĐ-CP", [Relation("amends", "214/2025/NB-CP", "", "partial"), Relation("amends", "214/2025/TT-BTC", "", "partial")])
    assert svc.payload_for("214/2025/NĐ-CP")["amended_by"] == ["349/2026/NĐ-CP"]
    amends = svc.payload_for("349/2026/NĐ-CP")["amends"]
    assert "214/2025/NĐ-CP" in amends and "214/2025/TT-BTC" in amends     # TT-BTC không bị nối nhầm sang NĐ-CP


def test_history_is_kept_and_capped(cfg):
    svc, _ = make(cfg)
    for _ in range(25):
        add(svc, "101/2030/NĐ-CP")
    rec = svc.get_record("101/2030/NĐ-CP")
    assert len(rec["history"]) == 20 and rec["history"][-1]["pages"] == "2/2"


def test_restore_record_rolls_back(cfg):
    svc, _ = make(cfg)
    add(svc, "1/2020/NĐ-CP")
    snap = svc.get_record("1/2020/NĐ-CP")
    add(svc, "1/2020/NĐ-CP", [Relation("amends", "9/2019/NĐ-CP", "", "partial")])
    svc.restore_record("1/2020/NĐ-CP", snap)
    assert svc.get_record("1/2020/NĐ-CP")["outgoing"] == []
    svc.restore_record("2/2020/NĐ-CP", None)                              # bản ghi chưa từng có => không lỗi
    assert svc.get_record("2/2020/NĐ-CP") is None


def test_persistence_and_resync_only_ingested_documents(cfg):
    svc, q = make(cfg)
    add(svc, "101/2030/NĐ-CP")
    add(svc, "202/2031/NĐ-CP", [Relation("amends", "101/2030/NĐ-CP", "", "partial"), Relation("repeals", "77/2018/NĐ-CP", "", "full")])
    reloaded = RelationsService(DummyQdrant(), cfg)
    assert reloaded.payload_for("101/2030/NĐ-CP")["status"] == "amended"
    synced = svc.resync_all()
    assert sorted(synced) == ["101/2030/NĐ-CP", "202/2031/NĐ-CP"] and "77/2018/NĐ-CP" not in q.payloads
    assert q.payloads["101/2030/NĐ-CP"]["status"] == "amended"
