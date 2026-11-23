from __future__ import annotations

from tools.research import ResearchEngine, build_match_query, chunk_text


TEXT_ONE = (
    "BM25 ranks documents using term frequency, inverse document frequency and "
    "document length normalisation. The parameter k1 controls term-frequency "
    "saturation and b controls length normalisation. SQLite FTS5 exposes BM25 "
    "through the bm25() auxiliary function, and better matches have a more "
    "negative score. Ranking queries should order by bm25(...) ascending."
)

TEXT_TWO = (
    "Weight-only quantisation stores linear weights in eight or four bits with "
    "one scale per output channel. It reduces checkpoint size but does not "
    "automatically make inference faster, because a naive implementation "
    "dequantises the weights on every forward pass."
)


def test_ingest_and_search_with_attribution(tmp_path):
    engine = ResearchEngine(db_path=tmp_path / "research.db")
    first = engine.ingest_text(TEXT_ONE, "bm25.md", title="BM25 notes")
    second = engine.ingest_text(TEXT_TWO, "quant.md", title="Quantisation notes")
    assert first["status"] == "added" and first["chunks"] >= 1
    assert second["status"] == "added"

    hits = engine.search("BM25 ranking", limit=3)
    assert hits
    top = hits[0]
    assert top.path == "bm25.md"
    assert top.title == "BM25 notes"
    assert top.score > 0
    assert "BM25" in top.text
    citation = top.citation()
    assert citation["path"] == "bm25.md"
    assert citation["end_char"] > citation["start_char"]

    quant_hits = engine.search("quantisation checkpoint", limit=3)
    assert quant_hits and quant_hits[0].path == "quant.md"


def test_empty_index_returns_no_results(tmp_path):
    engine = ResearchEngine(db_path=tmp_path / "empty.db")
    assert engine.search("anything") == []
    assert engine.search("") == []
    assert engine.stats()["documents"] == 0


def test_unmatched_query_and_operator_injection(tmp_path):
    engine = ResearchEngine(db_path=tmp_path / "safe.db")
    engine.ingest_text(TEXT_ONE, "bm25.md")
    assert engine.search("zebra quantum") == []
    # FTS5 operators and quotes must not raise.
    assert isinstance(engine.search('BM25 OR "unterminated'), list)
    assert build_match_query("a b") == '"a" AND "b"'
    # Operator words from free text are quoted as literal terms.
    assert build_match_query("a AND b") == '"a" AND "AND" AND "b"'


def test_reingest_replaces_chunks_and_remove(tmp_path):
    engine = ResearchEngine(db_path=tmp_path / "replace.db")
    engine.ingest_text(TEXT_ONE, "doc.md")
    before = engine.stats()
    updated = engine.ingest_text(TEXT_ONE + "\n\nExtra paragraph about ranking.", "doc.md")
    assert updated["status"] == "replaced"
    after = engine.stats()
    assert after["documents"] == before["documents"] == 1
    assert after["chunks"] >= before["chunks"]
    hits = engine.search("extra paragraph ranking")
    assert hits and "Extra paragraph" in hits[0].text
    assert engine.remove_document("doc.md") is True
    assert engine.remove_document("doc.md") is False
    assert engine.list_documents() == []


def test_chunking_preserves_offsets_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, chunk_chars=300, overlap=60)
    assert len(chunks) > 1
    for start, chunk in chunks:
        assert text[start : start + len(chunk)] == chunk
    assert all(len(chunk) <= 400 for _, chunk in chunks)
    assert chunks[0][0] == 0
    # Consecutive chunks overlap.
    for (start_a, chunk_a), (start_b, _) in zip(chunks, chunks[1:]):
        assert start_b < start_a + len(chunk_a)
