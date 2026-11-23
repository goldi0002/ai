from __future__ import annotations

import pytest

from memory.store import MemoryStore


def test_save_and_get(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    entry = store.save("user_timezone", "Europe/Berlin", tags="preference,time", importance=0.8)
    assert entry["status"] == "created"
    fetched = store.get("user_timezone")
    assert fetched["content"] == "Europe/Berlin"
    assert fetched["tags"] == "preference,time"
    assert fetched["importance"] == 0.8
    assert fetched["created_at"] == fetched["updated_at"]


def test_save_same_key_updates(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    first = store.save("project", "Atlas")
    second = store.save("project", "Atlas 2", importance=0.9)
    assert second["status"] == "updated"
    assert second["created_at"] == first["created_at"]
    assert store.get("project")["content"] == "Atlas 2"
    assert len(store.list()) == 1


def test_search_and_context(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.save("user_timezone", "Europe/Berlin", tags="preference")
    store.save("editor", "Neovim", tags="tool")
    store.save("project", "Atlas compiler", tags="project")
    hits = store.search("timezone", limit=5)
    assert hits and hits[0]["key"] == "user_timezone"
    context = store.to_context("Atlas", limit=3)
    assert "project" in context and "Atlas" in context
    assert store.to_context("nothingmatcheshere") == ""


def test_update_and_delete(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.save("note", "v1")
    updated = store.update("note", content="v2")
    assert updated["content"] == "v2"
    with pytest.raises(KeyError):
        store.update("missing", content="x")
    assert store.delete("note") is True
    assert store.delete("note") is False
    assert store.get("note") is None


def test_validation(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    with pytest.raises(ValueError):
        store.save("", "x")
    with pytest.raises(ValueError):
        store.save("k", "   ")
    with pytest.raises(ValueError):
        store.save("k" * 200, "x")
    with pytest.raises(ValueError):
        store.save("k", "x" * 5000)


def test_ordering_uses_importance(tmp_path):
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.save("low", "low importance", importance=0.1)
    store.save("high", "high importance", importance=0.95)
    entries = store.list()
    assert [entry["key"] for entry in entries] == ["high", "low"]
