import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine

from backend import db


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "engine", create_engine(f"sqlite:///{tmp_path / 'test.db'}"))
    db.Base.metadata.create_all(db.engine)


def test_save_list_and_clear_runs(temp_db):
    first = db.save_run("optimization", "2024-04-15T07:00:00", {"greens": [20, 30, 20, 34]})
    db.save_run("comparison", None, {"n_seeds": 3})
    assert first == 1
    rows = db.list_runs()
    assert [r["kind"] for r in rows] == ["comparison", "optimization"]      # newest first
    assert db.list_runs(kind="optimization")[0]["result"]["greens"] == [20, 30, 20, 34]
    assert db.clear_runs() == 2 and db.list_runs() == []


def test_users_are_unique(temp_db):
    assert db.create_user("sanat", "hash-1") is True
    assert db.create_user("sanat", "hash-2") is False            # duplicate username rejected
    assert db.get_password_hash("sanat") == "hash-1"
    assert db.get_password_hash("nobody") is None
