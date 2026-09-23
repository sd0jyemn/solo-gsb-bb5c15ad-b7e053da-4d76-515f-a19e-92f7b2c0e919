"""End-to-end tests for the read-only arrival-latency profile endpoint."""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.time_utils import store_event_time


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c, db_path


def create_session(client, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": ["a", "b", "empty"],
            "allowed_lateness_seconds": 10,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def profile(client, sid, **params):
    return client.get(f"/sessions/{sid}/arrival-latency-profile", params=params)


def ts(value: str) -> str:
    return store_event_time(datetime.fromisoformat(value.replace("Z", "+00:00")))


def insert_reading(db_path, *, source, seq, observed_at, received_at, status):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO readings (session_id, source_id, seq, observed_at,"
            " payload, status, first_received_at, published_watermark,"
            " quarantined_watermark)"
            " VALUES ('s', ?, ?, ?, NULL, ?, ?, NULL, NULL)",
            (source, seq, ts(observed_at), status, ts(received_at)),
        )


def setup_readings(db_path):
    # a: +2us and +5us -> min 2us, max 5us, arithmetic mean 3.5us.
    insert_reading(
        db_path, source="a", seq=1, status="published",
        observed_at="2026-01-01T10:00:00Z",
        received_at="2026-01-01T10:00:00.000002Z",
    )
    insert_reading(
        db_path, source="a", seq=2, status="buffered",
        observed_at="2026-01-01T10:00:05Z",
        received_at="2026-01-01T10:00:05.000005Z",
    )
    # b: +1s and -2s -> negative minimum/maximum are retained; mean -0.5s.
    insert_reading(
        db_path, source="b", seq=1, status="buffered",
        observed_at="2026-01-01T10:00:10Z",
        received_at="2026-01-01T10:00:11Z",
    )
    insert_reading(
        db_path, source="b", seq=2, status="quarantined",
        observed_at="2026-01-01T10:00:20Z",
        received_at="2026-01-01T10:00:18Z",
    )


def source_map(body):
    return {s["source_id"]: {k: v for k, v in s.items() if k != "source_id"}
            for s in body["sources"]}


class TestArrivalLatencyProfile:
    def test_groups_all_statuses_with_statistics_and_stable_order(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        body = profile(c, sid).json()
        assert body == {
            "session_id": sid,
            "source_id": None,
            "from": None,
            "to": None,
            "sources": [
                {
                    "source_id": "a",
                    "sample_count": 2,
                    "min_latency_seconds": pytest.approx(0.000002),
                    "max_latency_seconds": pytest.approx(0.000005),
                    "avg_latency_seconds": pytest.approx(0.0000035),
                },
                {
                    "source_id": "b",
                    "sample_count": 2,
                    "min_latency_seconds": -2.0,
                    "max_latency_seconds": 1.0,
                    "avg_latency_seconds": -0.5,
                },
            ],
        }
        # The declared-but-empty source is omitted; source order is source_id.
        assert [s["source_id"] for s in body["sources"]] == ["a", "b"]

    def test_negative_latency_is_preserved_for_min_max_and_mean(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        stats = source_map(profile(c, sid).json())["b"]
        assert stats["min_latency_seconds"] == -2.0
        assert stats["max_latency_seconds"] == 1.0
        assert stats["avg_latency_seconds"] == -0.5

    def test_microsecond_precision_is_exact(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        stats = source_map(profile(c, sid).json())["a"]
        assert stats["min_latency_seconds"] == pytest.approx(0.000002)
        assert stats["max_latency_seconds"] == pytest.approx(0.000005)
        assert stats["avg_latency_seconds"] == pytest.approx(0.0000035)

    def test_source_filter(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        r = profile(c, sid, source_id="b")
        assert r.status_code == 200
        body = r.json()
        assert body["source_id"] == "b"
        assert [s["source_id"] for s in body["sources"]] == ["b"]
        stats = body["sources"][0]
        assert stats["sample_count"] == 2
        assert stats["min_latency_seconds"] == -2.0
        assert stats["max_latency_seconds"] == 1.0
        assert stats["avg_latency_seconds"] == -0.5

    def test_source_with_no_samples_is_omitted(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        body = profile(c, sid, source_id="empty").json()
        assert body["sources"] == []

    def test_inclusive_event_time_window(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        # Endpoints are included: a#2 @10:00:05 and b#1 @10:00:10 only.
        body = profile(
            c,
            sid,
            **{"from": "2026-01-01T10:00:05Z", "to": "2026-01-01T10:00:10Z"},
        ).json()
        assert body["from"] == "2026-01-01T10:00:05Z"
        assert body["to"] == "2026-01-01T10:00:10Z"
        stats = source_map(body)
        assert stats["a"]["sample_count"] == 1
        assert stats["a"]["max_latency_seconds"] == pytest.approx(0.000005)
        assert stats["b"]["sample_count"] == 1
        assert stats["b"]["max_latency_seconds"] == 1.0

    def test_from_only_and_to_only_are_inclusive(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        body = profile(c, sid, **{"from": "2026-01-01T10:00:10Z"}).json()
        assert [s["source_id"] for s in body["sources"]] == ["b"]
        assert body["sources"][0]["sample_count"] == 2

        body = profile(c, sid, **{"to": "2026-01-01T10:00:10Z"}).json()
        stats = source_map(body)
        assert stats["a"]["sample_count"] == 2
        assert stats["b"]["sample_count"] == 1

    def test_time_bounds_normalise_offsets(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        body = profile(
            c,
            sid,
            **{"from": "2026-01-01T12:00:05+02:00", "to": "2026-01-01T12:00:10+02:00"},
        ).json()
        assert body["from"] == "2026-01-01T10:00:05Z"
        assert body["to"] == "2026-01-01T10:00:10Z"
        assert {s["source_id"] for s in body["sources"]} == {"a", "b"}

    def test_unknown_session_404(self, client):
        c, _ = client
        assert profile(c, "missing").status_code == 404

    def test_unknown_or_blank_source_400(self, client):
        c, _ = client
        sid = create_session(c)
        assert profile(c, sid, source_id="unknown").status_code == 400
        assert profile(c, sid, source_id=" ").status_code == 400

    def test_bad_time_bounds_400(self, client):
        c, _ = client
        sid = create_session(c)
        assert profile(c, sid, **{"from": "not-a-time"}).status_code == 400
        assert profile(c, sid, **{"to": "2026-13-99T00:00:00Z"}).status_code == 400

    def test_from_after_to_400(self, client):
        c, _ = client
        sid = create_session(c)
        r = profile(
            c,
            sid,
            **{"from": "2026-01-01T10:00:10Z", "to": "2026-01-01T10:00:00Z"},
        )
        assert r.status_code == 400


class TestArrivalLatencyReadOnly:
    def test_calls_change_nothing(self, client):
        c, db_path = client
        sid = create_session(c)
        setup_readings(db_path)

        def snapshot():
            with sqlite3.connect(db_path) as raw:
                raw.row_factory = sqlite3.Row
                return {
                    name: [dict(r) for r in raw.execute(f"SELECT * FROM {name}")]
                    for name in ("sessions", "session_sources", "readings")
                }

        before = snapshot()
        for _ in range(3):
            assert profile(c, sid).status_code == 200
            assert profile(
                c,
                sid,
                source_id="a",
                **{"from": "2026-01-01T10:00:00Z", "to": "2026-01-01T10:01:00Z"},
            ).status_code == 200
        assert snapshot() == before
