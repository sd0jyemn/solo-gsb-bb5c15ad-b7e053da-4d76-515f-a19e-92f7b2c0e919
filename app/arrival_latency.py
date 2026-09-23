"""Read-only per-source arrival-latency profiling.

``GET /sessions/{session_id}/arrival-latency-profile`` groups every stored
reading matching the requested filters by ``source_id`` and reports the
number of samples plus the minimum, maximum and arithmetic mean of each
reading's arrival latency::

    latency = first_received_at - observed_at

Buffered, published and quarantined readings are all measurements that the
service has received, so all three statuses participate.  Because clocks may
disagree or a caller may supply a future event time, negative latencies are
kept exactly instead of being clamped to zero.

Optional ``source_id`` and inclusive ``from`` / ``to`` bounds restrict the
event-time window.  Sources without a matching sample (including sources
with no readings at all) do not produce a group.  The aggregate query and
the session/source validation run in one deferred SQLite transaction, so
the complete response derives from one consistent read snapshot.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from .database import read_transaction
from .errors import BadRequestError
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["arrival_latency_profile"]

# Stored timestamps are fixed-width UTC strings "...HH:MM:SS.ffffffZ":
# the six microsecond digits start at position 20 (1-based 21).
_MICROS_EXPR = (
    "(CAST(strftime('%s', {col}) AS INTEGER) * 1000000"
    " + CAST(substr({col}, 21, 6) AS INTEGER))"
)


def _optional_time(value: Any, field: str) -> Optional[str]:
    """Validate/normalise an optional inclusive event-time bound."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError(f"{field} must be a non-empty ISO 8601 string")
    # TimestampError is mapped to HTTP 400 by the application handler.
    return store_event_time(parse_event_time(value))


def arrival_latency_profile(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    source_id: Any = None,
    from_: Any = None,
    to: Any = None,
) -> dict[str, Any]:
    """Return per-source latency statistics for matching readings.

    Read-only: validation and aggregation use one deferred read transaction
    and therefore one SQLite snapshot.  Malformed parameters raise
    :class:`~app.errors.BadRequestError` / :class:`TimestampError` (400); an
    unknown session raises 404 and an undeclared source raises 400.
    """
    if source_id is not None and (
        not isinstance(source_id, str) or not source_id.strip()
    ):
        raise BadRequestError("source_id must be a non-empty string")

    from_store = _optional_time(from_, "from")
    to_store = _optional_time(to, "to")
    if from_store is not None and to_store is not None and from_store > to_store:
        raise BadRequestError(
            f"'from' ({format_event_time(parse_event_time(from_store))}) must not"
            f" be later than 'to' ({format_event_time(parse_event_time(to_store))})"
        )

    where = "session_id = ?"
    params: list[Any] = [session_id]
    if source_id is not None:
        where += " AND source_id = ?"
        params.append(source_id)
    if from_store is not None:
        where += " AND observed_at >= ?"
        params.append(from_store)
    if to_store is not None:
        where += " AND observed_at <= ?"
        params.append(to_store)

    event_micros = _MICROS_EXPR.format(col="observed_at")
    received_micros = _MICROS_EXPR.format(col="first_received_at")
    sql = f"""
WITH matching_readings AS (
    SELECT source_id,
           ({received_micros}) - ({event_micros}) AS latency_micros
    FROM readings
    WHERE {where}
)
SELECT source_id,
       COUNT(*) AS sample_count,
       MIN(latency_micros) AS min_latency_micros,
       MAX(latency_micros) AS max_latency_micros,
       SUM(latency_micros) AS total_latency_micros
FROM matching_readings
GROUP BY source_id
ORDER BY source_id
"""

    with read_transaction(conn):
        session = _load_session(conn, session_id)
        if source_id is not None:
            allowed_sources = set(json.loads(session["source_ids"]))
            if source_id not in allowed_sources:
                raise BadRequestError(
                    f"unknown source_id {source_id!r} for this session;"
                    f" declared sources: {sorted(allowed_sources)}"
                )
        rows = conn.execute(sql, params).fetchall()

    sources = [
        {
            "source_id": row["source_id"],
            "sample_count": row["sample_count"],
            "min_latency_seconds": row["min_latency_micros"] / 1_000_000,
            "max_latency_seconds": row["max_latency_micros"] / 1_000_000,
            # Integer microseconds are summed in SQLite; divide only once for
            # the arithmetic mean so negative values and microsecond precision
            # remain exact up to the JSON floating-point representation.
            "avg_latency_seconds": (
                row["total_latency_micros"] / row["sample_count"]
            ) / 1_000_000,
        }
        for row in rows
    ]

    return {
        "session_id": session_id,
        "source_id": source_id,
        "from": (
            format_event_time(parse_event_time(from_store))
            if from_store is not None
            else None
        ),
        "to": (
            format_event_time(parse_event_time(to_store))
            if to_store is not None
            else None
        ),
        "sources": sources,
    }
