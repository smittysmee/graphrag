"""The advisory sync lock: one persona, one live `graphrag sync` at a time.

Two `sync` runs against the same persona can both re-ingest the same source concurrently,
competing for CPU for hours and risking a half-written document if one is killed mid-run.
Nothing else on disk says a run is already under way, so `acquire_sync_lock` is the thing that
notices and refuses. Everything here runs against a temp directory; no store or embedder is
needed since the lock never touches the graph.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from graphrag.sync import (
    SyncLock,
    SyncLockError,
    acquire_sync_lock,
    read_sync_lock,
    refresh_sync_lock,
    release_sync_lock,
)

PERSONA = "test-persona"
COMMAND = "graphrag sync test-persona"


@pytest.fixture
def lock_dir(tmp_path: Path) -> Path:
    return tmp_path / "locks"


def test_a_fresh_lock_can_be_acquired_and_read_back(lock_dir: Path) -> None:
    lock = acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
    assert lock.persona_id == PERSONA
    assert lock.command == COMMAND
    assert lock.started_at == lock.heartbeat_at

    on_disk = read_sync_lock(lock_dir, PERSONA)
    assert on_disk == lock


def test_release_removes_the_lock_file(lock_dir: Path) -> None:
    acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
    release_sync_lock(lock_dir, PERSONA)
    assert read_sync_lock(lock_dir, PERSONA) is None


def test_release_of_a_lock_that_was_never_taken_is_a_no_op(lock_dir: Path) -> None:
    release_sync_lock(lock_dir, PERSONA)  # must not raise
    assert read_sync_lock(lock_dir, PERSONA) is None


def test_a_second_run_against_a_live_lock_is_refused(lock_dir: Path) -> None:
    acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
    other_command = "graphrag sync test-persona --source docs"
    with pytest.raises(SyncLockError) as excinfo:
        acquire_sync_lock(lock_dir, PERSONA, other_command, stale_after_seconds=900)
    message = str(excinfo.value)
    assert PERSONA in message
    assert "--force-lock" in message
    assert excinfo.value.lock.command == COMMAND


class BoomError(Exception):
    """A stand-in for whatever a sync run might raise mid-ingest."""


def test_exiting_via_exception_still_releases_the_lock(lock_dir: Path) -> None:
    try:
        acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
        raise BoomError
    except BoomError:
        pass
    finally:
        release_sync_lock(lock_dir, PERSONA)
    assert read_sync_lock(lock_dir, PERSONA) is None


def test_a_stale_lock_is_reported_and_replaced(lock_dir: Path) -> None:
    old = datetime.now(UTC) - timedelta(hours=2)
    stale = SyncLock(
        persona_id=PERSONA,
        started_at=old.isoformat(),
        heartbeat_at=old.isoformat(),
        command="graphrag sync test-persona --refresh-attribution",
    )
    lock_dir.mkdir(parents=True)
    (lock_dir / f"{PERSONA}.lock.json").write_text(json.dumps(stale.to_json()), encoding="utf-8")

    seen: list[str] = []
    new_lock = acquire_sync_lock(
        lock_dir, PERSONA, COMMAND, stale_after_seconds=900, progress=seen.append
    )
    assert new_lock.command == COMMAND
    assert new_lock.started_at != stale.started_at
    assert any("abandoned" in line for line in seen)

    on_disk = read_sync_lock(lock_dir, PERSONA)
    assert on_disk == new_lock


def test_a_lock_within_the_staleness_window_is_not_replaced(lock_dir: Path) -> None:
    recent = datetime.now(UTC) - timedelta(seconds=30)
    live = SyncLock(
        persona_id=PERSONA,
        started_at=recent.isoformat(),
        heartbeat_at=recent.isoformat(),
        command=COMMAND,
    )
    lock_dir.mkdir(parents=True)
    (lock_dir / f"{PERSONA}.lock.json").write_text(json.dumps(live.to_json()), encoding="utf-8")
    with pytest.raises(SyncLockError):
        acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)


def test_force_lock_takes_over_a_live_lock(lock_dir: Path) -> None:
    acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
    seen: list[str] = []
    forced = acquire_sync_lock(
        lock_dir,
        PERSONA,
        "graphrag sync test-persona --force-lock",
        stale_after_seconds=900,
        force=True,
        progress=seen.append,
    )
    assert forced.command == "graphrag sync test-persona --force-lock"
    assert any("--force-lock" in line for line in seen)


def test_refresh_bumps_heartbeat_without_changing_started_at(lock_dir: Path) -> None:
    lock = acquire_sync_lock(lock_dir, PERSONA, COMMAND, stale_after_seconds=900)
    # Force the original heartbeat into the past so a refresh is observably different.
    backdated = SyncLock(
        persona_id=lock.persona_id,
        started_at=lock.started_at,
        heartbeat_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
        command=lock.command,
    )
    refreshed = refresh_sync_lock(lock_dir, backdated)
    assert refreshed.started_at == lock.started_at
    assert refreshed.heartbeat_at != backdated.heartbeat_at
    assert refreshed.age_seconds() < 5

    on_disk = read_sync_lock(lock_dir, PERSONA)
    assert on_disk == refreshed


def test_reading_a_missing_lock_returns_none(lock_dir: Path) -> None:
    assert read_sync_lock(lock_dir, PERSONA) is None


def test_reading_a_corrupt_lock_file_returns_none_rather_than_raising(lock_dir: Path) -> None:
    lock_dir.mkdir(parents=True)
    (lock_dir / f"{PERSONA}.lock.json").write_text("not json", encoding="utf-8")
    assert read_sync_lock(lock_dir, PERSONA) is None


def test_is_stale_respects_the_staleness_window(lock_dir: Path) -> None:
    now = datetime.now(UTC)
    lock = SyncLock(
        persona_id=PERSONA,
        started_at=now.isoformat(),
        heartbeat_at=(now - timedelta(minutes=10)).isoformat(),
        command=COMMAND,
    )
    assert not lock.is_stale(900, now=now)  # 10 minutes < 15 minute default window
    assert lock.is_stale(300, now=now)  # but past a 5 minute window
