import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from state import create_default_state
from tracking import SnapshotStore


def _default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError


class SnapshotStoreTests(unittest.TestCase):
    def test_records_full_panels_deduplicates_and_deletes_visitor(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(Path(directory) / "planner_snapshots.json")
            state_a = create_default_state()
            state_b = create_default_state()
            store.record_snapshot(
                visitor_id="visitor", tab_id="tab", reason="open", path="/",
                state_a=state_a, state_b=state_b,
            )
            store.record_snapshot(
                visitor_id="visitor", tab_id="tab", reason="same", path="/mode",
                state_a=state_a, state_b=state_b,
            )

            rows = store.list_snapshots()
            self.assertEqual(store.count_snapshots(), 1)
            self.assertEqual(rows[0]["reason"], "same")
            self.assertEqual(len(rows[0]["panel_a"]["models"]), len(state_a.models))
            self.assertIsNotNone(rows[0]["panel_b"])
            self.assertEqual(store.delete_visitor("visitor"), 1)
            self.assertEqual(store.count_snapshots(), 0)

    def test_imports_legacy_json_once_without_reducing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "planner_snapshots.json"
            panel = asdict(create_default_state())
            legacy.write_text(json.dumps({
                "version": 1,
                "visitors": {"v": {"tabs": {"t": {"snapshots": [{
                    "snapshot_id": "legacy-id", "created_at": "2026-01-01T00:00:00+00:00",
                    "last_seen": "2026-01-01T00:00:00+00:00", "reason": "open", "path": "/",
                    "state_hash": "hash", "panel_a": panel, "panel_b": None,
                    "summary": {"mode": panel["mode"], "compare_enabled": False},
                }]}}}},
            }, default=_default), encoding="utf-8")

            store = SnapshotStore(legacy)
            rows = store.list_snapshots()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["snapshot_id"], "legacy-id")
            self.assertEqual(rows[0]["panel_a"], json.loads(json.dumps(panel, default=_default)))
            self.assertTrue(legacy.exists())
            self.assertEqual(SnapshotStore(legacy).count_snapshots(), 1)

    def test_corrupt_legacy_json_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "planner_snapshots.json"
            legacy.write_text("{not json", encoding="utf-8")
            store = SnapshotStore(legacy)

            self.assertEqual(store.list_snapshots(), [])
            self.assertFalse(legacy.exists())
            self.assertEqual(len(list(Path(directory).glob("planner_snapshots.corrupt-*.json"))), 1)

    def test_pagination_and_configured_per_tab_retention(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PLANNER_SNAPSHOT_MAX_PER_TAB": "2"}
        ):
            store = SnapshotStore(Path(directory) / "snapshots.sqlite3")
            state = create_default_state()
            for index in range(3):
                state.task_il += 1
                store.record_snapshot(
                    visitor_id="v", tab_id="t", reason=str(index), path="/",
                    state_a=state, state_b=None,
                )
            self.assertEqual(store.count_snapshots(), 2)
            self.assertEqual(len(store.list_snapshots(limit=1, offset=1)), 1)


if __name__ == "__main__":
    unittest.main()
