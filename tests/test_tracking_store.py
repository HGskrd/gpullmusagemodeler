import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from data import PROJECT_PRESETS
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
                visitor_id="visitor",
                tab_id="tab",
                reason="open",
                path="/",
                state_a=state_a,
                state_b=state_b,
            )
            store.record_snapshot(
                visitor_id="visitor",
                tab_id="tab",
                reason="same",
                path="/mode",
                state_a=state_a,
                state_b=state_b,
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
            legacy.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "visitors": {
                            "v": {
                                "tabs": {
                                    "t": {
                                        "snapshots": [
                                            {
                                                "snapshot_id": "legacy-id",
                                                "created_at": "2026-01-01T00:00:00+00:00",
                                                "last_seen": "2026-01-01T00:00:00+00:00",
                                                "reason": "open",
                                                "path": "/",
                                                "state_hash": "hash",
                                                "panel_a": panel,
                                                "panel_b": None,
                                                "summary": {
                                                    "mode": panel["mode"],
                                                    "compare_enabled": False,
                                                },
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                    },
                    default=_default,
                ),
                encoding="utf-8",
            )

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

    def test_bounded_retention_defaults_when_env_unset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ) as env:
            env.pop("PLANNER_SNAPSHOT_RETENTION_DAYS", None)
            env.pop("PLANNER_SNAPSHOT_MAX_PER_TAB", None)
            store = SnapshotStore(Path(directory) / "snapshots.sqlite3")
            self.assertEqual(store.retention_days, 90)
            self.assertEqual(store.max_per_tab, 250)

    def test_zero_retention_still_means_unlimited(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"PLANNER_SNAPSHOT_RETENTION_DAYS": "0", "PLANNER_SNAPSHOT_MAX_PER_TAB": "0"},
            ),
        ):
            store = SnapshotStore(Path(directory) / "snapshots.sqlite3")
            self.assertEqual(store.retention_days, 0)
            self.assertEqual(store.max_per_tab, 0)

    def test_snapshot_rows_slim_builtin_use_case_defs_but_restore_on_read(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(Path(directory) / "snapshots.sqlite3")
            state = create_default_state()
            custom_def = {"key": "custom-uc", "label": "Custom", "wtp_per_m": 5.0}
            state.use_case_defs = list(state.use_case_defs) + [custom_def]
            store.record_snapshot(
                visitor_id="visitor",
                tab_id="tab",
                reason="edit",
                path="/",
                state_a=state,
                state_b=None,
            )

            with sqlite3.connect(store.path) as connection:
                raw = json.loads(
                    connection.execute("SELECT panel_a_json FROM snapshots").fetchone()[0]
                )
            stored_keys = {entry["key"] for entry in raw["use_case_defs"]}
            preset_keys = {preset["key"] for preset in PROJECT_PRESETS}
            self.assertEqual(stored_keys, {"custom-uc"})
            self.assertFalse(stored_keys & preset_keys)

            rows = store.list_snapshots()
            restored_keys = {entry["key"] for entry in rows[0]["panel_a"]["use_case_defs"]}
            self.assertEqual(restored_keys, preset_keys | {"custom-uc"})

    def test_pagination_and_configured_per_tab_retention(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"PLANNER_SNAPSHOT_MAX_PER_TAB": "2"}),
        ):
            store = SnapshotStore(Path(directory) / "snapshots.sqlite3")
            state = create_default_state()
            for index in range(3):
                state.task_il += 1
                store.record_snapshot(
                    visitor_id="v",
                    tab_id="t",
                    reason=str(index),
                    path="/",
                    state_a=state,
                    state_b=None,
                )
            self.assertEqual(store.count_snapshots(), 2)
            self.assertEqual(len(store.list_snapshots(limit=1, offset=1)), 1)


if __name__ == "__main__":
    unittest.main()
