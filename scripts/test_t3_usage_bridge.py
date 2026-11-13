import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bridge", Path(__file__).with_name("t3_usage_bridge.py"))
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def test_exports_only_counters_and_timing_and_does_not_rewrite_unchanged_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            root = source / "profile/antigravity-acp"
            transcript = root / "brain/session-1/.system_generated/logs/transcript_full.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text("\n".join(json.dumps(row) for row in [
                {"step_index": 0, "type": "USER_INPUT", "content": "PRIVATE PROMPT"},
                {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "RUNNING", "output_tokens": 1},
                {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "content": "PRIVATE RESPONSE",
                 "input_tokens": 80, "output_tokens": 12, "cache_read_tokens": 20, "created_at": "2026-11-13T12:00:00Z"},
                {"step_index": 2, "type": "PLANNER_RESPONSE", "status": "RUNNING", "output_tokens": 5},
            ]) + "\n{")
            (root / "conversations").mkdir()
            duration = bridge.encode(8) + bridge.encode(2)
            response = bridge.blob(11, duration) + bridge.blob(12, duration) + bridge.blob(21, b"gemini-test-high")
            response += bridge.blob(99, b"PRIVATE NATIVE RESPONSE")
            with sqlite3.connect(root / "conversations/session-1.db") as db:
                db.executescript("CREATE TABLE steps (idx INTEGER,step_type INTEGER,metadata BLOB);"
                                 "CREATE TABLE gen_metadata (idx INTEGER,data BLOB);")
                db.execute("INSERT INTO steps VALUES (1,15,?)", (bridge.blob(20, b""),))
                db.execute("INSERT INTO gen_metadata VALUES (0,?)", (bridge.blob(1, response),))
            self.assertEqual(bridge.sync(source, target), (1, 1))
            exported = target / "brain/session-1/.system_generated/logs/transcript_full.jsonl"
            rows = [json.loads(line) for line in exported.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["output_tokens"], 12)
            self.assertEqual(rows[0]["cache_read_tokens"], 20)
            self.assertEqual(rows[0]["model"], "gemini-test-high")
            for path in target.rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"PRIVATE", path.read_bytes())
            modified = exported.stat().st_mtime_ns
            self.assertEqual(bridge.sync(source, target), (1, 0))
            self.assertEqual(exported.stat().st_mtime_ns, modified)
            with sqlite3.connect(target / "conversations/session-1.db") as db:
                data = db.execute("SELECT data FROM gen_metadata").fetchone()[0]
                self.assertEqual(set(bridge.fields(bridge.fields(data)[1])), {11, 12, 21})


if __name__ == "__main__":
    unittest.main()
