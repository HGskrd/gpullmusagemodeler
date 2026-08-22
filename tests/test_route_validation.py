"""Malformed request handling for the planner mutation routes.

These routes used to parse form fields with bare int()/float(), so a missing or
non-numeric field surfaced as a 500 carrying the interpreter's message. Bad
input must be a 400 with a readable message, and genuine faults must be a 500
that does not leak internals.
"""

import ast
import unittest
import uuid
from unittest.mock import patch

from app_factory import create_test_app

import web.planner as web_planner
import web.session_store as session_store
from web.config import BASE_DIR


class RouteValidationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()
        self.headers = {"X-Tab-ID": str(uuid.uuid4())}
        self.client.get("/", headers=self.headers)
        scope = next(iter(session_store._states))
        self.gpu_uid = session_store.get_state(scope).gpus[0].uid

    def test_missing_and_malformed_numeric_fields_are_400(self):
        cases = [
            ("/gpu/qty", {"delta": "1"}),
            ("/gpu/qty", {"uid": "abc", "delta": "1"}),
            ("/gpu/qty", {"uid": str(self.gpu_uid), "delta": "xyz"}),
            ("/gpu/qty", {"uid": "", "delta": "1"}),
            ("/gpu/cost", {"uid": str(self.gpu_uid), "value": "nan"}),
            ("/gpu/cost", {"uid": str(self.gpu_uid), "value": "inf"}),
            ("/gpu/remove", {}),
            ("/model/count", {"uid": str(self.gpu_uid), "count": "many"}),
        ]
        for path, data in cases:
            with self.subTest(path=path, data=data):
                response = self.client.post(path, headers=self.headers, data=data)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.get_json()["error"])

    def test_valid_request_still_succeeds(self):
        response = self.client.post(
            "/gpu/qty",
            headers=self.headers,
            data={"uid": str(self.gpu_uid), "delta": "1"},
        )
        self.assertEqual(response.status_code, 200)

    def test_omitted_optional_value_keeps_its_default(self):
        # /gpu/cost historically defaulted a missing value to 0; that must hold.
        response = self.client.post(
            "/gpu/cost", headers=self.headers, data={"uid": str(self.gpu_uid)}
        )
        self.assertEqual(response.status_code, 200)

    def test_unexpected_failure_is_an_opaque_500(self):
        secret = "psycopg://user:hunter2@db.internal/planner"

        with patch.object(web_planner, "change_gpu_qty", side_effect=RuntimeError(secret)):
            response = self.client.post(
                "/gpu/qty",
                headers=self.headers,
                data={"uid": str(self.gpu_uid), "delta": "1"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Unexpected server error.")
        self.assertNotIn(secret.encode(), response.data)

    def _route_owning_sources(self):
        """app.py plus every module that owns routes.

        The handlers moved out of app.py into the web/ blueprints, so scanning
        app.py alone would silently stop checking them.
        """
        paths = [BASE_DIR / "app.py", BASE_DIR / "app.py"]
        paths += sorted((BASE_DIR / "web").glob("*.py"))
        return [(path, path.read_text(encoding="utf-8")) for path in paths]

    def test_route_modules_are_actually_being_scanned(self):
        """Guard the guard: the scan must see the real handlers."""
        sources = self._route_owning_sources()
        names = {path.name for path, _ in sources}
        self.assertLessEqual(
            {"app.py", "planner.py", "api.py", "admin.py", "scenarios.py", "use_cases.py"},
            names,
        )
        decorators = sum(text.count("_bp.route(") for _, text in sources)
        self.assertGreaterEqual(decorators, 58, "route decorators went missing from the scan")

    def test_no_route_still_returns_a_raw_exception_message(self):
        for path, source in self._route_owning_sources():
            with self.subTest(module=path.name):
                self.assertNotIn('jsonify({"error": str(e)}), 500', source)

    def test_no_try_block_has_duplicate_exception_handlers(self):
        for path, source in self._route_owning_sources():
            with self.subTest(module=path.name):
                self._assert_no_duplicate_handlers(source)

    def _assert_no_duplicate_handlers(self, source):
        tree = ast.parse(source)
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            handler_types = [
                ast.dump(handler.type, include_attributes=False)
                if handler.type is not None
                else None
                for handler in node.handlers
            ]
            if len(handler_types) != len(set(handler_types)):
                duplicates.append(node.lineno)

        self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
