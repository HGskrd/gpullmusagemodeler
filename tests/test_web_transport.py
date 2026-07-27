import gzip
import unittest
import uuid

import app as app_module


class WebTransportTests(unittest.TestCase):
    def setUp(self):
        self.original_tracking = app_module.TRACKING_ENABLED
        self.original_testing = app_module.app.config["TESTING"]
        app_module.TRACKING_ENABLED = False
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()
        self.headers = {"X-Tab-ID": str(uuid.uuid4())}

    def tearDown(self):
        app_module.TRACKING_ENABLED = self.original_tracking
        app_module.app.config.update(TESTING=self.original_testing)

    def test_large_html_response_is_gzipped_when_accepted(self):
        response = self.client.get(
            "/",
            headers={**self.headers, "Accept-Encoding": "gzip"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        self.assertIn("Accept-Encoding", response.headers["Vary"])
        self.assertIn(b"vLLM", gzip.decompress(response.data))

    def test_identity_response_remains_uncompressed(self):
        response = self.client.get("/", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Content-Encoding", response.headers)
        self.assertIn("Accept-Encoding", response.headers["Vary"])

    def test_static_assets_are_gzipped_and_match_the_identity_body(self):
        # Flask serves static files with direct_passthrough, which the
        # after_request compressor skips; the static view compresses them itself.
        for path in ("/static/vendor/echarts.min.js", "/static/style.css"):
            with self.subTest(path=path):
                compressed = self.client.get(path, headers={"Accept-Encoding": "gzip"})
                identity = self.client.get(path)

                self.assertEqual(compressed.status_code, 200)
                self.assertEqual(compressed.headers["Content-Encoding"], "gzip")
                self.assertIn("Accept-Encoding", compressed.headers["Vary"])
                self.assertNotIn("Content-Encoding", identity.headers)
                self.assertEqual(gzip.decompress(compressed.data), identity.data)
                self.assertLess(len(compressed.data), len(identity.data))

    def test_static_variants_carry_distinct_etags(self):
        compressed = self.client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
        identity = self.client.get("/static/app.js")

        self.assertTrue(compressed.headers["ETag"].endswith('-gzip"'))
        self.assertNotEqual(compressed.headers["ETag"], identity.headers["ETag"])

    def test_static_revalidation_returns_304_for_both_variants(self):
        compressed = self.client.get("/static/app.js", headers={"Accept-Encoding": "gzip"})
        identity = self.client.get("/static/app.js")

        recheck_gzip = self.client.get(
            "/static/app.js",
            headers={"Accept-Encoding": "gzip", "If-None-Match": compressed.headers["ETag"]},
        )
        recheck_identity = self.client.get(
            "/static/app.js", headers={"If-None-Match": identity.headers["ETag"]}
        )

        self.assertEqual(recheck_gzip.status_code, 304)
        self.assertEqual(recheck_identity.status_code, 304)

    def test_missing_and_traversing_static_paths_still_404(self):
        for path in ("/static/nope.js", "/static/../app.py"):
            with self.subTest(path=path):
                response = self.client.get(path, headers={"Accept-Encoding": "gzip"})
                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
