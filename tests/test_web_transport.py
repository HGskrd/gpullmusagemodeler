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


if __name__ == "__main__":
    unittest.main()
