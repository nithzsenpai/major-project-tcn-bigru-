from __future__ import annotations

import unittest

from vanet_detector.download import build_http_session


class DownloadTests(unittest.TestCase):
    def test_zenodo_retry_policy_covers_transient_failures(self) -> None:
        session = build_http_session()
        retry = session.get_adapter("https://").max_retries
        self.assertEqual(retry.total, 5)
        self.assertTrue({429, 500, 502, 503, 504}.issubset(retry.status_forcelist))
        self.assertIn("GET", retry.allowed_methods)


if __name__ == "__main__":
    unittest.main()
