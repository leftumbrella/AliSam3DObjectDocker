"""Regression checks for the standalone FC browser test client."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "test-client.html"
DEFAULT_ENDPOINT = "https://samd-object-duanxppffx.cn-shenzhen.fcapp.run"


class _ClientMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[dict[str, str | None]] = []
        self.remote_assets: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "input":
            self.inputs.append(attributes)
        if tag in {"script", "link"}:
            asset = attributes.get("src") or attributes.get("href")
            if asset and asset.startswith(("http://", "https://", "//")):
                self.remote_assets.append(asset)


class BrowserTestClientContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = CLIENT.read_text(encoding="utf-8")
        cls.parser = _ClientMarkupParser()
        cls.parser.feed(cls.html)

    def test_separate_default_endpoints_and_runtime_routes_are_present(self) -> None:
        self.assertIn(DEFAULT_ENDPOINT, self.html)
        self.assertIn('const DEFAULT_SEGMENT_ENDPOINT = "";', self.html)
        self.assertIn(
            f'const DEFAULT_THREE_D_ENDPOINT = "{DEFAULT_ENDPOINT}";',
            self.html,
        )
        self.assertIn('id="segment-endpoint"', self.html)
        self.assertIn('id="three-d-endpoint"', self.html)
        self.assertIn('"sam3d-segment-endpoint"', self.html)
        self.assertIn('"sam3d-three-d-endpoint"', self.html)
        for route in ("/healthz", "/gpu", "/readyz", "/segment", "/generate"):
            with self.subTest(route=route):
                self.assertIn(f'"{route}"', self.html)
        self.assertNotIn('"/initialize"', self.html)

    def test_lightweight_checks_are_sequential_and_never_initialize_models(self) -> None:
        match = re.search(
            r"async function runSequentialChecks\(\).*?function getSelectedFormat",
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        function_source = match.group(0)
        positions = [
            function_source.index(call)
            for call in (
                'apiRequest("segment", "/healthz"',
                'apiRequest("segment", "/readyz"',
                'apiRequest("threeD", "/healthz"',
                'apiRequest("threeD", "/gpu"',
                'apiRequest("threeD", "/readyz"',
            )
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("\n    runSequentialChecks();", self.html)
        self.assertNotIn("initializeModel", self.html)
        self.assertNotIn("initialize-button", self.html)
        self.assertIn("FC Initializer", self.html)

    def test_client_does_not_collect_cloud_credentials(self) -> None:
        credential_terms = ("accesskey", "secret", "credential", "authorization")
        for input_attributes in self.parser.inputs:
            searchable = " ".join(
                str(input_attributes.get(attribute, ""))
                for attribute in ("id", "name", "placeholder", "autocomplete")
            ).lower()
            self.assertFalse(any(term in searchable for term in credential_terms))
            self.assertNotEqual(input_attributes.get("type"), "password")
        self.assertIn('credentials: "omit"', self.html)

    def test_click_segmentation_is_single_flight_with_latest_state_queued(self) -> None:
        input_ids = {str(item.get("id", "")) for item in self.parser.inputs}
        self.assertIn("image-input", input_ids)
        self.assertNotIn("mask-input", input_ids)
        self.assertIn('name="selection-mode" value="1"', self.html)
        self.assertIn('name="selection-mode" value="0"', self.html)
        self.assertIn('form.append("points", JSON.stringify(points))', self.html)
        self.assertIn('apiRequest("segment", "/segment"', self.html)
        self.assertIn("if (state.segmentBusy) return;", self.html)
        self.assertIn("while (state.segmentQueued)", self.html)
        self.assertIn("state.pointsRevision", self.html)
        self.assertIn("state.maskRevision", self.html)
        self.assertIn("undoPoint", self.html)
        self.assertIn("clearPoints", self.html)

    def test_generated_mask_blob_is_previewed_and_sent_to_three_d(self) -> None:
        self.assertIn('id="mask-overlay"', self.html)
        self.assertIn('id="mask-preview"', self.html)
        self.assertIn("URL.createObjectURL(blob)", self.html)
        self.assertIn('form.append("mask", state.uploads.mask.file, "mask.png")', self.html)
        self.assertIn('apiRequest("threeD", "/generate"', self.html)
        self.assertIn('id="generate-button"', self.html)
        self.assertIn("state.maskRevision === state.pointsRevision", self.html)

    def test_upload_and_download_safety_contract_is_present(self) -> None:
        self.assertIn("20 * 1024 * 1024", self.html)
        self.assertIn("30 * 1024 * 1024", self.html)
        self.assertIn("40_000_000", self.html)
        self.assertIn("image.width !== mask.width", self.html)
        self.assertIn("new FormData()", self.html)
        self.assertIn("URL.revokeObjectURL", self.html)
        self.assertIn("download", self.html)

    def test_page_has_no_remote_runtime_assets(self) -> None:
        self.assertEqual(self.parser.remote_assets, [])

    def test_ui_baseline_avoids_known_regressions(self) -> None:
        self.assertIn('lang="zh-CN"', self.html)
        self.assertIn('name="viewport"', self.html)
        self.assertIn('class="skip-link"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('aria-keyshortcuts="Enter Space ArrowUp ArrowDown ArrowLeft ArrowRight"', self.html)
        self.assertIn('id="undo-point-button"', self.html)
        self.assertIn('id="clear-points-button"', self.html)
        self.assertIn("prefers-reduced-motion", self.html)
        self.assertNotIn("transition: all", self.html)
        self.assertNotIn("—", self.html)


if __name__ == "__main__":
    unittest.main()
