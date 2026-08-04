import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InterfaceAssetTests(unittest.TestCase):
    def test_all_carousel_images_are_packaged(self):
        hero_dir = ROOT / "static" / "hero"
        expected = {
            "farmer-phone.png",
            "smart-farm-analytics.png",
            "ai-farmer-companion.png",
            "ai-field-guidance.png",
        }
        self.assertEqual({path.name for path in hero_dir.glob("*.png")}, expected)
        for filename in expected:
            size = (hero_dir / filename).stat().st_size
            self.assertGreater(size, 0)
            self.assertLess(size, 2_500_000)

    def test_carousel_uses_component_v2_and_three_second_rotation(self):
        source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("st.components.v2.component", source)
        self.assertNotIn("components.v1", source)
        self.assertIn('"interval_ms": 3000', source)
        self.assertIn("prefers-reduced-motion: reduce", source)
        self.assertIn("motionQuery.matches", source)

    def test_static_serving_and_modern_theme_are_enabled(self):
        with (ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
            config = tomllib.load(config_file)
        self.assertTrue(config["server"]["enableStaticServing"])
        self.assertEqual(config["client"]["showErrorDetails"], "none")
        self.assertEqual(config["theme"]["primaryColor"], "#087A55")
        self.assertEqual(config["theme"]["backgroundColor"], "#FAFCF8")


if __name__ == "__main__":
    unittest.main()

