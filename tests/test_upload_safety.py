from io import BytesIO
import unittest

from PIL import Image

from upload_safety import prepare_image_upload, sanitize_filename


class UploadSafetyTests(unittest.TestCase):
    @staticmethod
    def valid_png() -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (4, 4), "green").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_valid_image_is_verified_and_mime_comes_from_content(self):
        prepared = prepare_image_upload("crop.jpg", self.valid_png())
        self.assertEqual(prepared["name"], "crop.jpg")
        self.assertEqual(prepared["mime"], "image/png")
        self.assertTrue(prepared["data"])

    def test_corrupt_image_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_image_upload("crop.png", b"\x89PNG\r\n\x1a\nGARBAGE")

    def test_filename_is_reduced_to_a_safe_display_name(self):
        self.assertEqual(sanitize_filename("../../private/crop.png"), "crop.png")
        self.assertEqual(sanitize_filename("..\\..\\crop.webp"), "crop.webp")
        self.assertEqual(sanitize_filename(".."), "crop-photo")

