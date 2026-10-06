import unittest

from local_eval import categorize_task, local_status


class LocalEvalTests(unittest.TestCase):
    def test_categorizes_attachment_tasks(self):
        self.assertEqual(categorize_task("Review image", "board.png"), "image_attachment")
        self.assertEqual(categorize_task("Listen", "audio.mp3"), "audio_attachment")
        self.assertEqual(categorize_task("Total sales", "sales.xlsx"), "spreadsheet_attachment")
        self.assertEqual(categorize_task("Output", "code.py"), "python_attachment")

    def test_categorizes_web_and_reasoning_tasks(self):
        self.assertEqual(
            categorize_task("Use English Wikipedia to answer", None),
            "web_research",
        )
        self.assertEqual(
            categorize_task("What is the opposite of left?", None),
            "reasoning",
        )

    def test_local_status(self):
        self.assertEqual(local_status("", "RuntimeError('x')"), "error")
        self.assertEqual(local_status("", ""), "blank")
        self.assertEqual(local_status("Tool error: missing", ""), "tool_error")
        self.assertEqual(local_status("4", ""), "answered_unverified")


if __name__ == "__main__":
    unittest.main()
