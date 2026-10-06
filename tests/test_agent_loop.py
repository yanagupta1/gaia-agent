import json
import unittest
from dataclasses import dataclass

from agent import GaiaAgent


@dataclass(frozen=True)
class FakeSearchResult:
    title: str
    url: str
    snippet: str = ""


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_prompt, user_prompt, json_mode=False):
        self.calls.append((system_prompt, user_prompt, json_mode))
        return self.responses.pop(0)


class FakeWeb:
    def __init__(self):
        self.searches = []
        self.visits = []

    def search(self, query):
        self.searches.append(query)
        return [
            FakeSearchResult(
                title="Example Result",
                url="https://example.test/page",
                snippet="Useful snippet",
            )
        ]

    def read_page(self, url):
        self.visits.append(url)
        return "Example page text"


class FakeAttachmentTool:
    def __init__(self):
        self.executed = False
        self.analyzed = False

    def summary(self, context):
        return f"Attachment: {context.file_name}"

    def read_text(self, context):
        return "print(123)"

    def execute_python(self, context):
        self.executed = True
        return "Return code: 0\nSTDOUT:\n123\nSTDERR:"

    def analyze_spreadsheet(self, context):
        self.analyzed = True
        return "Sheet: Sheet1\nNumeric column sums:\nsales    10.50"


class AgentLoopTests(unittest.TestCase):
    def test_returns_final_action(self):
        model = FakeModel([json.dumps({"type": "final", "answer": "right"})])
        agent = GaiaAgent(
            model_client=model,
            web_search=FakeWeb(),
            attachment_tool=FakeAttachmentTool(),
        )
        self.assertEqual(agent.run("What is the opposite of left?", "task"), "right")

    def test_executes_web_search_tool(self):
        model = FakeModel(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "web_search",
                        "args": {"query": "Mercedes Sosa albums"},
                    }
                ),
                json.dumps({"type": "final", "answer": "5"}),
            ]
        )
        web = FakeWeb()
        agent = GaiaAgent(
            model_client=model,
            web_search=web,
            attachment_tool=FakeAttachmentTool(),
        )
        self.assertEqual(agent.run("How many albums?", "task"), "5")
        self.assertEqual(web.searches, ["Mercedes Sosa albums"])

    def test_executes_visit_webpage_tool(self):
        model = FakeModel(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "visit_webpage",
                        "args": {"url": "https://example.test/page"},
                    }
                ),
                json.dumps({"type": "final", "answer": "done"}),
            ]
        )
        web = FakeWeb()
        agent = GaiaAgent(
            model_client=model,
            web_search=web,
            attachment_tool=FakeAttachmentTool(),
        )
        self.assertEqual(agent.run("Read this page", "task"), "done")
        self.assertEqual(web.visits, ["https://example.test/page"])

    def test_plain_text_response_is_not_accepted_as_loop_final(self):
        model = FakeModel(
            [
                "This is not JSON",
                json.dumps({"type": "final", "answer": "clean final"}),
            ]
        )
        agent = GaiaAgent(
            model_client=model,
            web_search=FakeWeb(),
            attachment_tool=FakeAttachmentTool(),
        )
        self.assertEqual(agent.run("Answer exactly", "task"), "clean final")

    def test_how_many_final_is_normalized_to_number(self):
        model = FakeModel(
            [
                json.dumps(
                    {
                        "type": "final",
                        "answer": "Mercedes Sosa published 4 studio albums.",
                    }
                )
            ]
        )
        agent = GaiaAgent(
            model_client=model,
            web_search=FakeWeb(),
            attachment_tool=FakeAttachmentTool(),
        )
        self.assertEqual(agent.run("How many studio albums?", "task"), "4")

    def test_executes_python_attachment_tool(self):
        model = FakeModel(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "execute_python_attachment",
                        "args": {},
                    }
                ),
                json.dumps({"type": "final", "answer": "123"}),
            ]
        )
        attachment_tool = FakeAttachmentTool()
        agent = GaiaAgent(
            model_client=model,
            web_search=FakeWeb(),
            attachment_tool=attachment_tool,
        )
        self.assertEqual(agent.run("What is the output?", "task", "code.py"), "123")
        self.assertTrue(attachment_tool.executed)

    def test_executes_spreadsheet_attachment_tool(self):
        model = FakeModel(
            [
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "analyze_spreadsheet",
                        "args": {},
                    }
                ),
                json.dumps({"type": "final", "answer": "$10.50"}),
            ]
        )
        attachment_tool = FakeAttachmentTool()
        agent = GaiaAgent(
            model_client=model,
            web_search=FakeWeb(),
            attachment_tool=attachment_tool,
        )
        self.assertEqual(agent.run("Total sales?", "task", "sales.xlsx"), "$10.50")
        self.assertTrue(attachment_tool.analyzed)


if __name__ == "__main__":
    unittest.main()
