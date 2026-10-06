import unittest

from agent import GaiaAgent


class AgentRoutingTests(unittest.TestCase):
    def test_uses_web_for_wikipedia_question(self):
        agent = GaiaAgent()
        self.assertTrue(
            agent._should_use_web(
                "How many studio albums were published by Mercedes Sosa? "
                "You can use the latest 2022 version of english wikipedia."
            )
        )

    def test_skips_web_for_reversed_text_question(self):
        agent = GaiaAgent()
        self.assertFalse(
            agent._should_use_web(
                '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw'
            )
        )

    def test_skips_web_for_table_reasoning_question(self):
        agent = GaiaAgent()
        self.assertFalse(
            agent._should_use_web(
                "Given this table defining * on the set S = {a, b, c, d, e}, "
                "provide the subset involved in counter-examples."
            )
        )


if __name__ == "__main__":
    unittest.main()
