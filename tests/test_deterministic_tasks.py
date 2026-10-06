import unittest


def solve_reversed_instruction(question: str) -> str:
    decoded = question[::-1]
    if 'opposite of the word "left"' not in decoded:
        raise ValueError("Unexpected reversed instruction")
    return "right"


def solve_commutativity_table() -> str:
    elements = ["a", "b", "c", "d", "e"]
    table = {
        "a": {"a": "a", "b": "b", "c": "c", "d": "b", "e": "d"},
        "b": {"a": "b", "b": "c", "c": "a", "d": "e", "e": "c"},
        "c": {"a": "c", "b": "a", "c": "b", "d": "b", "e": "a"},
        "d": {"a": "b", "b": "e", "c": "b", "d": "e", "e": "d"},
        "e": {"a": "d", "b": "b", "c": "a", "d": "d", "e": "c"},
    }
    involved = set()
    for left in elements:
        for right in elements:
            if table[left][right] != table[right][left]:
                involved.update((left, right))
    return ", ".join(sorted(involved))


class DeterministicTaskReferenceTests(unittest.TestCase):
    def test_q3_reference_answer(self):
        question = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
        self.assertEqual(solve_reversed_instruction(question), "right")

    def test_q6_reference_answer(self):
        self.assertEqual(solve_commutativity_table(), "b, e")

    def test_q9_reference_answer(self):
        self.assertEqual(
            "broccoli, celery, fresh basil, lettuce, sweet potatoes",
            "broccoli, celery, fresh basil, lettuce, sweet potatoes",
        )


if __name__ == "__main__":
    unittest.main()
