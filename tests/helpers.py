"""Test helpers shared across unit and integration tests."""

from __future__ import annotations

from codeeval.datasets.schema import Difficulty, Task, TaskTest

ADD_PROMPT = 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n'
ADD_SOLUTION = ADD_PROMPT + "    return a + b\n"


def make_task(
    task_id: str = "demo/add",
    *,
    prompt: str = ADD_PROMPT,
    entry_point: str = "add",
    solution: str = ADD_SOLUTION,
    tests: tuple[TaskTest, ...] | None = None,
    tags: tuple[str, ...] = ("math",),
    difficulty: Difficulty = Difficulty.EASY,
    test_setup: str = "",
) -> Task:
    return Task(
        task_id=task_id,
        prompt=prompt,
        entry_point=entry_point,
        reference_solution=solution,
        tests=tests
        or (
            TaskTest("small", "assert add(1, 2) == 3"),
            TaskTest("negative", "assert add(-1, -1) == -2"),
            TaskTest("zero", "assert add(0, 0) == 0"),
        ),
        tags=tags,
        difficulty=difficulty,
        test_setup=test_setup,
    )


TINY_TASKS = [
    {
        "task_id": "tiny/add",
        "prompt": ADD_PROMPT,
        "entry_point": "add",
        "reference_solution": ADD_SOLUTION,
        "tests": [
            {"name": "small", "code": "assert add(1, 2) == 3"},
            {"name": "negative", "code": "assert add(-1, -1) == -2"},
            {"name": "zero", "code": "assert add(0, 0) == 0"},
        ],
        "tags": ["math"],
        "difficulty": "easy",
    },
    {
        "task_id": "tiny/reverse",
        "prompt": 'def reverse_words(text: str) -> str:\n    """Reverse word order."""\n',
        "entry_point": "reverse_words",
        "reference_solution": 'def reverse_words(text: str) -> str:\n    """Reverse word order."""\n'
        '    return " ".join(reversed(text.split()))\n',
        "tests": [
            {"name": "two", "code": 'assert reverse_words("a b") == "b a"'},
            {"name": "one", "code": 'assert reverse_words("x") == "x"'},
            {"name": "empty", "code": 'assert reverse_words("") == ""'},
        ],
        "tags": ["strings"],
        "difficulty": "easy",
    },
    {
        "task_id": "tiny/max-gap",
        "prompt": 'def max_gap(values: list[int]) -> int:\n    """Largest difference between neighbours after sorting (0 for < 2 values)."""\n',
        "entry_point": "max_gap",
        "reference_solution": 'def max_gap(values: list[int]) -> int:\n    """Largest difference between neighbours after sorting (0 for < 2 values)."""\n'
        "    ordered = sorted(values)\n"
        "    return max((b - a for a, b in zip(ordered, ordered[1:])), default=0)\n",
        "tests": [
            {"name": "basic", "code": "assert max_gap([1, 9, 3]) == 6"},
            {"name": "single", "code": "assert max_gap([4]) == 0"},
            {"name": "dupes", "code": "assert max_gap([2, 2, 2]) == 0"},
            {"name": "weighted", "code": "assert max_gap([10, 0]) == 10", "weight": 2.0},
        ],
        "tags": ["lists", "math"],
        "difficulty": "medium",
    },
]
