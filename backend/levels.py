"""
levels.py
---------
Student Level Mode.  The level only changes HOW an answer is explained
(vocabulary, depth, length).  It never changes the verification: every level
is checked against the same knowledge base with the same risk formula.
"""

from typing import Dict, Optional

DEFAULT_LEVEL = 2

LEVELS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "School",
        "instruction": (
            "The reader is a school student (Level 1). Use simple everyday words and "
            "short sentences. Explain any new term in plain language and use one easy "
            "example if it helps. Answer in 3-5 sentences."
        ),
        "length": "3-5 sentences",
    },
    2: {
        "name": "College",
        "instruction": (
            "The reader is a college student (Level 2). Use the correct technical terms "
            "with a brief explanation, and cover the key concepts and the reasoning "
            "behind them. Answer in 4-6 sentences."
        ),
        "length": "4-6 sentences",
    },
    3: {
        "name": "Postgraduate",
        "instruction": (
            "The reader is a postgraduate student (Level 3). Use precise technical "
            "language and assume a solid background. Mention assumptions, nuances and "
            "limitations where they matter. Answer in 5-8 sentences."
        ),
        "length": "5-8 sentences",
    },
}


def normalise_level(level: Optional[int]) -> int:
    return level if level in LEVELS else DEFAULT_LEVEL


def level_instruction(level: Optional[int]) -> str:
    return LEVELS[normalise_level(level)]["instruction"]


def level_length(level: Optional[int]) -> str:
    return LEVELS[normalise_level(level)]["length"]


def level_name(level: Optional[int]) -> str:
    return LEVELS[normalise_level(level)]["name"]
