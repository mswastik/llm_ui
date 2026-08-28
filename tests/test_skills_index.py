"""Regression checks for the skills frontmatter parser:

Marketplace skills use YAML block scalars (`description: >` + indented lines).
The line-wise parser used to store the literal '>' as the description, so the
system-prompt skill index shipped several skills with an EMPTY real
description. The parser must join block-scalar continuations into one line and
still handle plain `key: value` frontmatter.

Run: PYTHONPATH=backend:. python tests/test_skills_index.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from tools.skills_tool import _parse_frontmatter  # noqa: E402


class TestSkillFrontmatter(unittest.TestCase):
    def test_plain_description(self):
        meta, body = _parse_frontmatter(
            "---\nname: x\ndescription: one line here\n---\n\nbody text\n"
        )
        self.assertEqual(meta["description"], "one line here")
        self.assertEqual(meta["name"], "x")
        self.assertIn("body text", body)

    def test_folded_block_scalar(self):
        src = (
            "---\nname: y\ndescription: >\n"
            "  Social media analytics and reporting — read native data\n"
            "  honestly and turn it into next actions.\n"
            "license: MIT\n---\n\nbody\n"
        )
        meta, _ = _parse_frontmatter(src)
        self.assertEqual(
            meta["description"],
            "Social media analytics and reporting — read native data honestly and turn it into next actions.",
        )
        self.assertEqual(meta["license"], "MIT")  # key after block still parses

    def test_literal_block_scalar_and_chomp(self):
        src = (
            "---\nname: z\ndescription: |-\n  line one\n  line two\n---\nbody\n"
        )
        meta, _ = _parse_frontmatter(src)
        self.assertEqual(meta["description"], "line one line two")

    def test_empty_value_key(self):
        src = "---\nname: w\ndescription:\nnext: val\n---\nbody\n"
        meta, _ = _parse_frontmatter(src)
        self.assertEqual(meta["description"], "")
        self.assertEqual(meta["next"], "val")


if __name__ == "__main__":
    unittest.main()
