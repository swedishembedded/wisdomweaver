"""
Tests for the automator API
"""
import sys
import textwrap

sys.path.append("../")

from wisdomweaver import Document  # noqa: E402


def test_document_split_paragraphs_on_newlines():
    """
    Verify that splitting paragraphs works as i
    """
    text = """
    This is a sample text that has multiple paragraphs. We have a few sentences
    per paragraph and the idea is that the resulting split will provide us with
    individual paragraphs.

    Small paragraphs should be included as well.

    Another paragraph is here and we simply add a little more text."""
    result = [
        """This is a sample text that has multiple paragraphs. We have a
few sentences per paragraph and the idea is that the resulting split will
provide us with individual paragraphs.""",
        "Small paragraphs should be included as well.",
        "Another paragraph is here and we simply add a little more text.",
    ]
    doc = Document(text)
    paragraphs = doc.paragraphs
    assert len(paragraphs) == 3
    for idx, p in enumerate(doc.paragraphs):
        assert textwrap.wrap(p.raw) == textwrap.wrap(result[idx])
