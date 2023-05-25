import sys
import textwrap

sys.path.append("../")

from wisdomweaver import Database  # noqa: E402
from wisdomweaver import Document  # noqa: E402

TEST_DATABASE = ".database-test"
COLLECTION = "test-database"


def test_database_can_return_similar_sentences():
    db = Database(TEST_DATABASE)
    text = """
This document describes how to index text with Python using sentence
transformers.

A cat sits on a tree and codes Python but is not aware of any sentence
transformers.

The dog barks at a cat and is very interested in making sure that the cat comes
down from the tree.
"""
    docs = []
    docs.append(Document(text))

    db.index_documents(COLLECTION, docs)

    results = db.search(
        COLLECTION,
        """a large animal with four legs is interested
in a cat""",
        3,
    )

    assert textwrap.wrap(results[0].payload["text"]) == textwrap.wrap(
        """The dog barks at a cat and is very interested in
making sure that the cat comes down from the tree."""
    )

    results = db.search(COLLECTION, """this written text describes""", 3)

    assert textwrap.wrap(results[0].payload["text"]) == textwrap.wrap(
        """This
document describes how to index text with Python using sentence
transformers."""
    )
