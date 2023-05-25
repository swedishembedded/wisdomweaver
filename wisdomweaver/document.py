import re
import pypdf

from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import docutils.nodes
from docutils.parsers.rst import Parser, roles
from docutils.utils import new_document
from docutils.frontend import OptionParser
from docutils import io


class Paragraph:
    def __init__(self, raw_text):
        self.original = raw_text

    @property
    def preprocessed(self):
        return self.original
        text = self.original
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back into a single string
        preprocessed_text = " ".join(tokens)
        return preprocessed_text

    @property
    def raw(self):
        """
        Returns raw original string
        """
        return self.original


class Text:
    def __init__(self, text):
        self.text = text


def ignore_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Ignore specific reStructuredText roles."""
    node = docutils.nodes.Text(text, rawtext)
    return [node], []


class Document:
    def __init__(self, source):
        self.min_length = 20
        self.paragraph_list = []
        if isinstance(source, Path):
            self.load(source)
        else:
            self.add_text(source)

    def load_rst(self, rst_data):
        # Prepare settings
        option_parser = OptionParser(components=(Parser,))
        settings = option_parser.get_default_values()
        settings.report_level = 5  # Only severe errors
        settings.halt_level = 5  # Only severe errors
        settings.warning_stream = io.NullOutput()  # Redirect warnings and errors to null

        # Register the role handler
        roles.register_local_role("ref", ignore_role)
        # Add here more roles to ignore if needed, e.g.
        # roles.register_local_role('another_role_to_ignore', ignore_role)

        # Initialize the document
        document = new_document("file", settings)

        # Create the RST parser and parse the RST data
        parser = Parser()
        parser.parse(rst_data, document)

        # Extract paragraphs and bullet point lists
        for node in document.traverse():
            if isinstance(node, docutils.nodes.paragraph):
                self.add_paragraph(node.astext())

    def load(self, path):
        """
        Loads a document from file
        """
        if path.suffix == ".pdf":
            with open(str(path), "rb") as file:
                reader = pypdf.PdfReader(file)
                pages = []
                for page_num in range(len(reader.pages)):
                    pages.extend(reader.pages[page_num].extract_text())
                self.add_text("\n".join(pages))
        elif path.suffix == ".rst":
            with open(str(path), "r") as file:
                self.load_rst(str(file.read()))
        elif path.suffix == ".adoc":
            with open(str(path), "r") as file:
                self.add_text(str(file.read()))
        else:
            raise Exception("Unknown file suffix '%s'" % (path.suffix))

    @property
    def paragraphs(self):
        """
        Returns paragraphs of the previously loaded document
        """
        return self.paragraph_list

    def add_paragraph(self, cleaned_paragraph):
        # Discard paragraphs that are too short or contain only numbers/punctuation
        if len(cleaned_paragraph) < self.min_length:
            return
        if re.fullmatch(r"[^\w\s]*", cleaned_paragraph):
            return

        self.paragraph_list.append(Paragraph(cleaned_paragraph))

    def add_text(self, text):
        """
        This method preprocesses text and stores it into the internal paragraphs
        array that is accessible using the @prop paragraphs.
        """
        # Split text into paragraphs using blank lines as delimiters
        raw_paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in raw_paragraphs:
            # Remove extra whitespace and line breaks
            cleaned_paragraph = " ".join(paragraph.split())
            self.add_paragraph(cleaned_paragraph)
