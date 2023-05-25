# __main__.py

import argparse
import textwrap
import guidance

from pathlib import Path

from wisdomweaver import Document, Database
from transformers import pipeline

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")

class Style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

postprocess_program = guidance(
    """
{{#user~}}
Make this text congruent and remove all "Question:", "Answer:" and "Score:" text:

{{query}}
{{~/user}}

{{#assistant~}}
{{gen 'response' temperature=0.9 max_tokens=500}}
{{~/assistant}}
"""
)

parser = argparse.ArgumentParser(
    prog="WisdomWeaver",
    description="Language models for document handling",
    epilog="Learn more at swedishembedded.com",
)

parser.add_argument("-v", "--verbose", action="store_true")

subparsers = parser.add_subparsers(dest="command")

index_parser = subparsers.add_parser("index")
index_parser.add_argument("-d", "--db", default=".database")
index_parser.add_argument("-c", "--collection", default="default")  # option that takes a value
index_parser.add_argument("-p", "--path", default=None)

search_parser = subparsers.add_parser("search")
search_parser.add_argument("-d", "--db", default=".database")
search_parser.add_argument("-c", "--collection", default="default")  # option that takes a value
search_parser.add_argument("query", help="Search query")

ask_parser = subparsers.add_parser("ask")
ask_parser.add_argument("-d", "--db", default=".database")
ask_parser.add_argument("-c", "--collection", default="default")  # option that takes a value
ask_parser.add_argument("--threshold", default=0.4, type=float)
ask_parser.add_argument("--max", default=3, type=int, help="Maximum number of results")
ask_parser.add_argument("--verify", default="")
ask_parser.add_argument("-i", "--interactive", action="store_true")
ask_parser.add_argument("--gpt", action="store_true")
ask_parser.add_argument("--opt", action="store_true")
ask_parser.add_argument("query", default="", nargs='?', help="Search query")

args = parser.parse_args()


def print_info(text):
    if args.verbose:
        print(Style.CYAN + text + Style.RESET)


def index(args):
    if not args.path:
            raise Exception("Indexing expects a path argument")

    db = Database(args.db)
    docs = []
    exts = ["adoc", "pdf", "rst"]
    for ext in exts:
        for file in Path(args.path).rglob(f"*.{ext}"):
            print_info("Found %s" % (file))
            doc = Document(file)
            docs.append(doc)
            if args.verbose:
                for p in doc.paragraphs:
                    print(p.raw)
                    print("---")

    print_info("Indexing %d documents" % (len(docs)))
    db.index_documents(args.collection, docs)

def ask(args):
    if not args.interactive and not args.query:
        raise BaseException("No search query provided")

    db = Database(args.db)
    question_answerer = pipeline("question-answering", model="deepset/roberta-large-squad2")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    while True:
        query = args.query
        if args.interactive:
            query = input(">>> ")

        results = db.search(args.collection, query, args.max)

        items = []
        response = f"Question: {query}\n"
        for i, result in enumerate(results):
            text = "\n".join(textwrap.wrap(result.payload["text"]))
            res = question_answerer(question=query, context=text)
            if res["score"] > args.threshold:
                ans_text = " ".join(textwrap.wrap(res["answer"]))
                response += Style.YELLOW
                response += f"Answer: {ans_text}\n"
                response += Style.RESET
                response += f"Score: {res['score']}\n"
                response += f"Context: {text}\n"

                if args.verify != "":
                    assert args.verify == ans_text, f"Expected: {args.verify}, got: {ans_text}"

        if args.gpt:
            prog = postprocess_program(query=response)
            ans = prog()
            print(Style.YELLOW)
            print(ans)
            print(Style.RESET)
            print(Style.GREEN + "\n".join(textwrap.wrap(ans["response"])) + Style.RESET)
        elif args.opt:
            generator = pipeline(task='text-generation', model="facebook/opt-125m",
                                 max_new_tokens=200)
            res = generator(response, max_length=200)[0]
            print(response)
            print(Style.GREEN + "\n".join(textwrap.wrap(res['generated_text']))+Style.RESET)
        else:
            print(Style.GREEN + response + Style.RESET)
            summary = summarizer(response, min_length=10, max_length=100)
            print(Style.GREEN)
            print("Summary: %s" % ("\n".join(textwrap.wrap(summary[0]["summary_text"]))))
            print(Style.RESET)
        if not args.interactive:
            break

def search(args):
    if not args.query:
        raise BaseException("No search query provided")

    print(f"Query: {args.query}")
    db = Database(args.db)
    results = db.search(args.collection, args.query, 3)

    items = []
    for result in results:
        items.append("\n".join(textwrap.wrap(result.payload["text"])))
    result = "\n".join(items)
    print("Result: %s" % (result))

def main():
    if args.command == "index":
        index(args)
    elif args.command == "ask":
        ask(args)
    elif args.command == "search":
        search(args)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
