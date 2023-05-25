# WisdomWeaver

This is a document question answering implementation that combines several concepts together to
produce an arbitrary document question answering machine.

- Sentence Transformers: used to index text into a vector database
- Question Answering Model: used to find answers in text most likely to contain them
- ChatGPT: used to post process the answers

## Getting started

```
pip3 install wisdomweaver
```

Index a directory of asciidoc files:

```
wisdom -v index --path /path/to/data
```

Search the index in interactive mode:

```
wisdom ask "What practice did many ancient C projects use?" --max 5 --threshold 0.2 --interactive --gpt
```

## License

Apache 2.0

## Learn Mode

- Learn techniques used in this project: https://swedishembedded.com/tag/gpt/
