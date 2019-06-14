#!/usr/bin/python
import tokenization
import sys

fname = sys.argv[1]
vocab_file = sys.argv[2]

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=False)

with open(fname) as f:
    for line in f:
        if len(line.strip()) > 0:
            tokens = tokenizer.tokenize(line.strip())
            print(line.strip())
            print(tokens)
            print()
