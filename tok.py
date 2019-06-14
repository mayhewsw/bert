#!/usr/bin/python
import sys
import tokenization

fname = sys.argv[1]

tokenizer = tokenization.BasicTokenizer(do_lower_case=False)
outfname = fname + ".tok"
with open(fname) as f, open(outfname, "w") as out:
    for line in f:
        outline = " ".join(tokenizer.tokenize(line)) 
        out.write(outline + "\n")


