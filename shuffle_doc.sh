# This script takes two arguments:
#  INPUT: a text file with one sentence per line, and blank lines indicating new documents
#  OUTPUT: a text file with document structure intact (with a document, the sentences have the same order), but documents shuffled.

INPUT=$1
OUTPUT=$2

cat $INPUT | awk '!NF{$0="@@DOCSEP@@"}1' | awk '$1=$1' ORS='@@NEWLINE@@' | sed 's/@@DOCSEP@@@@NEWLINE@@/\n/g' | shuf | sed 's/@@NEWLINE@@/\n/g' > $OUTPUT
