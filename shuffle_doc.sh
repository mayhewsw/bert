# This script takes two arguments:
#  INPUT: a text file with one sentence per line, and blank lines indicating new documents
#  OUTPUT: a text file with document structure intact (with a document, the sentences have the same order), but documents shuffled.

INPUT=$1
OUTPUT=$2

F=/tmp/shuf_${RANDOM}
cat $INPUT | awk '!NF{$0="@@DOCSEP@@"}1' | awk '$1=$1' ORS='@@NEWLINE@@' > $F
cat ${F} | sed 's/@@DOCSEP@@@@NEWLINE@@/\n/g' | shuf > ${F}.1

cat ${F}.1 | sed 's/@@NEWLINE@@/\n/g' > $OUTPUT
