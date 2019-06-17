
export BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uzugen
SHARD_DIR=$BERT_BASE_DIR/shards

VOCAB=$BERT_BASE_DIR/vocab.txt

./qsub-script -N mkvocab -o mkvocab2.out -e mkvocab2.err -pe 24 "source activate anlp8.2; python mkvocab.py ${SHARD_DIR} ${VOCAB}"
