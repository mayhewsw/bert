set -e

export BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uzugen
SHARD_DIR=$BERT_BASE_DIR/shards

# This needs vocab.txt, created by mkvocab.sh

# Run this script on the nlpgrid
mkdir -p $BERT_BASE_DIR/data

rm -rf $SHARD_DIR/*.tok

for f in $SHARD_DIR/*;
do
    # convert each shard to tf_record in a massively parallel computation
    qsub -v INPUT=$f,BERT_BASE_DIR=$BERT_BASE_DIR  -N $(basename $f) queue_data.sh
done


