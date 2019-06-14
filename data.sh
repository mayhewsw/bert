set -e

# Run this script on the nlpgrid

export BERT_BASE_DIR=/nlp/data/mayhew/bert_stuff/uzugen
SHARD_DIR=$BERT_BASE_DIR/shards

# IMPORTANT: if training on multiple languages, this input file needs to be document shuffled.
# Use shuffle_doc.sh
PRC_DATA_FPATH=/nlp/data/mayhew/bert_stuff/uzugen-shuf.txt

mkdir -p $SHARD_DIR
split -a 4 -l 25000 -d $PRC_DATA_FPATH $SHARD_DIR/shard_

mkdir -p $BERT_BASE_DIR/data

for f in $SHARD_DIR/*;
do
    # convert each shard to tf_record in a massively parallel computation
    qsub -v INPUT=$f,BERT_BASE_DIR=$BERT_BASE_DIR  -N $(basename $f) queue_data.sh
done

#rm -rf $SHARD_DIR
